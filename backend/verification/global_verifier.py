
import os
import json
import re
import time
import requests

from pathlib import Path
from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[2]

load_dotenv(ROOT / ".env")

API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_NAME = "gemini-3.5-flash-lite"

API_URL = (
    f"https://generativelanguage.googleapis.com/v1beta/"
    f"models/{MODEL_NAME}:generateContent"
)


class GlobalVerifier:

    def __init__(self):

        if not API_KEY:
            raise ValueError(
                "GEMINI_API_KEY not found. Check your .env file."
            )

        self.session = requests.Session()

        self.headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": API_KEY
        }

        print("Global verifier initialized.")

    def _call_gemini(self, prompt):

        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt}
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0,
                "maxOutputTokens": 2500,
                "responseMimeType": "application/json"
            }
        }

        response = None

        for attempt in range(4):

            try:

                response = self.session.post(
                    API_URL,
                    headers=self.headers,
                    json=payload,
                    timeout=(10, 90)
                )

                if response.status_code in [
                    429, 500, 502, 503, 504
                ]:

                    if attempt < 3:

                        wait_time = 2 ** (attempt + 1)

                        print(
                            f"Gemini HTTP {response.status_code}. "
                            f"Retrying in {wait_time} seconds..."
                        )

                        time.sleep(wait_time)
                        continue

                response.raise_for_status()
                break

            except requests.RequestException:

                if attempt == 3:
                    raise

                wait_time = 2 ** (attempt + 1)

                print(
                    f"Gemini request failed. "
                    f"Retrying in {wait_time} seconds..."
                )

                time.sleep(wait_time)

        response.raise_for_status()

        result = response.json()

        try:
            text = (
                result["candidates"][0]
                ["content"]["parts"][0]["text"]
            )

        except (KeyError, IndexError, TypeError) as error:

            raise ValueError(
                f"Unexpected Gemini response: {result}"
            ) from error

        text = re.sub(
            r"^```(?:json)?\s*|\s*```$",
            "",
            text.strip(),
            flags=re.IGNORECASE
        )

        return json.loads(text)

    def verify_claims(self, claims, evidence):

        if not claims:
            return []

        if not evidence:
            return [
                {
                    "claim_id": item.get("claim_id", "UNKNOWN"),
                    "claim": item.get("claim", ""),
                    "verdict": "INSUFFICIENT",
                    "reasoning": "No evidence was provided.",
                    "supporting_evidence": [],
                    "contradicting_evidence": []
                }
                for item in claims
            ]

        evidence_context = []

        for index, item in enumerate(evidence, start=1):

            evidence_context.append(
                f"""
[E{index}]
Title: {item.get('title', 'Unknown')}
Source: {item.get('source_dataset', 'Unknown')}
Record ID: {item.get('source_record_id', 'Unknown')}
Evidence:
{item.get('text', '')}
"""
            )

        claims_context = []

        for item in claims:

            claims_context.append(
                f"""
Claim ID: {item.get('claim_id', 'UNKNOWN')}
Claim: {item.get('claim', '')}
"""
            )

        prompt = f"""
You are a strict cybersecurity evidence verification system.

Verify every claim independently using ONLY the supplied evidence.

RETRIEVED EVIDENCE:
{"".join(evidence_context)}

CLAIMS:
{"".join(claims_context)}


VERDICT RULES:

1. SUPPORTED: The supplied evidence directly establishes the claim.

2. CONTRADICTED: The supplied evidence directly establishes that the claim is false or that the opposite occurred.

3. INSUFFICIENT: The supplied evidence does not establish whether the claim is true or false.

4. CRITICAL DISTINCTION:
   - Missing evidence is NOT contradiction.
   - Lack of proof is NOT proof that an event did not happen.
   - "No LSASS access evidence is present" does not prove that credentials were not stolen.
   - "Credential dumping was not detected" does not automatically prove that credential dumping did not occur.

5. If a claim asserts an event occurred but the evidence contains no artifact proving it, use INSUFFICIENT unless the evidence explicitly establishes the event did not occur.

6. If a claim asserts that the evidence confirms or proves an event, but the evidence explicitly states that it does not prove that event, use CONTRADICTED.

7. Do not use external knowledge. Evaluate every claim independently.

8. Every SUPPORTED verdict must include at least one valid supporting evidence ID.

9. Every CONTRADICTED verdict must include at least one valid contradicting evidence ID.

10. Never invent evidence identifiers.

Return ONLY valid JSON in this format:

{{
    "results": [
        {{
            "claim_id": "C1",
            "verdict": "SUPPORTED",
            "reasoning": "Evidence-based explanation.",
            "supporting_evidence": ["E1"],
            "contradicting_evidence": []
        }}
    ]
}}

Allowed verdicts:
SUPPORTED, CONTRADICTED, INSUFFICIENT.

Include all supplied claim IDs exactly once.
"""

        print(
            f"\nSending {len(claims)} claims "
            f"in one Gemini verification request..."
        )

        try:

            parsed = self._call_gemini(prompt)

            raw_results = parsed.get("results", [])

            if not isinstance(raw_results, list):
                raise ValueError(
                    "Gemini results must be a list."
                )

            expected_claims = {
                item.get("claim_id"): item
                for item in claims
            }

            valid_evidence_ids = {
                f"E{i}"
                for i in range(1, len(evidence) + 1)
            }

            verified = {}

            for result in raw_results:

                if not isinstance(result, dict):
                    continue

                claim_id = result.get("claim_id")

                if claim_id not in expected_claims:
                    continue

                if claim_id in verified:
                    continue

                verdict = str(
                    result.get("verdict", "")
                ).upper()

                if verdict not in {
                    "SUPPORTED",
                    "CONTRADICTED",
                    "INSUFFICIENT"
                }:
                    continue

                supporting = result.get(
                    "supporting_evidence", []
                )

                contradicting = result.get(
                    "contradicting_evidence", []
                )

                if not isinstance(supporting, list):
                    continue

                if not isinstance(contradicting, list):
                    continue

                if not all(
                    isinstance(ref, str)
                    and ref in valid_evidence_ids
                    for ref in supporting + contradicting
                ):
                    continue

                # A supported claim must cite valid evidence.
                if verdict == "SUPPORTED" and not supporting:
                    continue

                # A contradicted claim must cite valid evidence.
                if verdict == "CONTRADICTED" and not contradicting:
                    continue

                verified[claim_id] = {
                    "claim_id": claim_id,
                    "claim": expected_claims[claim_id].get(
                        "claim", ""
                    ),
                    "verdict": verdict,
                    "reasoning": result.get(
                        "reasoning", ""
                    ),
                    "supporting_evidence": supporting,
                    "contradicting_evidence": contradicting
                }

            final_results = []

            for item in claims:

                claim_id = item.get("claim_id", "UNKNOWN")

                result = verified.get(claim_id)

                if result is None:

                    result = {
                        "claim_id": claim_id,
                        "claim": item.get("claim", ""),
                        "verdict": "INSUFFICIENT",
                        "reasoning": (
                            "Gemini did not return a valid "
                            "verification result with required "
                            "evidence references."
                        ),
                        "supporting_evidence": [],
                        "contradicting_evidence": [],
                        "error": "Missing or invalid result"
                    }

                final_results.append(result)

                print(f"\nVerifying {claim_id}...")
                print("Verdict:", result["verdict"])
                print("Reasoning:", result["reasoning"])
                print(
                    "Supporting evidence:",
                    result["supporting_evidence"]
                )
                print(
                    "Contradicting evidence:",
                    result["contradicting_evidence"]
                )

            return final_results

        except (
            requests.RequestException,
            ValueError,
            KeyError,
            TypeError,
            json.JSONDecodeError
        ) as error:

            print("\nGlobal verification request failed:", error)

            return [
                {
                    "claim_id": item.get(
                        "claim_id", "UNKNOWN"
                    ),
                    "claim": item.get("claim", ""),
                    "verdict": "INSUFFICIENT",
                    "reasoning": (
                        "Global verification could not be "
                        "completed because of an API or "
                        "response error."
                    ),
                    "supporting_evidence": [],
                    "contradicting_evidence": [],
                    "error": str(error)
                }
                for item in claims
            ]


if __name__ == "__main__":

    from backend.retrieval.chroma_search import (
        CybersecurityChromaSearch
    )

    from backend.verification.claim_decomposition import (
        ClaimDecomposer
    )

    print("\nInitializing batch verification test...")

    search_engine = CybersecurityChromaSearch()
    decomposer = ClaimDecomposer()
    verifier = GlobalVerifier()

    question = "What is CVE-2026-102910?"

    print("\nRetrieving evidence...")

    evidence = search_engine.search(
        question,
        top_k=5
    )

    sample_answer = """
    CVE-2026-102910 is a SQL injection vulnerability.
    It affects SourceCodester Online Reviewer Management
    System 1.0.
    The vulnerability affects exam-delete.php.
    The vulnerability can be exploited remotely.
    """

    print("\nDecomposing answer...")

    claims = decomposer.decompose(sample_answer)

    print("\nRunning batch global verification...")

    results = verifier.verify_claims(
        claims,
        evidence
    )

    output_path = (
        ROOT / "data" / "processed" /
        "global_verification_results.json"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            results,
            file,
            indent=4,
            ensure_ascii=False
        )

    print("\nResults saved to:", output_path)

    print("\nBATCH GLOBAL VERIFICATION COMPLETED")
