
import json
import os
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


class ClaimDecomposer:

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

        print("Claim decomposer initialized.")

    def decompose(self, answer):

        if not answer or not answer.strip():
            return []

        prompt = f"""
You are a cybersecurity claim decomposition system.

Your task is to split the given answer into small,
atomic and independently verifiable factual claims.

Rules:
1. Each claim must contain one factual statement.
2. Do not add new facts.
3. Preserve technical names, CVE IDs and version numbers.
4. Avoid opinions and unnecessary explanations.
5. Return only valid JSON.
6. Do not include Markdown code blocks.

ANSWER:
{answer}

Return JSON in this exact format:

{{
    "claims": [
        {{
            "claim": "First factual claim"
        }},
        {{
            "claim": "Second factual claim"
        }}
    ]
}}
"""

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
                "maxOutputTokens": 1500,
                "responseMimeType": "application/json"
            }
        }

        response = None

        # Retry temporary Gemini API failures.
        for attempt in range(4):

            try:

                response = self.session.post(
                    API_URL,
                    headers=self.headers,
                    json=payload,
                    timeout=(10, 60)
                )

                if response.status_code in [
                    429, 500, 502, 503, 504
                ]:

                    if attempt < 3:

                        wait_time = 2 ** (attempt + 1)

                        print(
                            f"Gemini temporarily unavailable "
                            f"(HTTP {response.status_code}). "
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
            generated_text = (
                result["candidates"][0]["content"]["parts"][0]["text"]
            )

        except (KeyError, IndexError, TypeError) as error:

            raise ValueError(
                f"Invalid Gemini response: {result}"
            ) from error

        # Remove accidental Markdown code fences.
        generated_text = re.sub(
            r"^```(?:json)?\s*|\s*```$",
            "",
            generated_text.strip(),
            flags=re.IGNORECASE
        )

        try:
            parsed = json.loads(generated_text)

        except json.JSONDecodeError as error:

            raise ValueError(
                f"Gemini returned invalid JSON: {generated_text}"
            ) from error

        raw_claims = parsed.get("claims", [])

        if not isinstance(raw_claims, list):
            raise ValueError(
                "Gemini response must contain a claims list."
            )

        claims = []

        for item in raw_claims:

            if isinstance(item, dict):
                claim_text = item.get("claim", "")

            elif isinstance(item, str):
                claim_text = item

            else:
                continue

            claim_text = claim_text.strip()

            if claim_text:

                claims.append({
                    "claim_id": f"C{len(claims) + 1}",
                    "claim": claim_text
                })

        return claims


if __name__ == "__main__":

    decomposer = ClaimDecomposer()

    sample_answer = """
    CVE-2026-102910 is a SQL injection vulnerability.
    It affects SourceCodester Online Reviewer Management System 1.0.
    The vulnerability affects exam-delete.php.
    The vulnerability can be exploited remotely.
    """

    print("\nDecomposing answer into claims...\n")

    claims = decomposer.decompose(sample_answer)

    for item in claims:

        print(f"{item['claim_id']}: {item['claim']}")

    print("\nCLAIM DECOMPOSITION TEST COMPLETED")
