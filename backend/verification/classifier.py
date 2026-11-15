
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

GLOBAL_RESULTS_PATH = (
    ROOT / "data" / "processed" /
    "global_verification_results.json"
)

CLASSIFICATION_PATH = (
    ROOT / "data" / "processed" /
    "classification_results.json"
)


class ClaimClassifier:

    VERDICT_MAPPING = {
        "SUPPORTED": "ENTAILED",
        "CONTRADICTED": "CONTRADICTED"
    }

    ABSENCE_PATTERNS = [
        r"\bdoes not (?:specify|mention|contain|include|provide|identify)\b",
        r"\bnot (?:specified|mentioned|provided|available|documented)\b",
        r"\bno information (?:about|on|regarding)\b",
        r"\babsence of\b",
        r"\bmissing information\b",
        r"\bnot found in (?:the )?(?:provided )?evidence\b"
    ]

    def is_absence_claim(self, claim):

        text = claim.lower()

        return any(
            re.search(pattern, text)
            for pattern in self.ABSENCE_PATTERNS
        )

    def classify_claim(self, verification_result):

        verdict = verification_result.get(
            "verdict", ""
        ).upper()

        claim = verification_result.get("claim", "")

        # API errors must never become factual classifications.
        if verification_result.get("error"):

            classification = "UNVERIFIED"

        elif verdict in self.VERDICT_MAPPING:

            classification = self.VERDICT_MAPPING[verdict]

        elif verdict == "INSUFFICIENT":

            if self.is_absence_claim(claim):
                classification = "NOT_IN_EVIDENCE"
            else:
                classification = "BASELESS"

        else:

            classification = "UNVERIFIED"

        return {
            "claim_id": verification_result.get(
                "claim_id", "UNKNOWN"
            ),
            "claim": claim,
            "global_verdict": verdict,
            "classification": classification,
            "reasoning": verification_result.get(
                "reasoning", ""
            ),
            "supporting_evidence": verification_result.get(
                "supporting_evidence", []
            ),
            "contradicting_evidence": verification_result.get(
                "contradicting_evidence", []
            )
        }

    def classify_claims(self, verification_results):

        return [
            self.classify_claim(result)
            for result in verification_results
        ]

    def generate_summary(self, results):

        summary = {
            "total_claims": len(results),
            "entailed": 0,
            "contradicted": 0,
            "baseless": 0,
            "not_in_evidence": 0,
            "unverified": 0
        }

        for result in results:

            classification = result["classification"]

            if classification == "ENTAILED":
                summary["entailed"] += 1

            elif classification == "CONTRADICTED":
                summary["contradicted"] += 1

            elif classification == "BASELESS":
                summary["baseless"] += 1

            elif classification == "NOT_IN_EVIDENCE":
                summary["not_in_evidence"] += 1

            elif classification == "UNVERIFIED":
                summary["unverified"] += 1

        return summary


if __name__ == "__main__":

    print("\nCYBERSECURITY CLAIM CLASSIFIER")
    print("=" * 55)

    if not GLOBAL_RESULTS_PATH.exists():

        raise FileNotFoundError(
            f"Global verification results not found: "
            f"{GLOBAL_RESULTS_PATH}"
        )

    with open(
        GLOBAL_RESULTS_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        verification_results = json.load(file)

    classifier = ClaimClassifier()

    classified_results = classifier.classify_claims(
        verification_results
    )

    summary = classifier.generate_summary(
        classified_results
    )

    print("\nCLAIM CLASSIFICATION RESULTS")
    print("=" * 55)

    for result in classified_results:

        print(f"\nClaim ID: {result['claim_id']}")
        print(f"Claim: {result['claim']}")
        print(f"Global Verdict: {result['global_verdict']}")
        print(f"Final Classification: {result['classification']}")
        print(f"Supporting Evidence: {result['supporting_evidence']}")

    print("\nCLASSIFICATION SUMMARY")
    print("=" * 55)

    for key, value in summary.items():
        print(f"{key}: {value}")

    output = {
        "summary": summary,
        "results": classified_results
    }

    with open(
        CLASSIFICATION_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            output,
            file,
            indent=4,
            ensure_ascii=False
        )

    print("\nResults saved to:")
    print(CLASSIFICATION_PATH)

    print("\nCLAIM CLASSIFICATION COMPLETED")
