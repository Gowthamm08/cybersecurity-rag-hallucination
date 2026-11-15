
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

    ABSENCE_PATTERNS = [
        r"\bdoes not (?:prove|establish|confirm|demonstrate)\b",
        r"\bnot proven\b",
        r"\bno evidence (?:of|for)\b",
        r"\bno .* evidence (?:is )?present\b",
        r"\bno .* artifact\b",
        r"\babsence of\b",
        r"\bnot detected\b",
        r"\bnot established\b",
        r"\bnot confirmed\b",
        r"\bmissing information\b",
        r"\bnot found in .* evidence\b"
    ]

    def classify_claim(self, verification_result):

        verdict = verification_result.get(
            "verdict", ""
        ).upper()

        claim = verification_result.get("claim", "")
        reasoning = verification_result.get("reasoning", "")

        supporting = verification_result.get(
            "supporting_evidence", []
        )

        contradicting = verification_result.get(
            "contradicting_evidence", []
        )

        if verification_result.get("error"):
            classification = "UNVERIFIED"

        elif verdict == "SUPPORTED" and supporting:
            classification = "ENTAILED"

        elif verdict == "CONTRADICTED" and contradicting:

            # A claim that something is proven can be refuted
            # by evidence explicitly stating it is not proven.
            combined = (claim + " " + reasoning).lower()

            proof_claim = re.search(
                r"\b(proves?|proven|confirmed|established)\b",
                claim.lower()
            )

            absence_reasoning = any(
                re.search(pattern, reasoning.lower())
                for pattern in self.ABSENCE_PATTERNS
            )

            if proof_claim and absence_reasoning:
                classification = "BASELESS"
            else:
                classification = "CONTRADICTED"

        elif verdict == "INSUFFICIENT":
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
            "reasoning": reasoning,
            "supporting_evidence": supporting,
            "contradicting_evidence": contradicting
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
            "unverified": 0
        }

        for result in results:
            label = result["classification"].lower()

            if label in summary:
                summary[label] += 1

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

    for result in classified_results:
        print(f"\nClaim: {result['claim']}")
        print(f"Classification: {result['classification']}")

    print("\nCLASSIFICATION SUMMARY")
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

    print("\nResults saved to:", CLASSIFICATION_PATH)
