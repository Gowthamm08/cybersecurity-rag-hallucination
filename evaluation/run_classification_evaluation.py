
import json
from pathlib import Path

from backend.verification.global_verifier import GlobalVerifier
from backend.verification.classifier import ClaimClassifier


ROOT = Path(__file__).resolve().parents[1]

CLAIMS_FILE = ROOT / "data" / "evaluation" / "claim_classification_test.json"
EVIDENCE_FILE = ROOT / "data" / "processed" / "evidence_records.jsonl"
OUTPUT_FILE = (
    ROOT / "data" / "evaluation" /
    "classification_evaluation_results.json"
)

LABELS = ["ENTAILED", "CONTRADICTED", "BASELESS"]


def load_evidence(evaluation_ids):
    evidence_by_id = {}

    with open(EVIDENCE_FILE, "r", encoding="utf-8") as file:
        for line in file:
            record = json.loads(line)
            source_id = record.get("source_record_id")

            if source_id in evaluation_ids:
                evidence_by_id[source_id] = record

    missing = evaluation_ids - set(evidence_by_id)

    if missing:
        raise ValueError(f"Missing evidence records: {missing}")

    return evidence_by_id


def calculate_metrics(actual, predicted):
    metrics = {}

    for label in LABELS:
        true_positive = sum(
            a == label and p == label
            for a, p in zip(actual, predicted)
        )

        false_positive = sum(
            a != label and p == label
            for a, p in zip(actual, predicted)
        )

        false_negative = sum(
            a == label and p != label
            for a, p in zip(actual, predicted)
        )

        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive else 0.0
        )

        recall = (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative else 0.0
        )

        f1_score = (
            2 * precision * recall / (precision + recall)
            if precision + recall else 0.0
        )

        metrics[label] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1_score, 4)
        }

    return metrics


def main():
    with open(CLAIMS_FILE, "r", encoding="utf-8") as file:
        test_cases = json.load(file)

    evaluation_ids = {
        item["evaluation_id"] for item in test_cases
    }

    evidence_by_id = load_evidence(evaluation_ids)

    claims = [
        {
            "claim_id": item["claim_id"],
            "claim": item["claim"]
        }
        for item in test_cases
    ]

    evidence = [
        evidence_by_id[evaluation_id]
        for evaluation_id in sorted(evaluation_ids)
    ]

    print("\nInitializing Global Verifier...")
    verifier = GlobalVerifier()

    print("Initializing Claim Classifier...")
    classifier = ClaimClassifier()

    print("\nRunning claim verification...")
    verification_results = verifier.verify_claims(claims, evidence)

    predicted_results = [
        classifier.classify_claim(result)
        for result in verification_results
    ]

    actual_labels = [
        item["expected_label"] for item in test_cases
    ]

    predicted_labels = [
        item["classification"] for item in predicted_results
    ]

    invalid_labels = [
        label for label in predicted_labels
        if label not in LABELS
    ]

    if invalid_labels:
        raise ValueError(
            f"Unexpected classifier labels: {invalid_labels}. "
            "Inspect verification results before calculating metrics."
        )

    metrics = calculate_metrics(actual_labels, predicted_labels)

    claim_results = []

    for index, test_case in enumerate(test_cases):
        claim_results.append({
            "claim_id": test_case["claim_id"],
            "claim": test_case["claim"],
            "expected_label": actual_labels[index],
            "predicted_label": predicted_labels[index],
            "correct": actual_labels[index] == predicted_labels[index],
            "reasoning": verification_results[index].get(
                "reasoning", ""
            )
        })

    report = {
        "total_claims": len(test_cases),
        "metrics": metrics,
        "claim_results": claim_results
    }

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=4, ensure_ascii=False)

    print("\n========== CLASSIFICATION EVALUATION ==========")
    print("Total claims:", len(test_cases))

    for label, scores in metrics.items():
        print(f"\n{label}")
        print("Precision:", scores["precision"])
        print("Recall:", scores["recall"])
        print("F1-score:", scores["f1_score"])

    correct = sum(
        actual == predicted
        for actual, predicted in zip(actual_labels, predicted_labels)
    )

    print(f"\nCorrect predictions: {correct}/{len(test_cases)}")
    print("\nResults saved to:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()
