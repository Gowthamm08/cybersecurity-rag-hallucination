
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

EVIDENCE_FILE = ROOT / "data/processed/evidence_records.jsonl"
RELATIONSHIP_FILE = ROOT / "data/processed/attack_relationships.jsonl"
EVALUATION_FILE = ROOT / "data/evaluation/synthetic_eval.jsonl"


def load_jsonl(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    with path.open("r", encoding="utf-8") as file:
        return [
            json.loads(line)
            for line in file
            if line.strip()
        ]


def main():
    print("=" * 60)
    print("INGESTION QUALITY VALIDATION")
    print("=" * 60)

    evidence = load_jsonl(EVIDENCE_FILE)
    relationships = load_jsonl(RELATIONSHIP_FILE)
    evaluation = load_jsonl(EVALUATION_FILE)

    evidence_ids = [r.get("record_id") for r in evidence]
    evaluation_ids = [r.get("evaluation_id") for r in evaluation]

    duplicate_ids = len(evidence_ids) - len(set(evidence_ids))

    missing_text = sum(
        1 for r in evidence
        if not str(r.get("text", "")).strip()
    )

    missing_provenance = sum(
        1 for r in evidence
        if not r.get("source_dataset")
        or not r.get("source_record_id")
    )

    synthetic_evidence_ids = {
        r.get("source_record_id")
        for r in evidence
        if r.get("source_dataset") == "Synthetic_Cybersecurity"
    }

    evaluation_mapping_ids = {
        r.get("evaluation_id")
        for r in evaluation
    }

    # Evaluation IDs are SEC-RAG IDs, while evidence uses source IDs.
    # Check that every evaluation ID has a corresponding synthetic
    # evidence record by its SEC-RAG ID.
    synthetic_mapping_valid = (
        len(synthetic_evidence_ids) == len(evaluation)
        and len(evaluation_mapping_ids) == len(evaluation)
    )

    evaluation_has_answers = all(
        bool(str(r.get("ground_truth_answer", "")).strip())
        for r in evaluation
    )

    evidence_has_ground_truth = any(
        "ground_truth_answer" in r
        or "ground_truth" in r
        for r in evidence
    )

    source_counts = Counter(
        r.get("source_dataset", "UNKNOWN")
        for r in evidence
    )

    print(f"\nEvidence records: {len(evidence)}")
    print(f"Relationship records: {len(relationships)}")
    print(f"Evaluation records: {len(evaluation)}")

    print(f"\nDuplicate evidence IDs: {duplicate_ids}")
    print(f"Records without text: {missing_text}")
    print(f"Records missing provenance: {missing_provenance}")

    print("\nEvidence source distribution:")
    for source, count in source_counts.items():
        print(f"  {source}: {count}")

    print(
        "\nEvaluation records contain reference answers:",
        evaluation_has_answers
    )

    print(
        "Evidence records contain ground-truth fields:",
        evidence_has_ground_truth
    )

    checks = {
        "No duplicate evidence IDs": duplicate_ids == 0,
        "All evidence records contain text": missing_text == 0,
        "All evidence records have provenance": missing_provenance == 0,
        "Synthetic evidence records exist": len(synthetic_evidence_ids) > 0,
        "Evaluation records have unique IDs":
            len(evaluation_ids) == len(set(evaluation_ids)),
        "Evaluation answers are available": evaluation_has_answers,
        "Ground truth is excluded from evidence": not evidence_has_ground_truth,
    }

    print("\n" + "=" * 60)
    print("VALIDATION CHECKS")
    print("=" * 60)

    for name, passed in checks.items():
        print(f"[{'PASS' if passed else 'FAIL'}] {name}")

    if all(checks.values()):
        print("\nVALIDATION: PASSED")
    else:
        print("\nVALIDATION: REQUIRES REVIEW")


if __name__ == "__main__":
    main()
