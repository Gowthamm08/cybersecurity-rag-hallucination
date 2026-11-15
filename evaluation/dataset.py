
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

EVALUATION_FILE = (
    ROOT / "data" / "evaluation" / "synthetic_eval.jsonl"
)

EVIDENCE_FILE = (
    ROOT / "data" / "processed" / "evidence_records.jsonl"
)


def load_jsonl(file_path):

    records = []

    if not file_path.exists():
        raise FileNotFoundError(
            f"File not found: {file_path}"
        )

    with open(file_path, "r", encoding="utf-8") as file:

        for line_number, line in enumerate(file, start=1):

            if not line.strip():
                continue

            try:
                records.append(json.loads(line))

            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON at line {line_number}: "
                    f"{file_path}"
                ) from error

    return records


def load_evaluation_dataset():

    evaluation_records = load_jsonl(EVALUATION_FILE)
    evidence_records = load_jsonl(EVIDENCE_FILE)

    synthetic_evidence = {
        item["source_record_id"]: item
        for item in evidence_records
        if item.get("source_dataset") == "Synthetic_Cybersecurity"
        and item.get("source_record_id")
    }

    dataset = []
    missing_evidence = []

    for item in evaluation_records:

        evaluation_id = item.get("evaluation_id")

        if not evaluation_id:
            continue

        matching_evidence = synthetic_evidence.get(
            evaluation_id
        )

        if matching_evidence is None:
            missing_evidence.append(evaluation_id)
            continue

        dataset.append({
            "evaluation_id": evaluation_id,
            "question": item.get("question", ""),
            "ground_truth_answer": item.get(
                "ground_truth_answer", ""
            ),
            "category": item.get("category", ""),
            "subcategory": item.get("subcategory", ""),
            "mapping_tags": item.get("mapping_tags", ""),
            "relevant_record_ids": [evaluation_id],
            "evidence_title": matching_evidence.get(
                "title", ""
            )
        })

    report = {
        "evaluation_records": len(evaluation_records),
        "synthetic_evidence_records": len(synthetic_evidence),
        "mapped_records": len(dataset),
        "missing_evidence_records": missing_evidence
    }

    return dataset, report


if __name__ == "__main__":

    dataset, report = load_evaluation_dataset()

    print("\nEVALUATION DATASET VALIDATION")
    print("=" * 50)

    for key, value in report.items():
        print(f"{key}: {value}")

    print("\nFirst three evaluation samples:")

    for item in dataset[:3]:

        print("\nID:", item["evaluation_id"])
        print("Question:", item["question"])
        print("Relevant evidence:", item["relevant_record_ids"])

    if report["missing_evidence_records"]:
        print("\nWARNING: Some evaluation records lack evidence.")

    else:
        print("\nDATASET MAPPING VALIDATION PASSED")
