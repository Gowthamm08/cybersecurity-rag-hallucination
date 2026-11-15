
import json
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

EVAL_FILE = ROOT / "data/evaluation/synthetic_eval.jsonl"
EVIDENCE_FILE = ROOT / "data/processed/evidence_records.jsonl"
OUTPUT_FILE = ROOT / "data/evaluation/claim_annotation_batch.json"

TARGET_PER_CATEGORY = 3
TARGET_TOTAL = 30


def load_jsonl(path):
    records = []

    with open(path, encoding="utf-8") as file:
        for line in file:
            if line.strip():
                records.append(json.loads(line))

    return records


def main():
    evaluations = load_jsonl(EVAL_FILE)
    evidence_records = load_jsonl(EVIDENCE_FILE)

    evidence_map = {
        item.get("source_record_id"): item
        for item in evidence_records
        if item.get("source_dataset") == "Synthetic_Cybersecurity"
    }

    grouped = defaultdict(list)

    for item in evaluations:
        evaluation_id = item.get("evaluation_id")
        evidence = evidence_map.get(evaluation_id)

        if evidence:
            grouped[item.get("category", "Unknown")].append(
                (item, evidence)
            )

    selected = []

    # Select up to three records from each category.
    for category in sorted(grouped):
        records = sorted(
            grouped[category],
            key=lambda pair: pair[0]["evaluation_id"]
        )
        selected.extend(records[:TARGET_PER_CATEGORY])

    # Fill remaining places if some categories have fewer records.
    if len(selected) < TARGET_TOTAL:
        selected_ids = {
            item["evaluation_id"] for item, _ in selected
        }

        remaining = []

        for records in grouped.values():
            for item, evidence in records:
                if item["evaluation_id"] not in selected_ids:
                    remaining.append((item, evidence))

        remaining.sort(key=lambda pair: pair[0]["evaluation_id"])
        selected.extend(remaining[:TARGET_TOTAL - len(selected)])

    batch = []

    for item, evidence in selected:
        answer = item.get("ground_truth_answer", "")

        candidates = [
            sentence.strip()
            for sentence in re.split(r"(?<=[.!?])\s+", answer)
            if sentence.strip()
        ]

        batch.append({
            "evaluation_id": item["evaluation_id"],
            "category": item.get("category", ""),
            "subcategory": item.get("subcategory", ""),
            "question": item.get("question", ""),
            "ground_truth_answer": answer,
            "evidence_title": evidence.get("title", ""),
            "evidence_text": evidence.get("text", ""),
            "candidate_claims": [
                {
                    "claim": sentence,
                    "expected_label": "",
                    "reviewer_note": ""
                }
                for sentence in candidates
            ],
            "additional_claims": [
                {
                    "claim": "",
                    "expected_label": "",
                    "reviewer_note": ""
                }
            ]
        })

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
        json.dump(batch, file, indent=4, ensure_ascii=False)

    print("\nANNOTATION BATCH CREATED")
    print("Available evaluation records:", len(evaluations))
    print("Matched synthetic evidence:", len(evidence_map))
    print("Selected investigations:", len(batch))
    print("Categories represented:", len({
        item["category"] for item, _ in selected
    }))
    print("Output:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
