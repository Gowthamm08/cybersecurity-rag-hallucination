
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

FILES = {
    "EVIDENCE": ROOT / "data/processed/evidence_records.jsonl",
    "EVALUATION": ROOT / "data/evaluation/synthetic_eval.jsonl",
}


def inspect_file(label, path):
    print(f"\n{'=' * 60}")
    print(f"{label} FILE")
    print(f"{'=' * 60}")

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file):
            if not line.strip():
                continue

            record = json.loads(line)

            if line_number == 0:
                print("\nAvailable fields:")
                print(list(record.keys()))

                print("\nFirst record values:")
                for key, value in record.items():
                    value_text = str(value)
                    print(f"{key}: {value_text[:250]}")

            elif line_number == 1:
                print("\nSecond record ID-related fields:")
                for key, value in record.items():
                    if "id" in key.lower() or "source" in key.lower():
                        print(f"{key}: {value}")
                break


def main():
    for label, path in FILES.items():
        if not path.exists():
            print(f"Missing file: {path}")
            continue

        inspect_file(label, path)


if __name__ == "__main__":
    main()
