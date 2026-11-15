
import json
from pathlib import Path
from collections import defaultdict, Counter

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "data/processed/evidence_records.jsonl"


def main():
    records = []

    with EVIDENCE.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                records.append(json.loads(line))

    by_id = defaultdict(list)

    for record in records:
        by_id[record["record_id"]].append(record)

    duplicates = {
        key: values
        for key, values in by_id.items()
        if len(values) > 1
    }

    print("=" * 65)
    print("DUPLICATE RECORD DIAGNOSTICS")
    print("=" * 65)

    print("Duplicate record IDs:", len(duplicates))

    for record_id, matches in duplicates.items():
        print("\nDuplicate ID:", record_id)

        for index, record in enumerate(matches, start=1):
            print(f"\nRecord {index}")
            print("  Source:", record.get("source_dataset"))
            print("  Source ID:", record.get("source_record_id"))
            print("  Type:", record.get("record_type"))
            print("  Title:", record.get("title"))
            print("  STIX ID:", record.get("metadata", {}).get("stix_id"))

    print("\n" + "=" * 65)
    print("SOURCE RECORD ID CHECK")
    print("=" * 65)

    source_ids = defaultdict(list)

    for record in records:
        key = (
            record.get("source_dataset"),
            record.get("source_record_id"),
        )
        source_ids[key].append(record)

    repeated_sources = {
        key: value
        for key, value in source_ids.items()
        if len(value) > 1
    }

    print("Repeated source ID groups:", len(repeated_sources))

    for (source, source_id), matches in list(
        repeated_sources.items()
    )[:10]:
        print(f"\nSource: {source}")
        print("Source ID:", source_id)
        print("Occurrences:", len(matches))

    print("\n" + "=" * 65)
    print("REPEATED STIX SOURCE LABEL CHECK")
    print("=" * 65)

    for record in records:
        source = record.get("source_dataset", "")
        if source.count("Enterprise_ATT&CK_STIX") > 1:
            print(
                record.get("record_id"),
                record.get("source_record_id"),
                source,
            )

    print("\n" + "=" * 65)
    print("DIAGNOSTICS COMPLETED")
    print("=" * 65)


if __name__ == "__main__":
    main()
