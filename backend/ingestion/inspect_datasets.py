
import csv
import json
import zipfile
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"


def inspect_nvd():
    print("\n" + "=" * 60)
    print("1. NVD CVE DATASET")
    print("=" * 60)

    path = RAW / "nvd" / "extracted" / "nvdcve-2.0-recent.json"

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    vulnerabilities = data.get("vulnerabilities", [])

    print("File:", path.name)
    print("Total CVE records:", len(vulnerabilities))
    print("Top-level keys:", list(data.keys()))

    if vulnerabilities:
        cve = vulnerabilities[0].get("cve", {})
        print("Sample CVE ID:", cve.get("id"))
        print("CVE fields:", list(cve.keys()))


def inspect_mitre_csv():
    print("\n" + "=" * 60)
    print("2. MITRE ATT&CK CSV")
    print("=" * 60)

    path = RAW / "mitre" / "mitre_enterprise_attack_11_1.csv"

    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        rows = list(reader)

    print("File:", path.name)
    print("Total records:", len(rows))
    print("Columns:", reader.fieldnames)

    if rows:
        print("Sample ID:", rows[0].get("id"))
        print("Sample type:", rows[0].get("type"))
        print("Sample name:", rows[0].get("name"))

    print("Record types:", dict(Counter(
        row.get("type", "unknown") for row in rows
    )))


def inspect_stix():
    print("\n" + "=" * 60)
    print("3. ENTERPRISE ATT&CK STIX JSON")
    print("=" * 60)

    path = RAW / "enterprise_attack" / "enterprise-attack-11.1.json"

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    objects = data.get("objects", [])
    type_counts = Counter(obj.get("type", "unknown") for obj in objects)

    print("File:", path.name)
    print("STIX specification:", data.get("spec_version"))
    print("Total STIX objects:", len(objects))
    print("Object types:")

    for object_type, count in sorted(type_counts.items()):
        print(f"  {object_type}: {count}")

    collection = next(
        (obj for obj in objects if obj.get("type") == "x-mitre-collection"),
        None
    )

    if collection:
        print("Collection:", collection.get("name"))
        print("ATT&CK version:", collection.get("x_mitre_version"))


def inspect_synthetic_csv():
    print("\n" + "=" * 60)
    print("4. SYNTHETIC CYBERSECURITY DATASET")
    print("=" * 60)

    path = RAW / "synthetic" / "cybersecurity_rag_synthetic_dataset_expanded_3530.csv"

    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        rows = list(reader)

    print("File:", path.name)
    print("Total records:", len(rows))
    print("Columns:", reader.fieldnames)

    print("Category distribution:")

    for category, count in sorted(Counter(
        row.get("category", "unknown") for row in rows
    ).items()):
        print(f"  {category}: {count}")

    if rows:
        print("Sample record ID:", rows[0].get("id"))
        print("Sample category:", rows[0].get("category"))
        print("Raw text length:", len(rows[0].get("raw_text", "")))


def main():
    print("\nCYBERSECURITY RAG DATASET INSPECTION")
    print("Project:", ROOT)

    inspect_nvd()
    inspect_mitre_csv()
    inspect_stix()
    inspect_synthetic_csv()

    print("\n" + "=" * 60)
    print("DATASET INSPECTION COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()
