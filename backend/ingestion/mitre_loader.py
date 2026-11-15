
import json
from pathlib import Path
from datetime import datetime, timezone

import requests


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = PROJECT_ROOT / "data" / "raw" / "mitre"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

MITRE_URL = (
    "https://raw.githubusercontent.com/"
    "mitre-attack/attack-stix-data/master/"
    "enterprise-attack/enterprise-attack.json"
)

RAW_FILE = RAW_DIR / "enterprise-attack.json"
OUTPUT_FILE = PROCESSED_DIR / "mitre_attack_records.json"

HEADERS = {
    "User-Agent": "Cybersecurity-RAG-Hallucination-Detector/1.0"
}

# STIX object types useful for our cybersecurity RAG system.
RELEVANT_TYPES = {
    "attack-pattern",
    "x-mitre-tactic",
    "malware",
    "tool",
    "intrusion-set",
    "course-of-action",
    "campaign"
}


def download_dataset():
    """Download the official MITRE ATT&CK Enterprise STIX dataset."""

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    print("\nDownloading MITRE ATT&CK Enterprise dataset...")

    response = requests.get(
        MITRE_URL,
        headers=HEADERS,
        timeout=120
    )

    response.raise_for_status()

    # Validate JSON before saving it.
    data = response.json()

    if not isinstance(data, dict) or not isinstance(
        data.get("objects"), list
    ):
        raise ValueError("Unexpected MITRE STIX bundle format.")

    with open(RAW_FILE, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)

    print(f"Original dataset saved: {RAW_FILE}")

    return data


def get_attack_id(obj):
    """Extract the MITRE ATT&CK ID from external references."""

    for reference in obj.get("external_references", []):
        if reference.get("source_name") == "mitre-attack":
            return reference.get("external_id")

    return None


def normalize_object(obj):
    """Convert a STIX object into a retrieval-friendly record."""

    object_type = obj.get("type")

    name = obj.get("name", "Unnamed object")
    description = obj.get("description", "")

    attack_id = get_attack_id(obj)

    tactics = []

    for phase in obj.get("kill_chain_phases", []):
        if phase.get("kill_chain_name") == "mitre-attack":
            tactics.append(phase.get("phase_name"))

    aliases = obj.get("aliases", [])

    platforms = obj.get("x_mitre_platforms", [])

    detection = obj.get("x_mitre_detection", "")

    text_parts = [
        f"Name: {name}",
        f"Type: {object_type}",
    ]

    if attack_id:
        text_parts.append(f"ATT&CK ID: {attack_id}")

    if description:
        text_parts.append(f"Description: {description}")

    if tactics:
        text_parts.append(
            "Tactics: " + ", ".join(tactics)
        )

    if platforms:
        text_parts.append(
            "Platforms: " + ", ".join(platforms)
        )

    if aliases:
        text_parts.append(
            "Aliases: " + ", ".join(aliases)
        )

    if detection:
        text_parts.append(f"Detection: {detection}")

    return {
        "source": "MITRE ATT&CK",
        "stix_id": obj.get("id"),
        "attack_id": attack_id,
        "object_type": object_type,
        "name": name,
        "description": description,
        "tactics": tactics,
        "platforms": platforms,
        "aliases": aliases,
        "detection": detection,
        "url": next(
            (
                ref.get("url")
                for ref in obj.get("external_references", [])
                if ref.get("source_name") == "mitre-attack"
                and ref.get("url")
            ),
            None
        ),
        "text": "\n".join(text_parts)
    }


def process_dataset(data):
    """Extract and normalize relevant MITRE ATT&CK objects."""

    records = []

    for obj in data.get("objects", []):

        object_type = obj.get("type")

        if object_type not in RELEVANT_TYPES:
            continue

        # Ignore revoked and deprecated objects.
        if obj.get("revoked", False):
            continue

        if obj.get("x_mitre_deprecated", False):
            continue

        # Exclude objects without meaningful names or descriptions.
        if not obj.get("name") and not obj.get("description"):
            continue

        record = normalize_object(obj)

        records.append(record)

    return records


def save_processed_data(records):
    """Save normalized MITRE records for the RAG pipeline."""

    if not records:
        raise ValueError("No valid MITRE records were extracted.")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
        json.dump(
            records,
            file,
            indent=2,
            ensure_ascii=False
        )

    print(f"\nProcessed dataset saved: {OUTPUT_FILE}")


def main():
    """Execute the complete MITRE ingestion pipeline."""

    print("\n================================")
    print("MITRE ATT&CK DATA INGESTION")
    print("================================")

    try:
        data = download_dataset()

        records = process_dataset(data)

        save_processed_data(records)

        type_counts = {}

        for record in records:
            object_type = record["object_type"]

            type_counts[object_type] = (
                type_counts.get(object_type, 0) + 1
            )

        print("\n================================")
        print("INGESTION SUMMARY")
        print("================================")

        print(f"STIX objects downloaded: {len(data['objects'])}")
        print(f"Useful records extracted: {len(records)}")

        print("\nRecords by type:")

        for object_type, count in sorted(type_counts.items()):
            print(f"{object_type}: {count}")

        print("\nMITRE ATT&CK ingestion completed successfully.")

    except (
        requests.RequestException,
        ValueError,
        OSError
    ) as error:

        print(f"\nMITRE ingestion failed: {error}")


if __name__ == "__main__":
    main()
