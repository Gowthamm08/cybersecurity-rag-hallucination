
import csv
import json
import hashlib
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
EVALUATION = ROOT / "data" / "evaluation"

NVD_FILE = RAW / "nvd" / "extracted" / "nvdcve-2.0-recent.json"
MITRE_CSV = RAW / "mitre" / "mitre_enterprise_attack_11_1.csv"
STIX_FILE = RAW / "enterprise_attack" / "enterprise-attack-11.1.json"
SYNTHETIC_CSV = (
    RAW / "synthetic" /
    "cybersecurity_rag_synthetic_dataset_expanded_3530.csv"
)

EVIDENCE_FILE = PROCESSED / "evidence_records.jsonl"
RELATIONSHIPS_FILE = PROCESSED / "attack_relationships.jsonl"
EVALUATION_FILE = EVALUATION / "synthetic_eval.jsonl"
REPORT_FILE = PROCESSED / "ingestion_report.json"


def write_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_csv(path):
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def get_english_description(items):
    for item in items or []:
        if item.get("lang") == "en":
            return item.get("value", "")
    return items[0].get("value", "") if items else ""


def make_record_id(source, identifier):
    # Use stable source-specific identifiers, not display names
    value = f"{source}:{identifier}"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:20]


def extract_nvd_records():
    print("\n[1/4] Processing NVD CVE dataset...")

    with NVD_FILE.open("r", encoding="utf-8") as file:
        data = json.load(file)

    records = []

    for item in data.get("vulnerabilities", []):
        cve = item.get("cve", {})
        cve_id = cve.get("id", "")

        description = get_english_description(
            cve.get("descriptions", [])
        )

        if not cve_id or not description.strip():
            continue

        weaknesses = []
        for weakness in cve.get("weaknesses", []):
            for desc in weakness.get("description", []):
                if desc.get("lang") == "en":
                    weaknesses.append(desc.get("value", ""))

        references = [
            ref.get("url", "")
            for ref in cve.get("references", [])
            if ref.get("url")
        ]

        affected_products = [
            f"{a.get('vendor', '')} {a.get('product', '')}".strip()
            for a in cve.get("affected", [])
            if a.get("vendor") or a.get("product")
        ]

        text_parts = [
            f"Vulnerability ID: {cve_id}",
            f"Description: {description}",
        ]

        if weaknesses:
            text_parts.append("Weaknesses: " + ", ".join(weaknesses))

        if affected_products:
            text_parts.append(
                "Affected products: " + ", ".join(affected_products)
            )

        records.append({
            "record_id": make_record_id("nvd", cve_id),
            "source_dataset": "NVD",
            "source_record_id": cve_id,
            "source_version": data.get("version", "unknown"),
            "record_type": "vulnerability",
            "title": cve_id,
            "text": "\n".join(text_parts),
            "metadata": {
                "published": cve.get("published"),
                "last_modified": cve.get("lastModified"),
                "status": cve.get("vulnStatus"),
                "weaknesses": weaknesses,
                "affected_products": affected_products,
                "references": references,
            },
        })

    print("NVD evidence records:", len(records))
    return records


def extract_mitre_csv_records():
    print("\n[2/4] Processing MITRE ATT&CK CSV...")

    rows = read_csv(MITRE_CSV)
    records = []

    for row in rows:
        name = (row.get("name") or "").strip()
        description = (row.get("description") or "").strip()
        stix_id = (row.get("id") or "").strip()

        if not name or not description or not stix_id:
            continue

        # STIX UUID is unique even when names or external IDs repeat.
        records.append({
            "record_id": make_record_id("mitre", stix_id),
            "source_dataset": "MITRE_ATT&CK_CSV",
            "source_record_id": stix_id,
            "source_version": row.get("version") or "11.1",
            "record_type": row.get("type", "unknown"),
            "title": name,
            "text": f"{name}\n{description}",
            "metadata": {
                "stix_id": stix_id,
                "mitre_external_id": row.get("mitre_external_id", ""),
                "platforms": row.get("platforms", ""),
                "domains": row.get("domains", ""),
                "created": row.get("created"),
                "modified": row.get("modified"),
                "url": row.get("url"),
            },
        })

    print("MITRE CSV evidence records:", len(records))
    return records


def extract_stix_records(mitre_records):
    print("\n[3/4] Processing Enterprise ATT&CK STIX JSON...")

    with STIX_FILE.open("r", encoding="utf-8") as file:
        data = json.load(file)

    objects = data.get("objects", [])

    # Exact STIX ID matching prevents collisions caused by
    # duplicate external IDs or identical display names.
    csv_by_stix_id = {
        record["source_record_id"]: record
        for record in mitre_records
    }

    relationships = []
    additional_records = []
    merged_ids = set()
    merged_count = 0

    for obj in objects:
        obj_type = obj.get("type", "")
        obj_id = obj.get("id", "")

        if obj_type == "relationship":
            relationships.append({
                "relationship_id": obj_id,
                "relationship_type": obj.get("relationship_type"),
                "source_ref": obj.get("source_ref"),
                "target_ref": obj.get("target_ref"),
                "description": obj.get("description", ""),
                "source_dataset": "Enterprise_ATT&CK_STIX",
                "source_version": "11.1",
            })
            continue

        if obj_type in {
            "x-mitre-collection",
            "marking-definition",
            "identity",
        }:
            continue

        name = (obj.get("name") or "").strip()
        description = (obj.get("description") or "").strip()

        if not name or not obj_id:
            continue

        external_id = ""
        for ref in obj.get("external_references", []):
            if ref.get("source_name") == "mitre-attack":
                external_id = ref.get("external_id", "")
                break

        # Merge only when the exact STIX ID matches.
        existing = csv_by_stix_id.get(obj_id)

        if existing:
            if obj_id not in merged_ids:
                existing["metadata"]["stix_id"] = obj_id
                existing["metadata"]["stix_version"] = "11.1"
                existing["metadata"]["stix_platforms"] = (
                    obj.get("x_mitre_platforms", [])
                )
                existing["source_dataset"] = (
                    "MITRE_ATT&CK_CSV+Enterprise_ATT&CK_STIX"
                )
                merged_ids.add(obj_id)
                merged_count += 1
            continue

        # Keep STIX-only records, such as mitigations and data sources.
        if not description:
            continue

        additional_records.append({
            "record_id": make_record_id("stix", obj_id),
            "source_dataset": "Enterprise_ATT&CK_STIX",
            "source_record_id": obj_id,
            "source_version": "11.1",
            "record_type": obj_type,
            "title": name,
            "text": f"{name}\n{description}",
            "metadata": {
                "stix_id": obj_id,
                "mitre_external_id": external_id,
                "platforms": obj.get("x_mitre_platforms", []),
                "kill_chain_phases": obj.get(
                    "kill_chain_phases", []
                ),
                "external_references": obj.get(
                    "external_references", []
                ),
            },
        })

    print("STIX objects inspected:", len(objects))
    print("Exact CSV/STIX matches merged:", merged_count)
    print("Additional STIX evidence records:", len(additional_records))
    print("STIX relationships extracted:", len(relationships))

    return additional_records, relationships


def extract_synthetic_records():
    print("\n[4/4] Processing synthetic cybersecurity dataset...")

    rows = read_csv(SYNTHETIC_CSV)
    evidence_records = []
    evaluation_records = []

    for row in rows:
        record_id = (row.get("id") or "").strip()
        raw_text = (row.get("raw_text") or "").strip()

        if not record_id or not raw_text:
            continue

        evidence_records.append({
            "record_id": make_record_id("synthetic", record_id),
            "source_dataset": "Synthetic_Cybersecurity",
            "source_record_id": record_id,
            "source_version": "expanded_3530",
            "record_type": "synthetic_investigation",
            "title": row.get("context_title", ""),
            "text": raw_text,
            "metadata": {
                "category": row.get("category", ""),
                "subcategory": row.get("subcategory", ""),
                "mapping_tags": row.get("mapping_tags", ""),
            },
        })

        evaluation_records.append({
            "evaluation_id": record_id,
            "question": row.get("baseline_question", ""),
            "ground_truth_answer": row.get("ground_truth_answer", ""),
            "category": row.get("category", ""),
            "subcategory": row.get("subcategory", ""),
            "mapping_tags": row.get("mapping_tags", ""),
        })

    print("Synthetic evidence records:", len(evidence_records))
    print("Synthetic evaluation records:", len(evaluation_records))

    return evidence_records, evaluation_records


def main():
    print("=" * 65)
    print("CYBERSECURITY RAG UNIFIED DATA INGESTION")
    print("=" * 65)

    required_files = [
        NVD_FILE,
        MITRE_CSV,
        STIX_FILE,
        SYNTHETIC_CSV,
    ]

    for path in required_files:
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")

    PROCESSED.mkdir(parents=True, exist_ok=True)
    EVALUATION.mkdir(parents=True, exist_ok=True)

    nvd_records = extract_nvd_records()
    mitre_records = extract_mitre_csv_records()

    stix_records, relationships = extract_stix_records(
        mitre_records
    )

    synthetic_records, evaluation_records = (
        extract_synthetic_records()
    )

    evidence_records = (
        nvd_records
        + mitre_records
        + stix_records
        + synthetic_records
    )

    # Fail early rather than silently writing duplicate IDs.
    record_ids = [r["record_id"] for r in evidence_records]
    duplicate_ids = [
        key for key, count in Counter(record_ids).items()
        if count > 1
    ]

    if duplicate_ids:
        raise ValueError(
            f"Duplicate evidence IDs detected: {duplicate_ids[:10]}"
        )

    write_jsonl(EVIDENCE_FILE, evidence_records)
    write_jsonl(RELATIONSHIPS_FILE, relationships)
    write_jsonl(EVALUATION_FILE, evaluation_records)

    report = {
        "datasets": {
            "nvd_records": len(nvd_records),
            "mitre_csv_records": len(mitre_records),
            "additional_stix_records": len(stix_records),
            "synthetic_evidence_records": len(synthetic_records),
            "synthetic_evaluation_records": len(evaluation_records),
            "stix_relationships": len(relationships),
        },
        "total_searchable_evidence_records": len(evidence_records),
        "source_distribution": dict(Counter(
            r["source_dataset"] for r in evidence_records
        )),
        "record_type_distribution": dict(Counter(
            r["record_type"] for r in evidence_records
        )),
        "ground_truth_in_evidence": False,
        "duplicate_evidence_ids": 0,
    }

    with REPORT_FILE.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=4, ensure_ascii=False)

    print("\n" + "=" * 65)
    print("INGESTION SUMMARY")
    print("=" * 65)
    print("Searchable evidence records:", len(evidence_records))
    print("STIX relationships:", len(relationships))
    print("Evaluation records:", len(evaluation_records))
    print("Duplicate evidence IDs:", 0)
    print("Ground-truth leakage: PREVENTED")
    print("INGESTION COMPLETED")


if __name__ == "__main__":
    main()
