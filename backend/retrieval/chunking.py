
import json
import hashlib
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = ROOT / "data/processed/evidence_records.jsonl"
OUTPUT_FILE = ROOT / "data/processed/evidence_chunks.jsonl"
REPORT_FILE = ROOT / "data/processed/chunking_report.json"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
MIN_CHUNK_SIZE = 700


def load_jsonl(path):
    with path.open("r", encoding="utf-8") as file:
        return [
            json.loads(line)
            for line in file
            if line.strip()
        ]


def split_text(text):
    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        return []

    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        target_end = min(start + CHUNK_SIZE, text_length)
        end = target_end

        if target_end < text_length:
            minimum_end = min(
                start + MIN_CHUNK_SIZE,
                target_end
            )

            # Prefer a sentence boundary.
            sentence_matches = list(
                re.finditer(r'[.!?](?=\s|$)', text[minimum_end:target_end])
            )

            if sentence_matches:
                end = (
                    minimum_end
                    + sentence_matches[-1].end()
                )
            else:
                # Fall back to the last available word boundary.
                boundary = text.rfind(
                    " ",
                    minimum_end,
                    target_end
                )

                if boundary > start:
                    end = boundary

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= text_length:
            break

        # Calculate overlap and move forward to a word boundary.
        next_start = max(
            end - CHUNK_OVERLAP,
            start + 1
        )

        next_space = text.find(" ", next_start)

        if next_space != -1 and next_space < end:
            next_start = next_space + 1

        if next_start <= start:
            next_start = end

        start = next_start

    return chunks


def create_chunk_id(record_id, chunk_index, chunk_text):
    content = f"{record_id}:{chunk_index}:{chunk_text}"

    return hashlib.sha256(
        content.encode("utf-8")
    ).hexdigest()[:24]


def main():
    print("=" * 65)
    print("CYBERSECURITY RAG DOCUMENT CHUNKING")
    print("=" * 65)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Evidence file not found: {INPUT_FILE}"
        )

    records = load_jsonl(INPUT_FILE)
    all_chunks = []

    print(f"\nInput evidence records: {len(records)}")
    print(f"Chunk size: {CHUNK_SIZE}")
    print(f"Chunk overlap: {CHUNK_OVERLAP}")

    for record in records:
        text = record.get("text", "")

        if not isinstance(text, str) or not text.strip():
            continue

        text_chunks = split_text(text)

        for index, chunk_text in enumerate(text_chunks):
            chunk = {
                "chunk_id": create_chunk_id(
                    record["record_id"],
                    index,
                    chunk_text
                ),
                "record_id": record["record_id"],
                "source_dataset": record["source_dataset"],
                "source_record_id": record["source_record_id"],
                "source_version": record.get("source_version"),
                "record_type": record.get("record_type"),
                "title": record.get("title"),
                "chunk_index": index,
                "text": chunk_text,
                "metadata": record.get("metadata", {})
            }

            all_chunks.append(chunk)

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with OUTPUT_FILE.open("w", encoding="utf-8") as file:
        for chunk in all_chunks:
            file.write(
                json.dumps(chunk, ensure_ascii=False) + "\n"
            )

    chunk_lengths = [
        len(chunk["text"])
        for chunk in all_chunks
    ]

    report = {
        "input_records": len(records),
        "total_chunks": len(all_chunks),
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "minimum_chunk_length": min(
            chunk_lengths,
            default=0
        ),
        "maximum_chunk_length": max(
            chunk_lengths,
            default=0
        ),
        "average_chunk_length": (
            round(
                sum(chunk_lengths) / len(chunk_lengths),
                2
            )
            if chunk_lengths else 0
        ),
        "output_file": str(OUTPUT_FILE)
    }

    with REPORT_FILE.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)

    print("\n" + "=" * 65)
    print("CHUNKING SUMMARY")
    print("=" * 65)

    for key, value in report.items():
        print(f"{key}: {value}")

    print("\nCHUNKING COMPLETED")


if __name__ == "__main__":
    main()
