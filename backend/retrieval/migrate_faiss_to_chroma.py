
import json
from pathlib import Path

import chromadb
import faiss
import numpy as np


# Project paths
ROOT = Path(__file__).resolve().parents[2]

INDEX_PATH = ROOT / "vector_store" / "cybersecurity_faiss.index"
METADATA_PATH = ROOT / "vector_store" / "chunk_metadata.jsonl"
CHROMA_PATH = ROOT / "vector_store" / "chromadb"

COLLECTION_NAME = "cybersecurity_evidence"
BATCH_SIZE = 500


def load_metadata():
    records = []

    with open(METADATA_PATH, "r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                records.append(json.loads(line))

    return records


def clean_metadata(record):
    """Keep only scalar metadata supported by ChromaDB."""
    excluded = {"text"}

    return {
        key: value
        for key, value in record.items()
        if key not in excluded
        and isinstance(value, (str, int, float, bool))
    }


def main():
    print("Starting FAISS to ChromaDB migration...")

    # Check source files
    if not INDEX_PATH.exists():
        raise FileNotFoundError(f"FAISS index not found: {INDEX_PATH}")

    if not METADATA_PATH.exists():
        raise FileNotFoundError(f"Metadata file not found: {METADATA_PATH}")

    # Load metadata
    records = load_metadata()

    print(f"Metadata records loaded: {len(records)}")

    ids = [record["chunk_id"] for record in records]

    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate chunk IDs found in metadata.")

    if any(not record.get("text") for record in records):
        raise ValueError("Some records have missing text.")

    # Load FAISS index
    index = faiss.read_index(str(INDEX_PATH))

    print(f"FAISS vectors: {index.ntotal}")
    print(f"Embedding dimensions: {index.d}")

    if index.ntotal != len(records):
        raise ValueError(
            "FAISS vector count and metadata count do not match."
        )

    # Recover existing embeddings without regenerating them
    embeddings = index.reconstruct_n(0, index.ntotal)
    embeddings = np.asarray(embeddings, dtype=np.float32)

    if embeddings.shape != (len(records), index.d):
        raise ValueError("Unexpected embedding matrix shape.")

    if not np.isfinite(embeddings).all():
        raise ValueError("Embeddings contain invalid values.")

    print(f"Embedding matrix: {embeddings.shape}")

    # Create persistent ChromaDB database
    client = chromadb.PersistentClient(
        path=str(CHROMA_PATH)
    )

    # Avoid accidentally mixing old and new migration data
    existing = client.get_collection(
        name=COLLECTION_NAME
    ) if COLLECTION_NAME in [
        item.name for item in client.list_collections()
    ] else None

    if existing is not None:
        if existing.count() > 0:
            raise RuntimeError(
                f"Collection '{COLLECTION_NAME}' already contains "
                f"{existing.count()} records. Migration stopped safely. "
                "Inspect the collection before retrying."
            )
        collection = existing
    else:
        collection = client.create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )

    print("ChromaDB collection ready.")

    # Transfer vectors, text and metadata in batches
    for start in range(0, len(records), BATCH_SIZE):
        end = min(start + BATCH_SIZE, len(records))

        batch_records = records[start:end]

        collection.add(
            ids=ids[start:end],
            embeddings=embeddings[start:end].tolist(),
            documents=[
                record["text"] for record in batch_records
            ],
            metadatas=[
                clean_metadata(record) for record in batch_records
            ]
        )

        print(
            f"Migrated {end}/{len(records)} records"
        )

    # Verify migration
    migrated_count = collection.count()

    print("\n========== MIGRATION REPORT ==========")
    print(f"FAISS vectors: {index.ntotal}")
    print(f"Metadata records: {len(records)}")
    print(f"ChromaDB records: {migrated_count}")
    print(f"Embedding dimensions: {index.d}")
    print(f"Database location: {CHROMA_PATH}")

    if migrated_count != index.ntotal:
        raise RuntimeError("Migration verification failed.")

    print("\nMIGRATION COMPLETED SUCCESSFULLY!")


if __name__ == "__main__":
    main()
