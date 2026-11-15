
import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = ROOT / "data/processed/evidence_chunks.jsonl"
VECTOR_DIR = ROOT / "vector_store"

INDEX_FILE = VECTOR_DIR / "cybersecurity_faiss.index"
METADATA_FILE = VECTOR_DIR / "chunk_metadata.jsonl"

MODEL_NAME = "all-MiniLM-L6-v2"
BATCH_SIZE = 64


def load_chunks(path):
    chunks = []

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                chunks.append(json.loads(line))

    return chunks


def save_metadata(chunks, path):
    with path.open("w", encoding="utf-8") as file:
        for chunk in chunks:
            metadata = {
                "chunk_id": chunk["chunk_id"],
                "record_id": chunk["record_id"],
                "source_dataset": chunk["source_dataset"],
                "source_record_id": chunk["source_record_id"],
                "source_version": chunk.get("source_version"),
                "record_type": chunk.get("record_type"),
                "title": chunk.get("title"),
                "chunk_index": chunk["chunk_index"],
                "text": chunk["text"],
            }

            file.write(
                json.dumps(metadata, ensure_ascii=False) + "\n"
            )


def main():
    print("=" * 65)
    print("CYBERSECURITY RAG EMBEDDING GENERATION")
    print("=" * 65)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Chunk file not found: {INPUT_FILE}"
        )

    VECTOR_DIR.mkdir(parents=True, exist_ok=True)

    chunks = load_chunks(INPUT_FILE)

    if not chunks:
        raise ValueError("No chunks found.")

    chunk_ids = [chunk["chunk_id"] for chunk in chunks]

    if len(chunk_ids) != len(set(chunk_ids)):
        raise ValueError("Duplicate chunk IDs detected.")

    texts = [
        str(chunk["text"]).strip()
        for chunk in chunks
    ]

    if any(not text for text in texts):
        raise ValueError("Empty chunk text detected.")

    print(f"\nChunks loaded: {len(chunks)}")
    print(f"Embedding model: {MODEL_NAME}")
    print(f"Batch size: {BATCH_SIZE}")

    print("\nLoading Sentence Transformer model...")
    model = SentenceTransformer(MODEL_NAME)

    print("Generating embeddings...")

    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    embeddings = np.asarray(embeddings, dtype=np.float32)

    if embeddings.ndim != 2:
        raise ValueError("Invalid embedding dimensions.")

    if not np.isfinite(embeddings).all():
        raise ValueError("Embeddings contain invalid values.")

    vector_dimension = embeddings.shape[1]

    print("\nBuilding FAISS index...")

    index = faiss.IndexFlatIP(vector_dimension)
    index.add(embeddings)

    faiss.write_index(index, str(INDEX_FILE))
    save_metadata(chunks, METADATA_FILE)

    # Reload the index to verify that it was saved correctly.
    saved_index = faiss.read_index(str(INDEX_FILE))

    if saved_index.ntotal != len(chunks):
        raise RuntimeError("FAISS index verification failed.")

    if saved_index.d != vector_dimension:
        raise RuntimeError("FAISS vector dimension mismatch.")

    print("\n" + "=" * 65)
    print("EMBEDDING SUMMARY")
    print("=" * 65)

    print(f"Embedding model: {MODEL_NAME}")
    print(f"Total chunks: {len(chunks)}")
    print(f"Embedding dimension: {vector_dimension}")
    print(f"Embedding shape: {embeddings.shape}")
    print(f"FAISS vectors: {saved_index.ntotal}")
    print(f"Index type: {type(saved_index).__name__}")
    print(f"Index file: {INDEX_FILE}")
    print(f"Metadata file: {METADATA_FILE}")

    print("\nEMBEDDINGS AND FAISS INDEX CREATED SUCCESSFULLY")


if __name__ == "__main__":
    main()
