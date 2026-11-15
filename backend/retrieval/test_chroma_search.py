
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


ROOT = Path(__file__).resolve().parents[2]

CHROMA_PATH = ROOT / "vector_store" / "chromadb"
COLLECTION_NAME = "cybersecurity_evidence"

MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K = 5


def main():
    print("Loading ChromaDB...", flush=True)

    client = chromadb.PersistentClient(
        path=str(CHROMA_PATH)
    )

    collection = client.get_collection(
        name=COLLECTION_NAME
    )

    print("Total evidence chunks:", collection.count())

    print("\nLoading embedding model...", flush=True)

    model = SentenceTransformer(MODEL_NAME)

    while True:
        query = input(
            "\nEnter cybersecurity question (or 'exit'): "
        ).strip()

        if query.lower() in {"exit", "quit"}:
            break

        if not query:
            continue

        query_embedding = model.encode(
            [query],
            normalize_embeddings=True
        )

        results = collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=TOP_K,
            include=[
                "documents",
                "metadatas",
                "distances"
            ]
        )

        print("\n========== CHROMADB RESULTS ==========")

        for i in range(len(results["ids"][0])):
            metadata = results["metadatas"][0][i]
            document = results["documents"][0][i]
            distance = results["distances"][0][i]

            similarity = 1.0 - distance

            print(f"\nResult {i + 1}")
            print("Title:", metadata.get("title"))
            print("Source:", metadata.get("source_dataset"))
            print("Record type:", metadata.get("record_type"))
            print("Source ID:", metadata.get("source_record_id"))
            print(f"Similarity: {similarity:.4f}")
            print("Evidence:", document[:700])

        print("\n======================================")


if __name__ == "__main__":
    main()
