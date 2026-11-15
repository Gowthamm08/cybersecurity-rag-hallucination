
import re
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


ROOT = Path(__file__).resolve().parents[2]

CHROMA_PATH = ROOT / "vector_store" / "chromadb"
COLLECTION_NAME = "cybersecurity_evidence"

MODEL_NAME = "all-MiniLM-L6-v2"


class CybersecurityChromaSearch:

    def __init__(self):
        self.client = chromadb.PersistentClient(
            path=str(CHROMA_PATH)
        )

        self.collection = self.client.get_collection(
            name=COLLECTION_NAME
        )

        self.model = SentenceTransformer(MODEL_NAME)

        print(
            f"ChromaDB loaded: {self.collection.count()} chunks"
        )

    def extract_cve(self, query):
        match = re.search(
            r"\bCVE-\d{4}-\d{4,}\b",
            query,
            re.IGNORECASE
        )

        return match.group(0).upper() if match else None

    def search(self, query, top_k=5):

        cve_id = self.extract_cve(query)

        # Exact CVE lookup first
        if cve_id:
            exact = self.collection.get(
                where={
                    "source_record_id": cve_id
                },
                include=[
                    "documents",
                    "metadatas"
                ]
            )

            if exact["ids"]:
                results = []

                for i in range(len(exact["ids"])):
                    results.append({
                        "chunk_id": exact["ids"][i],
                        "title": exact["metadatas"][i].get("title"),
                        "source_dataset": exact["metadatas"][i].get(
                            "source_dataset"
                        ),
                        "source_record_id": exact["metadatas"][i].get(
                            "source_record_id"
                        ),
                        "record_type": exact["metadatas"][i].get(
                            "record_type"
                        ),
                        "text": exact["documents"][i],
                        "similarity_score": 1.0,
                        "match_type": "exact_identifier"
                    })

                return results[:top_k]

        # Semantic retrieval
        embedding = self.model.encode(
            [query],
            normalize_embeddings=True
        )

        response = self.collection.query(
            query_embeddings=embedding.tolist(),
            n_results=top_k,
            include=[
                "documents",
                "metadatas",
                "distances"
            ]
        )

        results = []

        for i in range(len(response["ids"][0])):
            metadata = response["metadatas"][0][i]

            results.append({
                "chunk_id": response["ids"][0][i],
                "title": metadata.get("title"),
                "source_dataset": metadata.get("source_dataset"),
                "source_record_id": metadata.get(
                    "source_record_id"
                ),
                "record_type": metadata.get("record_type"),
                "text": response["documents"][0][i],
                "similarity_score": 1.0 - response["distances"][0][i],
                "match_type": "semantic"
            })

        return results


if __name__ == "__main__":

    search_engine = CybersecurityChromaSearch()

    while True:
        query = input(
            "\nEnter cybersecurity question (or 'exit'): "
        ).strip()

        if query.lower() in {"exit", "quit"}:
            break

        if not query:
            continue

        results = search_engine.search(query)

        for i, result in enumerate(results, 1):
            print(f"\nResult {i}")
            print("Title:", result["title"])
            print("Source:", result["source_dataset"])
            print("Match:", result["match_type"])
            print(
                "Similarity:",
                round(result["similarity_score"], 4)
            )
            print("Evidence:", result["text"][:700])
