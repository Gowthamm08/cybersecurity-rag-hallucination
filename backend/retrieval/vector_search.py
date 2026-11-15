
import json
import re
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[2]

VECTOR_DIR = ROOT / "vector_store"

INDEX_FILE = VECTOR_DIR / "cybersecurity_faiss.index"
METADATA_FILE = VECTOR_DIR / "chunk_metadata.jsonl"

MODEL_NAME = "all-MiniLM-L6-v2"


class CybersecurityVectorSearch:

    def __init__(self):
        if not INDEX_FILE.exists():
            raise FileNotFoundError("FAISS index not found.")

        if not METADATA_FILE.exists():
            raise FileNotFoundError("Chunk metadata not found.")

        print("Loading embedding model...")
        self.model = SentenceTransformer(MODEL_NAME)

        print("Loading FAISS index...")
        self.index = faiss.read_index(str(INDEX_FILE))

        print("Loading chunk metadata...")
        self.metadata = []

        with METADATA_FILE.open("r", encoding="utf-8") as file:
            for line in file:
                if line.strip():
                    self.metadata.append(json.loads(line))

        if self.index.ntotal != len(self.metadata):
            raise ValueError(
                "FAISS index and metadata count do not match."
            )

        print(f"Loaded vectors: {self.index.ntotal}")

    def exact_identifier_search(self, query, top_k):
        identifiers = re.findall(
            r"\bCVE-\d{4}-\d{4,}\b",
            query,
            flags=re.IGNORECASE
        )

        if not identifiers:
            return None

        identifier = identifiers[0].upper()
        results = []

        for record in self.metadata:
            searchable_fields = [
                str(record.get("title", "")),
                str(record.get("text", "")),
                str(record.get("source_record_id", ""))
            ]

            if any(
                re.search(
                    rf"\b{re.escape(identifier)}\b",
                    field,
                    flags=re.IGNORECASE
                )
                for field in searchable_fields
            ):
                result = record.copy()
                result["similarity_score"] = 1.0
                result["match_type"] = "exact_identifier"
                results.append(result)

        if results:
            return results[:top_k]

        return None

    def search(self, query, top_k=5):

        if not query or not query.strip():
            raise ValueError("Search query cannot be empty.")

        if top_k < 1:
            raise ValueError("top_k must be at least 1.")

        # First, try exact CVE identifier matching.
        exact_results = self.exact_identifier_search(
            query,
            top_k
        )

        if exact_results is not None:
            return exact_results

        # Otherwise, perform semantic vector search.
        query_vector = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        query_vector = np.asarray(
            query_vector,
            dtype=np.float32
        )

        candidate_count = min(
            top_k * 10,
            self.index.ntotal
        )

        scores, indices = self.index.search(
            query_vector,
            candidate_count
        )

        results = []
        seen_record_ids = set()

        for score, index in zip(scores[0], indices[0]):

            if index < 0:
                continue

            record = self.metadata[int(index)]

            record_id = record.get("record_id")

            if record_id in seen_record_ids:
                continue

            seen_record_ids.add(record_id)

            result = record.copy()
            result["similarity_score"] = float(score)
            result["match_type"] = "semantic"

            results.append(result)

            if len(results) >= top_k:
                break

        return results


def main():

    print("=" * 65)
    print("CYBERSECURITY RAG SEMANTIC SEARCH")
    print("=" * 65)

    search_engine = CybersecurityVectorSearch()

    while True:

        query = input(
            "\nEnter your cybersecurity question (or 'exit'): "
        ).strip()

        if query.lower() in {"exit", "quit"}:
            print("Search session ended.")
            break

        try:
            results = search_engine.search(query, top_k=5)

            print("\n" + "=" * 65)
            print("TOP RETRIEVED EVIDENCE")
            print("=" * 65)

            if not results:
                print("\nNo matching evidence found.")
                continue

            for position, result in enumerate(results, start=1):

                print(f"\n[{position}] {result['title']}")
                print(f"Source: {result['source_dataset']}")
                print(f"Record type: {result['record_type']}")
                print(f"Match type: {result['match_type']}")
                print(
                    f"Similarity: "
                    f"{result['similarity_score']:.4f}"
                )

                print("\nEvidence:")
                print(result["text"][:700])

                print("-" * 65)

        except Exception as error:
            print(f"Search error: {error}")


if __name__ == "__main__":
    main()
