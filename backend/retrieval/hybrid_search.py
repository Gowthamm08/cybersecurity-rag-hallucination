
import json
import re
from pathlib import Path

import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from backend.retrieval.query_expansion import expand_query


ROOT = Path(__file__).resolve().parents[2]

CHROMA_PATH = ROOT / "vector_store" / "chromadb"
COLLECTION_NAME = "cybersecurity_evidence"

CHUNK_FILE = ROOT / "data" / "processed" / "evidence_chunks.jsonl"

MODEL_NAME = "all-MiniLM-L6-v2"

SEMANTIC_CANDIDATES = 30
BM25_CANDIDATES = 30

SEMANTIC_WEIGHT = 0.70
BM25_WEIGHT = 0.30

ORIGINAL_QUERY_WEIGHT = 0.60
EXPANDED_QUERY_WEIGHT = 0.40


def tokenize(text):
    return re.findall(r"[a-zA-Z0-9_.-]+", text.lower())


class CybersecurityHybridSearch:

    def __init__(self):

        print("Loading ChromaDB...")

        self.client = chromadb.PersistentClient(
            path=str(CHROMA_PATH)
        )

        self.collection = self.client.get_collection(
            name=COLLECTION_NAME
        )

        print("Loading embedding model...")

        self.model = SentenceTransformer(MODEL_NAME)

        print("Loading BM25 documents...")

        self.documents = self.load_documents()

        self.bm25 = BM25Okapi([
            tokenize(item["text"])
            for item in self.documents
        ])

        self.metadata_by_chunk = {
            item["chunk_id"]: item
            for item in self.documents
        }

        print("\nMULTI-QUERY HYBRID SEARCH READY")
        print("ChromaDB chunks:", self.collection.count())
        print("BM25 documents:", len(self.documents))
        print("Semantic weight:", SEMANTIC_WEIGHT)
        print("BM25 weight:", BM25_WEIGHT)
        print("Original query weight:", ORIGINAL_QUERY_WEIGHT)
        print("Expanded query weight:", EXPANDED_QUERY_WEIGHT)

    def load_documents(self):

        if not CHUNK_FILE.exists():
            raise FileNotFoundError(
                f"Chunk file not found: {CHUNK_FILE}"
            )

        records = []

        with CHUNK_FILE.open(
            "r",
            encoding="utf-8"
        ) as file:

            for line in file:

                if not line.strip():
                    continue

                item = json.loads(line)

                records.append({
                    "chunk_id": item["chunk_id"],
                    "record_id": item["record_id"],
                    "source_dataset": item["source_dataset"],
                    "source_record_id": item["source_record_id"],
                    "source_version": item.get("source_version"),
                    "record_type": item.get("record_type"),
                    "title": item.get("title"),
                    "text": item["text"]
                })

        return records

    def semantic_search(self, query):

        embedding = self.model.encode(
            [query],
            normalize_embeddings=True
        )

        response = self.collection.query(
            query_embeddings=embedding.tolist(),
            n_results=min(
                SEMANTIC_CANDIDATES,
                self.collection.count()
            ),
            include=[
                "documents",
                "metadatas",
                "distances"
            ]
        )

        results = []

        for rank, chunk_id in enumerate(
            response["ids"][0],
            start=1
        ):

            metadata = response["metadatas"][0][rank - 1]

            results.append({
                "chunk_id": chunk_id,
                "source_record_id": metadata.get(
                    "source_record_id"
                ),
                "similarity_score": (
                    1.0 - response["distances"][0][rank - 1]
                ),
                "rank": rank
            })

        return results

    def keyword_search(self, query):

        tokens = tokenize(query)

        if not tokens:
            return []

        scores = self.bm25.get_scores(tokens)

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda index: scores[index],
            reverse=True
        )

        results = []

        for index in ranked_indices[:BM25_CANDIDATES]:

            score = float(scores[index])

            if score <= 0:
                continue

            item = self.documents[index]

            results.append({
                "chunk_id": item["chunk_id"],
                "source_record_id": item["source_record_id"],
                "bm25_score": score,
                "rank": len(results) + 1
            })

        return results

    @staticmethod
    def normalize_scores(scores):

        if not scores:
            return {}

        minimum = min(scores.values())
        maximum = max(scores.values())

        if maximum == minimum:
            return {
                key: 1.0
                for key in scores
            }

        return {
            key: (value - minimum) / (maximum - minimum)
            for key, value in scores.items()
        }

    def search(self, query, top_k=5):

        expanded_query = expand_query(query)

        # Retrieve original query candidates.
        original_semantic = self.semantic_search(query)
        original_keyword = self.keyword_search(query)

        # Retrieve expanded query candidates.
        expanded_semantic = self.semantic_search(expanded_query)
        expanded_keyword = self.keyword_search(expanded_query)

        # Keep separate score maps for each retrieval method.
        original_semantic_scores = {
            item["chunk_id"]: item["similarity_score"]
            for item in original_semantic
        }

        expanded_semantic_scores = {
            item["chunk_id"]: item["similarity_score"]
            for item in expanded_semantic
        }

        original_keyword_scores = {
            item["chunk_id"]: item["bm25_score"]
            for item in original_keyword
        }

        expanded_keyword_scores = {
            item["chunk_id"]: item["bm25_score"]
            for item in expanded_keyword
        }

        # Normalize each query's scores independently.
        original_semantic_norm = self.normalize_scores(
            original_semantic_scores
        )

        expanded_semantic_norm = self.normalize_scores(
            expanded_semantic_scores
        )

        original_keyword_norm = self.normalize_scores(
            original_keyword_scores
        )

        expanded_keyword_norm = self.normalize_scores(
            expanded_keyword_scores
        )

        all_chunk_ids = (
            set(original_semantic_norm)
            | set(expanded_semantic_norm)
            | set(original_keyword_norm)
            | set(expanded_keyword_norm)
        )

        ranked_chunks = []

        for chunk_id in all_chunk_ids:

            # Weighted semantic score across both queries.
            semantic_score = (
                ORIGINAL_QUERY_WEIGHT
                * original_semantic_norm.get(chunk_id, 0.0)
                +
                EXPANDED_QUERY_WEIGHT
                * expanded_semantic_norm.get(chunk_id, 0.0)
            )

            # Weighted BM25 score across both queries.
            keyword_score = (
                ORIGINAL_QUERY_WEIGHT
                * original_keyword_norm.get(chunk_id, 0.0)
                +
                EXPANDED_QUERY_WEIGHT
                * expanded_keyword_norm.get(chunk_id, 0.0)
            )

            # Final hybrid score.
            final_score = (
                SEMANTIC_WEIGHT * semantic_score
                +
                BM25_WEIGHT * keyword_score
            )

            ranked_chunks.append({
                "chunk_id": chunk_id,
                "fusion_score": final_score,
                "semantic_score": semantic_score,
                "keyword_score": keyword_score
            })

        ranked_chunks.sort(
            key=lambda item: (
                -item["fusion_score"],
                item["chunk_id"]
            )
        )

        # Keep the best-ranked chunk for each evidence record.
        best_by_record = {}

        for result in ranked_chunks:

            chunk_id = result["chunk_id"]

            item = self.metadata_by_chunk.get(chunk_id)

            if item is None:
                continue

            record_id = item["source_record_id"]

            if record_id in best_by_record:
                continue

            best_by_record[record_id] = {
                "chunk_id": chunk_id,
                "title": item["title"],
                "source_dataset": item["source_dataset"],
                "source_record_id": record_id,
                "record_type": item["record_type"],
                "text": item["text"],
                "fusion_score": result["fusion_score"],
                "semantic_score": result["semantic_score"],
                "keyword_score": result["keyword_score"],
                "match_type": "multi_query_weighted_hybrid"
            }

            if len(best_by_record) >= top_k:
                break

        return list(best_by_record.values())


if __name__ == "__main__":

    search_engine = CybersecurityHybridSearch()

    while True:

        query = input(
            "\nEnter cybersecurity question (or 'exit'): "
        ).strip()

        if query.lower() in {"exit", "quit"}:
            break

        if not query:
            continue

        expanded = expand_query(query)

        print("\nExpanded query:")
        print(expanded)

        results = search_engine.search(
            query,
            top_k=5
        )

        for rank, result in enumerate(results, 1):

            print(f"\nResult {rank}")

            print("Record:", result["source_record_id"])
            print("Fusion score:", round(result["fusion_score"], 5))
            print("Semantic score:", round(result["semantic_score"], 4))
            print("Keyword score:", round(result["keyword_score"], 4))
            print("Title:", result["title"])
            print("Evidence:", result["text"][:500])
