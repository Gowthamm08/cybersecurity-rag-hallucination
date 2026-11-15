
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


ROOT = Path(__file__).resolve().parents[2]

MODEL_NAME = "all-MiniLM-L6-v2"


class LocalVerifier:

    def __init__(self):
        print("Loading local verification model...")

        self.model = SentenceTransformer(MODEL_NAME)

        print("Local verifier ready.")

    def verify_claim(self, claim, evidence, top_n=3):
        """
        Find the most semantically relevant evidence chunks
        for an individual claim.

        Similarity is a candidate-support signal, not proof
        of entailment.
        """

        if not claim or not claim.strip():
            raise ValueError("Claim cannot be empty.")

        if not evidence:
            return {
                "claim": claim,
                "matches": [],
                "best_similarity": None
            }

        evidence_texts = [
            item.get("text", "")
            for item in evidence
        ]

        valid_indices = [
            i for i, text in enumerate(evidence_texts)
            if text.strip()
        ]

        if not valid_indices:
            return {
                "claim": claim,
                "matches": [],
                "best_similarity": None
            }

        valid_texts = [
            evidence_texts[i] for i in valid_indices
        ]

        claim_embedding = self.model.encode(
            [claim],
            normalize_embeddings=True
        )

        evidence_embeddings = self.model.encode(
            valid_texts,
            normalize_embeddings=True
        )

        scores = np.dot(
            evidence_embeddings,
            claim_embedding[0]
        )

        ranked_indices = np.argsort(scores)[::-1][:top_n]

        matches = []

        for index in ranked_indices:
            original_index = valid_indices[index]
            item = evidence[original_index]

            matches.append({
                "chunk_id": item.get("chunk_id"),
                "title": item.get("title"),
                "source_dataset": item.get("source_dataset"),
                "source_record_id": item.get("source_record_id"),
                "similarity_score": round(
                    float(scores[index]), 4
                ),
                "evidence_text": item.get("text", "")
            })

        best_similarity = (
            matches[0]["similarity_score"]
            if matches else None
        )

        return {
            "claim": claim,
            "matches": matches,
            "best_similarity": best_similarity
        }

    def verify_claims(self, claims, evidence):
        """Run local evidence matching for every claim."""

        results = []

        for item in claims:
            result = self.verify_claim(
                claim=item["claim"],
                evidence=evidence
            )

            result["claim_id"] = item["claim_id"]

            results.append(result)

        return results


if __name__ == "__main__":

    from backend.retrieval.chroma_search import (
        CybersecurityChromaSearch
    )

    search_engine = CybersecurityChromaSearch()
    verifier = LocalVerifier()

    question = "What is CVE-2026-102910?"

    evidence = search_engine.search(question, top_k=5)

    sample_claims = [
        {
            "claim_id": "C1",
            "claim": (
                "CVE-2026-102910 is a SQL injection vulnerability."
            )
        },
        {
            "claim_id": "C2",
            "claim": (
                "CVE-2026-102910 affects "
                "SourceCodester Online Reviewer Management System 1.0."
            )
        },
        {
            "claim_id": "C3",
            "claim": (
                "The vulnerability affects exam-delete.php."
            )
        },
        {
            "claim_id": "C4",
            "claim": (
                "The vulnerability can be exploited remotely."
            )
        }
    ]

    print("\nRunning local claim verification...\n")

    results = verifier.verify_claims(
        sample_claims,
        evidence
    )

    for result in results:
        print("\nClaim:", result["claim_id"])
        print(result["claim"])

        print(
            "Best similarity:",
            result["best_similarity"]
        )

        for match in result["matches"]:
            print(
                f"  Evidence: {match['title']} | "
                f"Similarity: {match['similarity_score']}"
            )

    print("\nLOCAL VERIFICATION TEST COMPLETED")
