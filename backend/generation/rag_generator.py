
import os
from pathlib import Path

import requests
from dotenv import load_dotenv

from backend.retrieval.chroma_search import CybersecurityChromaSearch


ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

GEMINI_MODEL = "gemini-3.5-flash-lite"

GEMINI_URL = (
    f"https://generativelanguage.googleapis.com/"
    f"v1beta/models/{GEMINI_MODEL}:generateContent"
)


class CybersecurityRAG:

    def __init__(self, top_k=5):
        self.api_key = os.getenv("GEMINI_API_KEY")

        if not self.api_key:
            raise ValueError(
                "GEMINI_API_KEY not found. Check your project .env file."
            )

        self.search_engine = CybersecurityChromaSearch()
        self.top_k = top_k

        self.session = requests.Session()

    def retrieve_evidence(self, question):
        return self.search_engine.search(
            question,
            top_k=self.top_k
        )

    def build_context(self, evidence):
        if not evidence:
            return "No relevant evidence was retrieved."

        sections = []

        for i, item in enumerate(evidence, start=1):
            sections.append(
                f"""
[E{i}]
Title: {item.get('title')}
Source: {item.get('source_dataset')}
Record ID: {item.get('source_record_id')}
Record Type: {item.get('record_type')}
Evidence:
{item.get('text')}
"""
            )

        return "\n".join(sections)

    def generate_answer(self, question, evidence):
        context = self.build_context(evidence)

        prompt = f"""
You are a cybersecurity research assistant.

Answer the user's question using ONLY the supplied evidence.

STRICT RULES:
1. Do not invent CVE details, attack techniques, or facts.
2. Every factual claim must have an evidence citation.
3. Cite evidence using labels such as [E1] and [E2].
4. If the evidence does not answer the question, clearly say so.
5. If sources disagree, explain the disagreement.
6. Distinguish evidence-supported facts from uncertainty.

USER QUESTION:
{question}

RETRIEVED EVIDENCE:
{context}

Provide:
1. Direct answer.
2. Supporting explanation.
3. Evidence citations.
4. Any limitations in the available evidence.
"""

        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt}
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.1,
                "maxOutputTokens": 1200
            }
        }

        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": self.api_key
        }

        response = self.session.post(
            GEMINI_URL,
            headers=headers,
            json=payload,
            timeout=(10, 60)
        )

        if not response.ok:
            raise RuntimeError(
                f"Gemini API error {response.status_code}: "
                f"{response.text}"
            )

        result = response.json()

        try:
            return result["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError, TypeError):
            raise RuntimeError(
                f"Unexpected Gemini response: {result}"
            )

    def ask(self, question):
        print("\nRetrieving cybersecurity evidence...")

        evidence = self.retrieve_evidence(question)

        print(f"Retrieved evidence records: {len(evidence)}")

        if not evidence:
            return {
                "question": question,
                "answer": (
                    "I cannot answer this question because "
                    "no relevant evidence was retrieved."
                ),
                "evidence": []
            }

        answer = self.generate_answer(question, evidence)

        return {
            "question": question,
            "answer": answer,
            "evidence": evidence
        }


def main():
    rag = CybersecurityRAG(top_k=5)

    print("\n========================================")
    print(" CYBERSECURITY RAG ASSISTANT")
    print(" ChromaDB + Sentence-BERT + Gemini")
    print("========================================")

    while True:
        question = input(
            "\nAsk a cybersecurity question (or 'exit'): "
        ).strip()

        if question.lower() in {"exit", "quit"}:
            print("Exiting Cybersecurity RAG Assistant.")
            break

        if not question:
            continue

        try:
            result = rag.ask(question)

            print("\n========== GENERATED ANSWER ==========\n")
            print(result["answer"])

            print("\n========== RETRIEVED SOURCES ==========\n")

            for i, item in enumerate(result["evidence"], start=1):
                print(f"[E{i}] {item['title']}")
                print(f"Source: {item['source_dataset']}")
                print(f"Record ID: {item['source_record_id']}")
                print(
                    f"Match type: {item['match_type']}"
                )
                print(
                    f"Similarity: {item['similarity_score']:.4f}"
                )
                print()

        except requests.exceptions.Timeout:
            print("Gemini request timed out. Please try again.")

        except requests.exceptions.RequestException as error:
            print("Network error:", error)

        except Exception as error:
            print("RAG pipeline error:", error)


if __name__ == "__main__":
    main()
