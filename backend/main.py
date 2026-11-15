
import json
from datetime import datetime
from pathlib import Path

from backend.generation.rag_generator import CybersecurityRAG
from backend.verification.claim_decomposition import ClaimDecomposer
from backend.verification.local_verifier import LocalVerifier
from backend.verification.global_verifier import GlobalVerifier
from backend.verification.classifier import ClaimClassifier


ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = ROOT / "data" / "processed"

OUTPUT_FILE = OUTPUT_DIR / "full_pipeline_results.json"


class CybersecurityRAGPipeline:

    def __init__(self, top_k=5):

        print("\nInitializing Cybersecurity RAG Pipeline...")

        self.rag = CybersecurityRAG(top_k=top_k)

        self.decomposer = ClaimDecomposer()

        self.local_verifier = LocalVerifier()

        self.global_verifier = GlobalVerifier()

        self.classifier = ClaimClassifier()

        print("\nAll pipeline components initialized successfully.")

    def run(self, question):

        if not question or not question.strip():
            raise ValueError("Question cannot be empty.")

        print("\n" + "=" * 65)
        print("STEP 1: RETRIEVING CYBERSECURITY EVIDENCE")
        print("=" * 65)

        rag_result = self.rag.ask(question)

        evidence = rag_result["evidence"]
        answer = rag_result["answer"]

        print(f"Retrieved evidence records: {len(evidence)}")

        if not evidence:
            return {
                "question": question,
                "answer": answer,
                "evidence": [],
                "claims": [],
                "local_verification": [],
                "global_verification": [],
                "classification": {
                    "summary": {
                        "total_claims": 0,
                        "entailed": 0,
                        "contradicted": 0,
                        "baseless": 0,
                        "unverified": 0
                    },
                    "results": []
                }
            }

        print("\n" + "=" * 65)
        print("STEP 2: GENERATING CYBERSECURITY ANSWER")
        print("=" * 65)

        print("\nGenerated answer:\n")
        print(answer)

        print("\n" + "=" * 65)
        print("STEP 3: CLAIM DECOMPOSITION")
        print("=" * 65)

        claims = self.decomposer.decompose(answer)

        if not claims:
            raise RuntimeError(
                "Claim decomposition returned no claims."
            )

        for claim in claims:
            print(
                f"{claim['claim_id']}: {claim['claim']}"
            )

        print(f"\nTotal claims extracted: {len(claims)}")

        print("\n" + "=" * 65)
        print("STEP 4: LOCAL CLAIM VERIFICATION")
        print("=" * 65)

        local_results = self.local_verifier.verify_claims(
            claims,
            evidence
        )

        print("\nLocal verification completed.")

        print("\n" + "=" * 65)
        print("STEP 5: GLOBAL CLAIM VERIFICATION")
        print("=" * 65)

        global_results = self.global_verifier.verify_claims(
            claims,
            evidence
        )

        print("\nGlobal verification completed.")

        print("\n" + "=" * 65)
        print("STEP 6: CLAIM CLASSIFICATION")
        print("=" * 65)

        classified_results = self.classifier.classify_claims(
            global_results
        )

        summary = self.classifier.generate_summary(
            classified_results
        )

        # Combine local and global verification for each claim.
        local_by_id = {
            item.get("claim_id"): item
            for item in local_results
        }

        for result in classified_results:

            local_result = local_by_id.get(
                result["claim_id"],
                {}
            )

            result["local_verification"] = local_result

        print("\nFinal classification:")

        for result in classified_results:

            print(
                f"\n{result['claim_id']}: "
                f"{result['classification']}"
            )

            print(f"Claim: {result['claim']}")

            local_result = result.get(
                "local_verification", {}
            )

            print(
                "Best local similarity:",
                local_result.get("best_similarity")
            )

            print(
                "Global verdict:",
                result["global_verdict"]
            )

        print("\n" + "=" * 65)
        print("FINAL CLASSIFICATION SUMMARY")
        print("=" * 65)

        for key, value in summary.items():
            print(f"{key}: {value}")

        complete_result = {
            "timestamp": datetime.now().astimezone().isoformat(),
            "question": question,
            "answer": answer,
            "evidence": evidence,
            "claims": claims,
            "local_verification": local_results,
            "global_verification": global_results,
            "classification": {
                "summary": summary,
                "results": classified_results
            }
        }

        OUTPUT_DIR.mkdir(
            parents=True,
            exist_ok=True
        )

        with open(
            OUTPUT_FILE,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                complete_result,
                file,
                indent=4,
                ensure_ascii=False
            )

        print("\nComplete pipeline results saved to:")
        print(OUTPUT_FILE)

        return complete_result


def main():

    print("\n" + "=" * 65)
    print("       EVIDENCE-GROUNDED CYBERSECURITY RAG")
    print("=" * 65)

    print("\nTechnology Stack:")
    print("- ChromaDB")
    print("- Sentence-BERT")
    print("- Gemini")
    print("- Local Verification")
    print("- Global Verification")
    print("- Claim Classification")

    try:

        pipeline = CybersecurityRAGPipeline(top_k=5)

    except Exception as error:

        print("\nPipeline initialization failed:")
        print(error)
        return

    while True:

        question = input(
            "\nAsk a cybersecurity question (or type 'exit'): "
        ).strip()

        if question.lower() in {"exit", "quit"}:

            print("\nExiting Cybersecurity RAG Pipeline.")
            break

        if not question:
            print("Please enter a question.")
            continue

        try:

            result = pipeline.run(question)

            print("\n" + "=" * 65)
            print("PIPELINE EXECUTION COMPLETED")
            print("=" * 65)

            print("\nYour answer:")
            print(result["answer"])

            print("\nFinal classifications:")

            for item in result["classification"]["results"]:

                print(
                    f"{item['claim_id']} - "
                    f"{item['classification']}"
                )

        except Exception as error:

            print("\nPipeline execution failed:")
            print(f"{type(error).__name__}: {error}")

            print(
                "\nCheck the error before attempting "
                "the next question."
            )


if __name__ == "__main__":
    main()
