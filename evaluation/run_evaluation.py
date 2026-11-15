
import json
import argparse
from pathlib import Path

from backend.retrieval.chroma_search import (
    CybersecurityChromaSearch
)

from evaluation.dataset import load_evaluation_dataset

from evaluation.metrics import calculate_retrieval_metrics


ROOT = Path(__file__).resolve().parents[1]

OUTPUT_FILE = (
    ROOT / "data" / "evaluation" /
    "retrieval_evaluation_results.json"
)


def run_evaluation(limit=20, top_k=5):

    print("\nCYBERSECURITY RAG RETRIEVAL EVALUATION")
    print("=" * 60)

    dataset, report = load_evaluation_dataset()

    print("\nDataset mapping:")
    print(json.dumps(report, indent=4))

    if not dataset:
        raise ValueError("No valid evaluation records found.")

    if report["missing_evidence_records"]:
        raise ValueError(
            "Missing evidence mappings detected. "
            "Fix the dataset before evaluation."
        )

    if limit is not None:
        dataset = dataset[:limit]

    print(f"\nQuestions selected: {len(dataset)}")
    print(f"Retrieval top_k: {top_k}")

    search_engine = CybersecurityChromaSearch()

    results = []

    for index, item in enumerate(dataset, start=1):

        question = item["question"]
        relevant_ids = item["relevant_record_ids"]

        print(
            f"\n[{index}/{len(dataset)}] "
            f"{item['evaluation_id']}"
        )

        print("Question:", question)

        evidence = search_engine.search(
            question,
            top_k=top_k
        )

        retrieved_ids = list(dict.fromkeys(
            record.get("source_record_id")
            for record in evidence
            if record.get("source_record_id")
        ))

        relevant_found = any(
            record_id in relevant_ids
            for record_id in retrieved_ids
        )

        print("Retrieved:", retrieved_ids)
        print("Relevant record found:", relevant_found)

        results.append({
            "evaluation_id": item["evaluation_id"],
            "question": question,
            "relevant_ids": relevant_ids,
            "retrieved_ids": retrieved_ids,
            "relevant_found": relevant_found
        })

    metrics = calculate_retrieval_metrics(
        results,
        k=top_k
    )

    output = {
        "dataset_size": len(dataset),
        "top_k": top_k,
        "metrics": metrics,
        "results": results
    }

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            output,
            file,
            indent=4,
            ensure_ascii=False
        )

    print("\n" + "=" * 60)
    print("RETRIEVAL EVALUATION SUMMARY")
    print("=" * 60)

    for name, value in metrics.items():

        if isinstance(value, float):
            print(f"{name}: {value:.4f}")
        else:
            print(f"{name}: {value}")

    print("\nResults saved to:")
    print(OUTPUT_FILE)

    print("\nRETRIEVAL EVALUATION COMPLETED")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Number of evaluation questions. Use 0 for all."
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5
    )

    args = parser.parse_args()

    limit = None if args.limit == 0 else args.limit

    run_evaluation(
        limit=limit,
        top_k=args.top_k
    )
