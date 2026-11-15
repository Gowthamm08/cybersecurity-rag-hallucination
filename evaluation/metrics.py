
import re
from collections import Counter


def precision_at_k(retrieved_ids, relevant_ids, k=5):

    if k <= 0:
        raise ValueError("k must be greater than zero.")

    retrieved = list(dict.fromkeys(retrieved_ids))[:k]
    relevant = set(relevant_ids)

    if not retrieved:
        return 0.0

    relevant_retrieved = sum(
        1 for item in retrieved if item in relevant
    )

    return relevant_retrieved / k


def recall_at_k(retrieved_ids, relevant_ids, k=5):

    retrieved = list(dict.fromkeys(retrieved_ids))[:k]
    relevant = set(relevant_ids)

    if not relevant:
        return 0.0

    relevant_retrieved = sum(
        1 for item in retrieved if item in relevant
    )

    return relevant_retrieved / len(relevant)


def reciprocal_rank(retrieved_ids, relevant_ids):

    relevant = set(relevant_ids)

    for rank, item in enumerate(retrieved_ids, start=1):

        if item in relevant:
            return 1.0 / rank

    return 0.0


def mean_reciprocal_rank(rank_values):

    if not rank_values:
        return 0.0

    return sum(rank_values) / len(rank_values)


def calculate_retrieval_metrics(results, k=5):

    if not results:
        return {}

    precision_values = []
    recall_values = []
    reciprocal_rank_values = []

    for result in results:

        retrieved = result.get("retrieved_ids", [])
        relevant = result.get("relevant_ids", [])

        precision_values.append(
            precision_at_k(retrieved, relevant, k)
        )

        recall_values.append(
            recall_at_k(retrieved, relevant, k)
        )

        reciprocal_rank_values.append(
            reciprocal_rank(retrieved, relevant)
        )

    return {
        f"precision@{k}": (
            sum(precision_values) / len(precision_values)
        ),
        f"recall@{k}": (
            sum(recall_values) / len(recall_values)
        ),
        "mrr": mean_reciprocal_rank(
            reciprocal_rank_values
        ),
        "evaluated_queries": len(results)
    }


def normalize_text(text):

    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    return " ".join(text.split())


def exact_match(prediction, reference):

    return int(
        normalize_text(prediction)
        == normalize_text(reference)
    )


def token_f1(prediction, reference):

    prediction_tokens = normalize_text(
        prediction
    ).split()

    reference_tokens = normalize_text(
        reference
    ).split()

    if not prediction_tokens or not reference_tokens:
        return 0.0

    prediction_counts = Counter(prediction_tokens)
    reference_counts = Counter(reference_tokens)

    common = prediction_counts & reference_counts

    overlap = sum(common.values())

    if overlap == 0:
        return 0.0

    precision = overlap / len(prediction_tokens)
    recall = overlap / len(reference_tokens)

    return (
        2 * precision * recall /
        (precision + recall)
    )


def calculate_answer_metrics(results):

    if not results:
        return {}

    exact_values = []
    f1_values = []

    for result in results:

        prediction = result.get("prediction", "")
        reference = result.get("reference", "")

        exact_values.append(
            exact_match(prediction, reference)
        )

        f1_values.append(
            token_f1(prediction, reference)
        )

    return {
        "exact_match": sum(exact_values) / len(exact_values),
        "token_f1": sum(f1_values) / len(f1_values),
        "evaluated_answers": len(results)
    }


if __name__ == "__main__":

    print("Testing evaluation metrics...")

    retrieved = [
        "SEC-RAG-0001",
        "SEC-RAG-0003",
        "SEC-RAG-0002"
    ]

    relevant = ["SEC-RAG-0002"]

    print(
        "Precision@3:",
        precision_at_k(retrieved, relevant, 3)
    )

    print(
        "Recall@3:",
        recall_at_k(retrieved, relevant, 3)
    )

    print(
        "Reciprocal Rank:",
        reciprocal_rank(retrieved, relevant)
    )

    prediction = "SQL injection vulnerability"
    reference = "SQL injection vulnerability"

    print(
        "Exact Match:",
        exact_match(prediction, reference)
    )

    print(
        "Token F1:",
        token_f1(prediction, reference)
    )

    print("\nMETRICS TEST COMPLETED")
