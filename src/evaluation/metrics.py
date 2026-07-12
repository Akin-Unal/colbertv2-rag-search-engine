from __future__ import annotations

import math
from typing import Any


def passage_results_to_document_ranking(
    passage_results: list[dict[str, Any]],
    max_docs: int = 10,
) -> list[str]:
    """
    Convert passage-level retrieval results into a deduplicated
    document-level ranking.

    Example:
        passages:
            A::p0, A::p1, B::p0, C::p0

        document ranking:
            A, B, C
    """
    seen_doc_ids: set[str] = set()
    ranked_doc_ids: list[str] = []

    for result in passage_results:
        doc_id = str(result["doc_id"])

        if doc_id in seen_doc_ids:
            continue

        seen_doc_ids.add(doc_id)
        ranked_doc_ids.append(doc_id)

        if len(ranked_doc_ids) >= max_docs:
            break

    return ranked_doc_ids


def recall_at_k(
    ranked_doc_ids: list[str],
    relevant_doc_ids: set[str],
    k: int,
) -> float:
    """
    Recall@k = relevant documents retrieved in top-k / total relevant documents.
    """
    if not relevant_doc_ids:
        return 0.0

    retrieved_at_k = set(ranked_doc_ids[:k])
    found_relevant = retrieved_at_k.intersection(relevant_doc_ids)

    return len(found_relevant) / len(relevant_doc_ids)


def reciprocal_rank_at_k(
    ranked_doc_ids: list[str],
    relevant_doc_ids: set[str],
    k: int,
) -> float:
    """
    Reciprocal rank of the first relevant document within top-k.
    """
    if not relevant_doc_ids:
        return 0.0

    for rank, doc_id in enumerate(ranked_doc_ids[:k], start=1):
        if doc_id in relevant_doc_ids:
            return 1.0 / rank

    return 0.0


def dcg_at_k(
    ranked_doc_ids: list[str],
    relevance_by_doc_id: dict[str, int],
    k: int,
) -> float:
    """
    Discounted Cumulative Gain.

    Higher-ranked relevant documents contribute more than
    lower-ranked relevant documents.
    """
    dcg = 0.0

    for rank, doc_id in enumerate(ranked_doc_ids[:k], start=1):
        relevance = relevance_by_doc_id.get(doc_id, 0)

        if relevance <= 0:
            continue

        gain = (2**relevance) - 1
        discount = math.log2(rank + 1)

        dcg += gain / discount

    return dcg


def ndcg_at_k(
    ranked_doc_ids: list[str],
    relevance_by_doc_id: dict[str, int],
    k: int,
) -> float:
    """
    Normalized Discounted Cumulative Gain.
    """
    if not relevance_by_doc_id:
        return 0.0

    actual_dcg = dcg_at_k(
        ranked_doc_ids=ranked_doc_ids,
        relevance_by_doc_id=relevance_by_doc_id,
        k=k,
    )

    ideal_relevances = sorted(
        relevance_by_doc_id.values(),
        reverse=True,
    )

    ideal_dcg = 0.0

    for rank, relevance in enumerate(ideal_relevances[:k], start=1):
        if relevance <= 0:
            continue

        gain = (2**relevance) - 1
        discount = math.log2(rank + 1)

        ideal_dcg += gain / discount

    if ideal_dcg == 0:
        return 0.0

    return actual_dcg / ideal_dcg


def evaluate_single_query(
    ranked_doc_ids: list[str],
    relevance_by_doc_id: dict[str, int],
) -> dict[str, float]:
    """
    Calculate all retrieval metrics for one query.
    """
    relevant_doc_ids = {
        doc_id
        for doc_id, relevance in relevance_by_doc_id.items()
        if relevance > 0
    }

    return {
        "recall_at_5": recall_at_k(
            ranked_doc_ids=ranked_doc_ids,
            relevant_doc_ids=relevant_doc_ids,
            k=5,
        ),
        "recall_at_10": recall_at_k(
            ranked_doc_ids=ranked_doc_ids,
            relevant_doc_ids=relevant_doc_ids,
            k=10,
        ),
        "mrr_at_10": reciprocal_rank_at_k(
            ranked_doc_ids=ranked_doc_ids,
            relevant_doc_ids=relevant_doc_ids,
            k=10,
        ),
        "ndcg_at_10": ndcg_at_k(
            ranked_doc_ids=ranked_doc_ids,
            relevance_by_doc_id=relevance_by_doc_id,
            k=10,
        ),
    }


def average_metrics(
    per_query_metrics: list[dict[str, float]],
) -> dict[str, float]:
    """
    Average metric values across all evaluated queries.
    """
    if not per_query_metrics:
        raise ValueError("Cannot average an empty metric list.")

    metric_names = per_query_metrics[0].keys()

    averaged: dict[str, float] = {}

    for metric_name in metric_names:
        averaged[metric_name] = sum(
            query_metrics[metric_name]
            for query_metrics in per_query_metrics
        ) / len(per_query_metrics)

    return averaged