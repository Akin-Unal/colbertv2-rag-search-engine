from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.retrieval.bm25 import BM25Retriever
from src.retrieval.dense import DenseRetriever


DEFAULT_PASSAGES_PATH = Path("data/processed/passages.jsonl")
DEFAULT_DENSE_INDEX_DIR = Path("indexes/dense")


def shorten(text: str, max_length: int = 85) -> str:
    """
    Convert text into a compact single-line preview.
    """
    text = " ".join(text.split())

    if len(text) <= max_length:
        return text

    return text[: max_length - 3].rstrip() + "..."


def print_method_results(
    method_name: str,
    results: list[dict[str, Any]],
) -> None:
    """
    Print retrieval results in a compact table-like format.
    """
    print()
    print("=" * 120)
    print(method_name)
    print("=" * 120)

    header = (
        f"{'Rank':<6}"
        f"{'Score':<12}"
        f"{'Doc ID':<15}"
        f"{'Passage ID':<22}"
        f"Title"
    )

    print(header)
    print("-" * 120)

    for result in results:
        print(
            f"{result['rank']:<6}"
            f"{result['score']:<12.4f}"
            f"{str(result['doc_id']):<15}"
            f"{result['passage_id']:<22}"
            f"{shorten(result['title'])}"
        )


def unique_document_ids(
    results: list[dict[str, Any]],
) -> list[str]:
    """
    Return document IDs while preserving retrieval order
    and removing duplicates.
    """
    seen: set[str] = set()
    ordered_document_ids: list[str] = []

    for result in results:
        doc_id = str(result["doc_id"])

        if doc_id in seen:
            continue

        seen.add(doc_id)
        ordered_document_ids.append(doc_id)

    return ordered_document_ids


def create_comparison_summary(
    bm25_results: list[dict[str, Any]],
    dense_results: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Compare passage-level and document-level overlap.
    """
    bm25_passage_ids = {
        result["passage_id"]
        for result in bm25_results
    }

    dense_passage_ids = {
        result["passage_id"]
        for result in dense_results
    }

    bm25_document_ids = unique_document_ids(bm25_results)
    dense_document_ids = unique_document_ids(dense_results)

    common_passages = sorted(
        bm25_passage_ids.intersection(dense_passage_ids)
    )

    common_documents = sorted(
        set(bm25_document_ids).intersection(dense_document_ids)
    )

    union_documents = set(bm25_document_ids).union(
        dense_document_ids
    )

    document_jaccard = (
        len(common_documents) / len(union_documents)
        if union_documents
        else 0.0
    )

    return {
        "common_passage_count": len(common_passages),
        "common_document_count": len(common_documents),
        "common_passage_ids": common_passages,
        "common_document_ids": common_documents,
        "bm25_unique_document_count": len(bm25_document_ids),
        "dense_unique_document_count": len(dense_document_ids),
        "document_jaccard": document_jaccard,
    }


def print_comparison_summary(
    summary: dict[str, Any],
) -> None:
    print()
    print("=" * 120)
    print("OVERLAP SUMMARY")
    print("=" * 120)

    print(
        "Common passages:       "
        f"{summary['common_passage_count']}"
    )
    print(
        "Common documents:      "
        f"{summary['common_document_count']}"
    )
    print(
        "BM25 unique documents: "
        f"{summary['bm25_unique_document_count']}"
    )
    print(
        "Dense unique documents:"
        f" {summary['dense_unique_document_count']}"
    )
    print(
        "Document Jaccard:      "
        f"{summary['document_jaccard']:.4f}"
    )

    if summary["common_document_ids"]:
        print(
            "Common document IDs:  "
            + ", ".join(summary["common_document_ids"])
        )
    else:
        print("Common document IDs:  None")


def save_comparison(
    output_path: Path,
    query: str,
    bm25_results: list[dict[str, Any]],
    dense_results: list[dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {
        "query": query,
        "bm25_results": bm25_results,
        "dense_results": dense_results,
        "summary": summary,
    }

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(
            payload,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print()
    print(f"Comparison saved to: {output_path.resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run BM25 and dense retrieval for the same query."
        )
    )

    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="Query to run with both retrieval methods.",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of passages returned by each method.",
    )

    parser.add_argument(
        "--passages",
        type=Path,
        default=DEFAULT_PASSAGES_PATH,
        help="Path to the processed passage collection.",
    )

    parser.add_argument(
        "--dense-index-dir",
        type=Path,
        default=DEFAULT_DENSE_INDEX_DIR,
        help="Path to the saved dense FAISS index.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device used for dense query encoding.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON path for saving the comparison.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.top_k <= 0:
        raise ValueError("top_k must be greater than zero.")

    print("Loading BM25 retriever...")

    bm25_retriever = BM25Retriever.from_jsonl(
        args.passages
    )

    print()
    print("Loading dense retriever...")

    dense_retriever = DenseRetriever(
        index_dir=args.dense_index_dir,
        device=args.device,
    )

    print()
    print(f"Running query: {args.query}")

    bm25_results = bm25_retriever.search(
        query=args.query,
        top_k=args.top_k,
    )

    dense_results = dense_retriever.search(
        query=args.query,
        top_k=args.top_k,
    )

    print_method_results(
        method_name="BM25 RESULTS",
        results=bm25_results,
    )

    print_method_results(
        method_name="DENSE RETRIEVAL RESULTS",
        results=dense_results,
    )

    summary = create_comparison_summary(
        bm25_results=bm25_results,
        dense_results=dense_results,
    )

    print_comparison_summary(summary)

    if args.output is not None:
        save_comparison(
            output_path=args.output,
            query=args.query,
            bm25_results=bm25_results,
            dense_results=dense_results,
            summary=summary,
        )


if __name__ == "__main__":
    main()