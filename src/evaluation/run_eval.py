from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

from tqdm import tqdm

from src.evaluation.metrics import (
    average_metrics,
    evaluate_single_query,
    passage_results_to_document_ranking,
)
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.dense import DenseRetriever


DEFAULT_QUERIES_PATH = Path("data/processed/queries.jsonl")
DEFAULT_QRELS_PATH = Path("data/processed/qrels.jsonl")
DEFAULT_PASSAGES_PATH = Path("data/processed/passages.jsonl")
DEFAULT_DENSE_INDEX_DIR = Path("indexes/dense")
DEFAULT_OUTPUT_DIR = Path("experiments/evaluation")


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    """
    Read a JSONL file.
    """
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON at line {line_number} in {path}"
                ) from exc


def load_queries(path: Path) -> list[dict[str, Any]]:
    """
    Load query records.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Queries file not found: {path}\n"
            "Run: python -m src.ingestion.load_scifact"
        )

    queries = list(read_jsonl(path))

    if not queries:
        raise ValueError("No queries found.")

    return queries


def load_qrels(
    path: Path,
) -> dict[str, dict[str, int]]:
    """
    Load qrels as:

        {
            query_id: {
                doc_id: relevance
            }
        }
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Qrels file not found: {path}\n"
            "Run: python -m src.ingestion.load_scifact"
        )

    qrels_by_query: dict[str, dict[str, int]] = {}

    for row in read_jsonl(path):
        query_id = str(row["query_id"])
        doc_id = str(row["doc_id"])
        relevance = int(row["relevance"])

        qrels_by_query.setdefault(query_id, {})
        qrels_by_query[query_id][doc_id] = relevance

    if not qrels_by_query:
        raise ValueError("No qrels found.")

    return qrels_by_query


def filter_evaluable_queries(
    queries: list[dict[str, Any]],
    qrels_by_query: dict[str, dict[str, int]],
    max_queries: int | None,
) -> list[dict[str, Any]]:
    """
    Keep only queries that have at least one qrel.
    """
    evaluable_queries = [
        query
        for query in queries
        if str(query["query_id"]) in qrels_by_query
    ]

    if max_queries is not None:
        evaluable_queries = evaluable_queries[:max_queries]

    if not evaluable_queries:
        raise ValueError("No evaluable queries found.")

    return evaluable_queries


def evaluate_retriever(
    method_name: str,
    retriever: Any,
    queries: list[dict[str, Any]],
    qrels_by_query: dict[str, dict[str, int]],
    candidate_k: int,
    eval_k: int,
) -> dict[str, Any]:
    """
    Evaluate one retriever over all evaluable queries.
    """
    per_query_results: list[dict[str, Any]] = []
    per_query_metrics: list[dict[str, float]] = []

    for query in tqdm(queries, desc=f"Evaluating {method_name}"):
        query_id = str(query["query_id"])
        query_text = query["text"]

        passage_results = retriever.search(
            query=query_text,
            top_k=candidate_k,
        )

        ranked_doc_ids = passage_results_to_document_ranking(
            passage_results=passage_results,
            max_docs=eval_k,
        )

        relevance_by_doc_id = qrels_by_query[query_id]

        metrics = evaluate_single_query(
            ranked_doc_ids=ranked_doc_ids,
            relevance_by_doc_id=relevance_by_doc_id,
        )

        per_query_metrics.append(metrics)

        per_query_results.append(
            {
                "query_id": query_id,
                "query": query_text,
                "ranked_doc_ids": ranked_doc_ids,
                "relevant_doc_ids": sorted(
                    [
                        doc_id
                        for doc_id, relevance in relevance_by_doc_id.items()
                        if relevance > 0
                    ]
                ),
                "metrics": metrics,
            }
        )

    averaged = average_metrics(per_query_metrics)

    return {
        "method": method_name,
        "query_count": len(queries),
        "candidate_k": candidate_k,
        "eval_k": eval_k,
        "metrics": averaged,
        "per_query": per_query_results,
    }


def save_json_report(
    report: dict[str, Any],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(
            report,
            file,
            ensure_ascii=False,
            indent=2,
        )


def format_metric(value: float) -> str:
    return f"{value:.4f}"


def create_markdown_table(
    reports: list[dict[str, Any]],
) -> str:
    """
    Create a README-friendly Markdown result table.
    """
    lines = [
        "| Method | Recall@5 | Recall@10 | MRR@10 | nDCG@10 | Notes |",
        "|---|---:|---:|---:|---:|---|",
    ]

    notes = {
        "bm25": "Lexical keyword baseline",
        "dense": "Sentence Transformers + FAISS",
    }

    for report in reports:
        method = report["method"]
        metrics = report["metrics"]

        line = (
            f"| {method.upper()} "
            f"| {format_metric(metrics['recall_at_5'])} "
            f"| {format_metric(metrics['recall_at_10'])} "
            f"| {format_metric(metrics['mrr_at_10'])} "
            f"| {format_metric(metrics['ndcg_at_10'])} "
            f"| {notes.get(method, '')} |"
        )

        lines.append(line)

    return "\n".join(lines)


def save_markdown_report(
    reports: list[dict[str, Any]],
    output_path: Path,
) -> None:
    table = create_markdown_table(reports)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    content = (
        "# Retrieval Evaluation Results\n\n"
        "The following metrics were generated by the local evaluation pipeline.\n\n"
        f"{table}\n"
    )

    with output_path.open("w", encoding="utf-8") as file:
        file.write(content)


def print_summary(reports: list[dict[str, Any]]) -> None:
    print()
    print("=" * 90)
    print("EVALUATION SUMMARY")
    print("=" * 90)
    print(create_markdown_table(reports))
    print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate retrieval methods on SciFact qrels."
    )

    parser.add_argument(
        "--method",
        choices=["bm25", "dense", "all"],
        default="all",
        help="Retrieval method to evaluate.",
    )

    parser.add_argument(
        "--queries",
        type=Path,
        default=DEFAULT_QUERIES_PATH,
        help="Path to queries JSONL.",
    )

    parser.add_argument(
        "--qrels",
        type=Path,
        default=DEFAULT_QRELS_PATH,
        help="Path to qrels JSONL.",
    )

    parser.add_argument(
        "--passages",
        type=Path,
        default=DEFAULT_PASSAGES_PATH,
        help="Path to passages JSONL.",
    )

    parser.add_argument(
        "--dense-index-dir",
        type=Path,
        default=DEFAULT_DENSE_INDEX_DIR,
        help="Path to dense FAISS index directory.",
    )

    parser.add_argument(
        "--candidate-k",
        type=int,
        default=50,
        help=(
            "Number of passage candidates retrieved before "
            "document-level deduplication."
        ),
    )

    parser.add_argument(
        "--eval-k",
        type=int,
        default=10,
        help="Maximum document rank used for evaluation.",
    )

    parser.add_argument(
        "--max-queries",
        type=int,
        default=None,
        help="Optional limit for quick debugging.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where evaluation reports will be saved.",
    )

    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Device for dense query encoding.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.candidate_k <= 0:
        raise ValueError("candidate_k must be greater than zero.")

    if args.eval_k <= 0:
        raise ValueError("eval_k must be greater than zero.")

    if args.candidate_k < args.eval_k:
        raise ValueError("candidate_k should be greater than or equal to eval_k.")

    print("Loading queries and qrels...")

    queries = load_queries(args.queries)
    qrels_by_query = load_qrels(args.qrels)

    evaluable_queries = filter_evaluable_queries(
        queries=queries,
        qrels_by_query=qrels_by_query,
        max_queries=args.max_queries,
    )

    print(f"Loaded queries:       {len(queries):,}")
    print(f"Queries with qrels:   {len(evaluable_queries):,}")
    print(f"Candidate passages k: {args.candidate_k}")
    print(f"Evaluation docs k:    {args.eval_k}")

    methods_to_run: list[str]

    if args.method == "all":
        methods_to_run = ["bm25", "dense"]
    else:
        methods_to_run = [args.method]

    reports: list[dict[str, Any]] = []

    for method in methods_to_run:
        if method == "bm25":
            print()
            print("Loading BM25 retriever...")

            retriever = BM25Retriever.from_jsonl(args.passages)

        elif method == "dense":
            print()
            print("Loading dense retriever...")

            retriever = DenseRetriever(
                index_dir=args.dense_index_dir,
                device=args.device,
            )

        else:
            raise ValueError(f"Unsupported method: {method}")

        report = evaluate_retriever(
            method_name=method,
            retriever=retriever,
            queries=evaluable_queries,
            qrels_by_query=qrels_by_query,
            candidate_k=args.candidate_k,
            eval_k=args.eval_k,
        )

        reports.append(report)

        json_output_path = args.output_dir / f"{method}_eval.json"

        save_json_report(
            report=report,
            output_path=json_output_path,
        )

        print(f"Saved JSON report: {json_output_path.resolve()}")

    markdown_output_path = args.output_dir / "summary.md"

    save_markdown_report(
        reports=reports,
        output_path=markdown_output_path,
    )

    print(f"Saved Markdown summary: {markdown_output_path.resolve()}")

    print_summary(reports)


if __name__ == "__main__":
    main()