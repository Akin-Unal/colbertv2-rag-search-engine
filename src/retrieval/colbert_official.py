from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

from colbert import Searcher
from colbert.infra import ColBERTConfig, Run, RunConfig


DEFAULT_ROOT = Path("colbert_experiments")
DEFAULT_EXPERIMENT_NAME = "scifact_colbertv2"
DEFAULT_INDEX_NAME = "scifact.nbits=2"
DEFAULT_MAPPING_PATH = Path("data/processed/colbert/pid_mapping.jsonl")


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
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


def load_pid_mapping(
    mapping_path: Path,
) -> dict[int, dict[str, Any]]:
    if not mapping_path.exists():
        raise FileNotFoundError(
            f"PID mapping file not found: {mapping_path}\n"
            "Run: python -m src.retrieval.prepare_colbert_official"
        )

    mapping: dict[int, dict[str, Any]] = {}

    for row in read_jsonl(mapping_path):
        pid = int(row["pid"])
        mapping[pid] = row

    if not mapping:
        raise ValueError("PID mapping is empty.")

    return mapping


class OfficialColBERTRetriever:
    """
    Official Stanford ColBERT retriever wrapper.
    """

    def __init__(
        self,
        root: Path = DEFAULT_ROOT,
        experiment_name: str = DEFAULT_EXPERIMENT_NAME,
        index_name: str = DEFAULT_INDEX_NAME,
        mapping_path: Path = DEFAULT_MAPPING_PATH,
        nranks: int = 1,
    ) -> None:
        self.root = root
        self.experiment_name = experiment_name
        self.index_name = index_name
        self.mapping_path = mapping_path
        self.nranks = nranks

        expected_index_path = (
            root / experiment_name / "indexes" / index_name
        )

        if not expected_index_path.exists():
            raise FileNotFoundError(
                f"ColBERT index not found: {expected_index_path}\n"
                "Run: python -m src.retrieval.build_colbert_official_index"
            )

        print(f"Loading PID mapping: {mapping_path}")
        self.pid_mapping = load_pid_mapping(mapping_path)

        print("Loading official ColBERT searcher...")
        print(f"Root:       {root.resolve()}")
        print(f"Experiment: {experiment_name}")
        print(f"Index:      {index_name}")

        with Run().context(
            RunConfig(
                nranks=nranks,
                experiment=experiment_name,
                root=str(root),
            )
        ):
            config = ColBERTConfig(
                root=str(root),
            )

            self.searcher = Searcher(
                index=index_name,
                config=config,
            )

        print("Official ColBERT retriever ready.")

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        query = query.strip()

        if not query:
            raise ValueError("Query cannot be empty.")

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        pids, ranks, scores = self.searcher.search(
            query,
            k=top_k,
        )

        results: list[dict[str, Any]] = []

        for fallback_rank, (pid, rank, score) in enumerate(
            zip(pids, ranks, scores),
            start=1,
        ):
            pid = int(pid)
            mapping = self.pid_mapping.get(pid, {})

            results.append(
                {
                    "rank": int(rank) if rank is not None else fallback_rank,
                    "pid": pid,
                    "passage_id": mapping.get("passage_id", str(pid)),
                    "doc_id": mapping.get("doc_id", ""),
                    "title": mapping.get("title", ""),
                    "text": mapping.get("text", ""),
                    "source": mapping.get("source", ""),
                    "score": float(score),
                    "metadata": mapping.get("metadata", {}),
                }
            )

        return results


def print_results(
    query: str,
    results: list[dict[str, Any]],
) -> None:
    print()
    print("=" * 90)
    print(f"QUERY: {query}")
    print("METHOD: Official Stanford ColBERTv2")
    print("=" * 90)

    for result in results:
        print()
        print(f"Rank:       {result['rank']}")
        print(f"Score:      {result['score']:.4f}")
        print(f"PID:        {result['pid']}")
        print(f"Passage ID: {result['passage_id']}")
        print(f"Document:   {result['doc_id']}")
        print(f"Title:      {result['title']}")
        print(f"Source:     {result['source']}")
        print("-" * 90)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Search with official Stanford ColBERT."
    )

    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="Search query.",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of results.",
    )

    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
    )

    parser.add_argument(
        "--experiment",
        type=str,
        default=DEFAULT_EXPERIMENT_NAME,
    )

    parser.add_argument(
        "--index-name",
        type=str,
        default=DEFAULT_INDEX_NAME,
    )

    parser.add_argument(
        "--mapping",
        type=Path,
        default=DEFAULT_MAPPING_PATH,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    retriever = OfficialColBERTRetriever(
        root=args.root,
        experiment_name=args.experiment,
        index_name=args.index_name,
        mapping_path=args.mapping,
    )

    results = retriever.search(
        query=args.query,
        top_k=args.top_k,
    )

    print_results(
        query=args.query,
        results=results,
    )


if __name__ == "__main__":
    main()