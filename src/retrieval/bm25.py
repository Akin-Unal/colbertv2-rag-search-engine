from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from rank_bm25 import BM25Okapi


DEFAULT_PASSAGES_PATH = Path("data/processed/passages.jsonl")


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    """
    Read a JSONL file line by line.

    Each line in a JSONL file is expected to contain
    one valid JSON object.
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


def tokenize(text: str) -> list[str]:
    """
    Convert text into lowercase word tokens.

    SciFact is an English dataset, so a simple regex tokenizer
    is sufficient for the first BM25 baseline.
    """
    return re.findall(r"[a-z0-9]+", text.lower())


class BM25Retriever:
    """
    Keyword-based passage retriever using BM25Okapi.
    """

    def __init__(self, passages: list[dict[str, Any]]) -> None:
        if not passages:
            raise ValueError("Passage collection cannot be empty.")

        self.passages = passages

        self.tokenized_corpus = [
            tokenize(passage["text"])
            for passage in passages
        ]

        self.index = BM25Okapi(self.tokenized_corpus)

    @classmethod
    def from_jsonl(cls, path: Path) -> "BM25Retriever":
        if not path.exists():
            raise FileNotFoundError(
                f"Passage file not found: {path}\n"
                "Run: python -m src.ingestion.chunk_docs"
            )

        passages = list(read_jsonl(path))

        print(f"Loaded {len(passages):,} passages.")
        print("Building BM25 index...")

        retriever = cls(passages)

        print("BM25 index ready.")

        return retriever

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """
        Search the passage collection and return the top-k results.
        """
        query = query.strip()

        if not query:
            raise ValueError("Query cannot be empty.")

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        query_tokens = tokenize(query)

        if not query_tokens:
            raise ValueError("Query does not contain searchable tokens.")

        scores = self.index.get_scores(query_tokens)

        top_k = min(top_k, len(self.passages))

        top_indices = np.argsort(scores)[::-1][:top_k]

        results: list[dict[str, Any]] = []

        for rank, passage_index in enumerate(top_indices, start=1):
            passage = self.passages[int(passage_index)]

            results.append(
                {
                    "rank": rank,
                    "passage_id": passage["passage_id"],
                    "doc_id": passage["doc_id"],
                    "title": passage["title"],
                    "text": passage["text"],
                    "source": passage["source"],
                    "score": float(scores[passage_index]),
                    "metadata": passage.get("metadata", {}),
                }
            )

        return results


def print_results(
    query: str,
    results: list[dict[str, Any]],
    text_preview_length: int = 500,
) -> None:
    print()
    print("=" * 90)
    print(f"QUERY: {query}")
    print("=" * 90)

    for result in results:
        preview = result["text"].replace("\n", " ").strip()

        if len(preview) > text_preview_length:
            preview = preview[:text_preview_length].rstrip() + "..."

        print()
        print(f"Rank:       {result['rank']}")
        print(f"Score:      {result['score']:.4f}")
        print(f"Passage ID: {result['passage_id']}")
        print(f"Document:   {result['doc_id']}")
        print(f"Title:      {result['title']}")
        print(f"Source:     {result['source']}")
        print(f"Passage:    {preview}")
        print("-" * 90)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Search SciFact passages using BM25."
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
        help="Number of results to return.",
    )

    parser.add_argument(
        "--passages",
        type=Path,
        default=DEFAULT_PASSAGES_PATH,
        help="Path to the passage JSONL file.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    retriever = BM25Retriever.from_jsonl(args.passages)

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