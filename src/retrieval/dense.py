from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer


DEFAULT_INDEX_DIR = Path("indexes/dense")


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    """
    Read JSON objects from a JSONL file.
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


def resolve_device(requested_device: str) -> str:
    """
    Resolve the PyTorch device used for query encoding.
    """
    if requested_device != "auto":
        return requested_device

    if torch.cuda.is_available():
        return "cuda"

    return "cpu"


class DenseRetriever:
    """
    Single-vector dense passage retriever backed by FAISS.
    """

    def __init__(
        self,
        index_dir: Path,
        device: str = "auto",
    ) -> None:
        self.index_dir = index_dir

        metadata_path = index_dir / "metadata.json"

        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Dense metadata not found: {metadata_path}\n"
                "Run: python -m src.retrieval.build_dense_index"
            )

        with metadata_path.open("r", encoding="utf-8") as file:
            self.metadata = json.load(file)

        index_path = index_dir / self.metadata["index_file"]
        passages_path = (
            index_dir / self.metadata["passage_mapping_file"]
        )

        if not index_path.exists():
            raise FileNotFoundError(
                f"FAISS index not found: {index_path}"
            )

        if not passages_path.exists():
            raise FileNotFoundError(
                f"Passage mapping not found: {passages_path}"
            )

        print(f"Loading FAISS index: {index_path}")
        self.index = faiss.read_index(str(index_path))

        print(f"Loading passage mapping: {passages_path}")
        self.passages = list(read_jsonl(passages_path))

        expected_count = int(self.metadata["passage_count"])
        expected_dimension = int(
            self.metadata["embedding_dimension"]
        )

        if len(self.passages) != expected_count:
            raise ValueError(
                "Passage mapping count does not match metadata."
            )

        if self.index.ntotal != expected_count:
            raise ValueError(
                "FAISS index count does not match metadata."
            )

        if self.index.d != expected_dimension:
            raise ValueError(
                "FAISS vector dimension does not match metadata."
            )

        self.device = resolve_device(device)
        self.model_name = self.metadata["model_name"]

        print(f"Loading embedding model: {self.model_name}")
        print(f"Query encoding device:   {self.device}")

        self.model = SentenceTransformer(
            self.model_name,
            device=self.device,
        )

        print(
            f"Dense retriever ready with "
            f"{self.index.ntotal:,} passages."
        )

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """
        Encode a query and retrieve the top-k nearest passages.
        """
        query = query.strip()

        if not query:
            raise ValueError("Query cannot be empty.")

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        top_k = min(top_k, len(self.passages))

        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        query_embedding = np.asarray(
            query_embedding,
            dtype=np.float32,
        )

        if query_embedding.shape[1] != self.index.d:
            raise ValueError(
                "Query embedding dimension does not match "
                "the FAISS index dimension."
            )

        scores, indices = self.index.search(
            query_embedding,
            top_k,
        )

        results: list[dict[str, Any]] = []

        for rank, (score, passage_index) in enumerate(
            zip(scores[0], indices[0]),
            start=1,
        ):
            if passage_index < 0:
                continue

            passage = self.passages[int(passage_index)]

            results.append(
                {
                    "rank": rank,
                    "passage_id": passage["passage_id"],
                    "doc_id": passage["doc_id"],
                    "title": passage["title"],
                    "text": passage["text"],
                    "source": passage["source"],
                    "score": float(score),
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
    print("METHOD: Dense Retrieval")
    print("=" * 90)

    for result in results:
        preview = result["text"].replace("\n", " ").strip()

        if len(preview) > text_preview_length:
            preview = (
                preview[:text_preview_length].rstrip()
                + "..."
            )

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
        description=(
            "Search the SciFact passage collection "
            "using dense retrieval and FAISS."
        )
    )

    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="Natural-language search query.",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of passages to retrieve.",
    )

    parser.add_argument(
        "--index-dir",
        type=Path,
        default=DEFAULT_INDEX_DIR,
        help="Directory containing the FAISS index.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device used for query encoding.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    retriever = DenseRetriever(
        index_dir=args.index_dir,
        device=args.device,
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