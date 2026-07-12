from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Iterable

import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer


DEFAULT_PASSAGES_PATH = Path("data/processed/passages.jsonl")
DEFAULT_OUTPUT_DIR = Path("indexes/dense")
DEFAULT_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"


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


def write_jsonl(
    path: Path,
    rows: Iterable[dict[str, Any]],
) -> int:
    """
    Write dictionaries to a JSONL file.
    """
    count = 0

    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1

    return count


def resolve_device(requested_device: str) -> str:
    """
    Resolve the PyTorch device used for embedding generation.
    """
    if requested_device != "auto":
        return requested_device

    if torch.cuda.is_available():
        return "cuda"

    return "cpu"


def validate_passages(
    passages: list[dict[str, Any]],
) -> None:
    """
    Validate that the passage collection contains required fields.
    """
    if not passages:
        raise ValueError("Passage collection cannot be empty.")

    required_fields = {
        "passage_id",
        "doc_id",
        "title",
        "text",
        "source",
    }

    for index, passage in enumerate(passages):
        missing_fields = required_fields - passage.keys()

        if missing_fields:
            raise ValueError(
                f"Passage at index {index} is missing fields: "
                f"{sorted(missing_fields)}"
            )

        if not passage["text"].strip():
            raise ValueError(
                f"Passage {passage['passage_id']} has empty text."
            )


def build_index(
    passages_path: Path,
    output_dir: Path,
    model_name: str,
    batch_size: int,
    requested_device: str,
) -> None:
    """
    Encode passages and build a FAISS inner-product index.
    """
    if not passages_path.exists():
        raise FileNotFoundError(
            f"Passage file not found: {passages_path}\n"
            "Run: python -m src.ingestion.chunk_docs"
        )

    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero.")

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading passages from: {passages_path}")
    passages = list(read_jsonl(passages_path))

    validate_passages(passages)

    print(f"Loaded {len(passages):,} passages.")

    device = resolve_device(requested_device)

    print(f"Embedding model: {model_name}")
    print(f"Device:          {device}")
    print(f"Batch size:      {batch_size}")

    model = SentenceTransformer(
        model_name,
        device=device,
    )

    passage_texts = [
        passage["text"]
        for passage in passages
    ]

    started_at = time.perf_counter()

    print("Generating normalized passage embeddings...")

    embeddings = model.encode(
        passage_texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    embeddings = np.asarray(
        embeddings,
        dtype=np.float32,
    )

    if embeddings.ndim != 2:
        raise ValueError(
            f"Expected a two-dimensional embedding matrix, "
            f"received shape {embeddings.shape}."
        )

    passage_count, embedding_dimension = embeddings.shape

    if passage_count != len(passages):
        raise ValueError(
            "Embedding count does not match passage count."
        )

    print(
        f"Embedding matrix: "
        f"{passage_count:,} x {embedding_dimension}"
    )

    print("Building FAISS IndexFlatIP...")

    index = faiss.IndexFlatIP(embedding_dimension)
    index.add(embeddings)

    if index.ntotal != len(passages):
        raise ValueError(
            "FAISS index size does not match passage count."
        )

    index_path = output_dir / "index.faiss"
    passages_output_path = output_dir / "passages.jsonl"
    metadata_path = output_dir / "metadata.json"

    faiss.write_index(
        index,
        str(index_path),
    )

    saved_passage_count = write_jsonl(
        passages_output_path,
        passages,
    )

    elapsed_seconds = time.perf_counter() - started_at

    metadata = {
        "model_name": model_name,
        "embedding_dimension": embedding_dimension,
        "passage_count": passage_count,
        "normalized_embeddings": True,
        "similarity": "cosine",
        "faiss_index_type": "IndexFlatIP",
        "batch_size": batch_size,
        "build_device": device,
        "source_passages_path": str(passages_path),
        "index_file": index_path.name,
        "passage_mapping_file": passages_output_path.name,
        "build_seconds": round(elapsed_seconds, 4),
    }

    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(
            metadata,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print()
    print("Dense index created successfully.")
    print(f"Indexed passages: {index.ntotal:,}")
    print(f"Saved passages:   {saved_passage_count:,}")
    print(f"Vector dimension: {embedding_dimension}")
    print(f"Build time:       {elapsed_seconds:.2f} seconds")
    print(f"Index file:       {index_path.resolve()}")
    print(f"Metadata file:    {metadata_path.resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate dense passage embeddings and "
            "build a FAISS index."
        )
    )

    parser.add_argument(
        "--passages",
        type=Path,
        default=DEFAULT_PASSAGES_PATH,
        help="Path to the processed passage JSONL file.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where the dense index will be saved.",
    )

    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help="Sentence Transformers model name.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Number of passages encoded in each batch.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device used for embedding generation.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    build_index(
        passages_path=args.passages,
        output_dir=args.output_dir,
        model_name=args.model,
        batch_size=args.batch_size,
        requested_device=args.device,
    )


if __name__ == "__main__":
    main()