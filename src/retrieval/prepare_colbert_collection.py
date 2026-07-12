from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable


DEFAULT_PASSAGES_PATH = Path("data/processed/passages.jsonl")
DEFAULT_OUTPUT_DIR = Path("data/processed/colbert")


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


def clean_text_for_tsv(text: str) -> str:
    """
    Make passage text safe for TSV-based ColBERT collection files.
    """
    return " ".join(text.replace("\t", " ").split())


def prepare_colbert_collection(
    passages_path: Path,
    output_dir: Path,
) -> None:
    if not passages_path.exists():
        raise FileNotFoundError(
            f"Passage file not found: {passages_path}\n"
            "Run: python -m src.ingestion.chunk_docs"
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    collection_path = output_dir / "collection.tsv"
    mapping_path = output_dir / "pid_mapping.jsonl"
    metadata_path = output_dir / "metadata.json"

    passage_count = 0

    with (
        collection_path.open("w", encoding="utf-8") as collection_file,
        mapping_path.open("w", encoding="utf-8") as mapping_file,
    ):
        for pid, passage in enumerate(read_jsonl(passages_path)):
            passage_text = clean_text_for_tsv(passage["text"])

            if not passage_text:
                continue

            collection_file.write(f"{pid}\t{passage_text}\n")

            mapping = {
                "pid": pid,
                "passage_id": passage["passage_id"],
                "doc_id": passage["doc_id"],
                "title": passage["title"],
                "source": passage["source"],
                "metadata": passage.get("metadata", {}),
            }

            mapping_file.write(
                json.dumps(mapping, ensure_ascii=False) + "\n"
            )

            passage_count += 1

    metadata = {
        "source_passages_path": str(passages_path),
        "collection_file": str(collection_path),
        "mapping_file": str(mapping_path),
        "passage_count": passage_count,
        "format": "pid<TAB>passage_text",
        "notes": (
            "This collection is prepared for ColBERT-style indexing. "
            "pid_mapping.jsonl maps integer pids back to original passage "
            "and document metadata."
        ),
    }

    with metadata_path.open("w", encoding="utf-8") as metadata_file:
        json.dump(
            metadata,
            metadata_file,
            ensure_ascii=False,
            indent=2,
        )

    print("ColBERT collection prepared successfully.")
    print(f"Passages:   {passage_count:,}")
    print(f"Collection: {collection_path.resolve()}")
    print(f"Mapping:    {mapping_path.resolve()}")
    print(f"Metadata:   {metadata_path.resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a ColBERT-compatible collection from passages."
    )

    parser.add_argument(
        "--passages",
        type=Path,
        default=DEFAULT_PASSAGES_PATH,
        help="Path to processed passages JSONL.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for ColBERT collection files.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    prepare_colbert_collection(
        passages_path=args.passages,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()