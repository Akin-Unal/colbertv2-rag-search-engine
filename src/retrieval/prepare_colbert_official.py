from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable


DEFAULT_PASSAGES_PATH = Path("data/processed/passages.jsonl")
DEFAULT_QUERIES_PATH = Path("data/processed/queries.jsonl")
DEFAULT_QRELS_PATH = Path("data/processed/qrels.jsonl")
DEFAULT_OUTPUT_DIR = Path("data/processed/colbert")


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


def clean_tsv_text(text: str) -> str:
    """
    Make text safe for ColBERT TSV files.

    ColBERT expects tab-separated files, so tabs and newlines
    inside text must be removed.
    """
    return " ".join(
        text.replace("\t", " ")
        .replace("\r", " ")
        .replace("\n", " ")
        .split()
    )


def write_collection(
    passages_path: Path,
    collection_path: Path,
    mapping_path: Path,
) -> int:
    count = 0

    with (
        collection_path.open("w", encoding="utf-8") as collection_file,
        mapping_path.open("w", encoding="utf-8") as mapping_file,
    ):
        for pid, passage in enumerate(read_jsonl(passages_path)):
            passage_text = clean_tsv_text(passage["text"])

            if not passage_text:
                continue

            collection_file.write(f"{pid}\t{passage_text}\n")

            mapping = {
                "pid": pid,
                "passage_id": passage["passage_id"],
                "doc_id": str(passage["doc_id"]),
                "title": passage["title"],
                "text": passage_text,
                "source": passage["source"],
                "metadata": passage.get("metadata", {}),
            }

            mapping_file.write(
                json.dumps(mapping, ensure_ascii=False) + "\n"
            )

            count += 1

    return count


def write_queries(
    queries_path: Path,
    output_path: Path,
) -> int:
    count = 0

    with output_path.open("w", encoding="utf-8") as output_file:
        for query in read_jsonl(queries_path):
            query_id = clean_tsv_text(str(query["query_id"]))
            query_text = clean_tsv_text(query["text"])

            if not query_id or not query_text:
                continue

            output_file.write(f"{query_id}\t{query_text}\n")
            count += 1

    return count


def write_qrels(
    qrels_path: Path,
    output_path: Path,
) -> int:
    """
    Writes qrels for reference.

    Note:
    SciFact qrels are document-level. ColBERT ranking will return
    passage-level pids, so our evaluation pipeline will still map
    pids/passages back to doc_id.
    """
    count = 0

    with output_path.open("w", encoding="utf-8") as output_file:
        for qrel in read_jsonl(qrels_path):
            query_id = clean_tsv_text(str(qrel["query_id"]))
            doc_id = clean_tsv_text(str(qrel["doc_id"]))
            relevance = int(qrel["relevance"])

            output_file.write(f"{query_id}\t0\t{doc_id}\t{relevance}\n")
            count += 1

    return count


def prepare_colbert_files(
    passages_path: Path,
    queries_path: Path,
    qrels_path: Path,
    output_dir: Path,
) -> None:
    if not passages_path.exists():
        raise FileNotFoundError(
            f"Passages file not found: {passages_path}"
        )

    if not queries_path.exists():
        raise FileNotFoundError(
            f"Queries file not found: {queries_path}"
        )

    if not qrels_path.exists():
        raise FileNotFoundError(
            f"Qrels file not found: {qrels_path}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    collection_path = output_dir / "collection.tsv"
    queries_output_path = output_dir / "queries.tsv"
    qrels_output_path = output_dir / "qrels.tsv"
    mapping_path = output_dir / "pid_mapping.jsonl"
    metadata_path = output_dir / "metadata.json"

    passage_count = write_collection(
        passages_path=passages_path,
        collection_path=collection_path,
        mapping_path=mapping_path,
    )

    query_count = write_queries(
        queries_path=queries_path,
        output_path=queries_output_path,
    )

    qrel_count = write_qrels(
        qrels_path=qrels_path,
        output_path=qrels_output_path,
    )

    metadata = {
        "passage_count": passage_count,
        "query_count": query_count,
        "qrel_count": qrel_count,
        "collection_file": str(collection_path),
        "queries_file": str(queries_output_path),
        "qrels_file": str(qrels_output_path),
        "mapping_file": str(mapping_path),
        "collection_format": "pid<TAB>passage_text",
        "queries_format": "qid<TAB>query_text",
        "qrels_format": "qid<TAB>0<TAB>doc_id<TAB>relevance",
    }

    with metadata_path.open("w", encoding="utf-8") as metadata_file:
        json.dump(metadata, metadata_file, ensure_ascii=False, indent=2)

    print("Official ColBERT files prepared successfully.")
    print(f"Passages:   {passage_count:,}")
    print(f"Queries:    {query_count:,}")
    print(f"Qrels:      {qrel_count:,}")
    print(f"Collection: {collection_path.resolve()}")
    print(f"Queries:    {queries_output_path.resolve()}")
    print(f"Qrels:      {qrels_output_path.resolve()}")
    print(f"Mapping:    {mapping_path.resolve()}")
    print(f"Metadata:   {metadata_path.resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare TSV files for official Stanford ColBERT."
    )

    parser.add_argument(
        "--passages",
        type=Path,
        default=DEFAULT_PASSAGES_PATH,
    )

    parser.add_argument(
        "--queries",
        type=Path,
        default=DEFAULT_QUERIES_PATH,
    )

    parser.add_argument(
        "--qrels",
        type=Path,
        default=DEFAULT_QRELS_PATH,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    prepare_colbert_files(
        passages_path=args.passages,
        queries_path=args.queries,
        qrels_path=args.qrels,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()