from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Dict, Any

import ir_datasets
from tqdm import tqdm


DATASET_ID = "beir/scifact/test"
OUT_DIR = Path("data/processed")


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1

    return count


def load_documents(dataset) -> list[dict[str, Any]]:
    documents = []

    for doc in tqdm(dataset.docs_iter(), desc="Loading documents"):
        documents.append(
            {
                "doc_id": doc.doc_id,
                "title": doc.title or "",
                "text": doc.text or "",
                "source": DATASET_ID,
                "metadata": {
                    "dataset": "SciFact",
                    "task": "scientific claim verification",
                    "original_doc_id": doc.doc_id,
                    "text_length": len(doc.text or ""),
                },
            }
        )

    return documents


def load_queries(dataset) -> list[dict[str, Any]]:
    queries = []

    for query in tqdm(dataset.queries_iter(), desc="Loading queries"):
        queries.append(
            {
                "query_id": query.query_id,
                "text": query.text,
                "source": DATASET_ID,
                "metadata": {
                    "dataset": "SciFact",
                    "split": "test",
                },
            }
        )

    return queries


def load_qrels(dataset) -> list[dict[str, Any]]:
    qrels = []

    for qrel in tqdm(dataset.qrels_iter(), desc="Loading qrels"):
        qrels.append(
            {
                "query_id": qrel.query_id,
                "doc_id": qrel.doc_id,
                "relevance": int(qrel.relevance),
                "metadata": {
                    "iteration": qrel.iteration,
                },
            }
        )

    return qrels


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset: {DATASET_ID}")
    dataset = ir_datasets.load(DATASET_ID)

    documents = load_documents(dataset)
    queries = load_queries(dataset)
    qrels = load_qrels(dataset)

    doc_count = write_jsonl(OUT_DIR / "documents.jsonl", documents)
    query_count = write_jsonl(OUT_DIR / "queries.jsonl", queries)
    qrel_count = write_jsonl(OUT_DIR / "qrels.jsonl", qrels)

    print("\nDone.")
    print(f"Documents: {doc_count}")
    print(f"Queries:   {query_count}")
    print(f"Qrels:     {qrel_count}")
    print(f"Output dir: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()