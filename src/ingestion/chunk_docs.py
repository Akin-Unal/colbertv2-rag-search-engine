from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, Iterable


INPUT_PATH = Path("data/processed/documents.jsonl")
OUTPUT_PATH = Path("data/processed/passages.jsonl")

CHUNK_SIZE_WORDS = 180
CHUNK_OVERLAP_WORDS = 40

def combine_title_and_text(title: str, text: str) -> str:
    """
    Combine a document title and body without producing
    duplicate punctuation such as '..'.
    """
    title = title.strip()
    text = text.strip()

    if not title:
        return text

    if not text:
        return title

    if title.endswith((".", "!", "?", ":", ";")):
        return f"{title} {text}"

    return f"{title}. {text}"

def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def chunk_words(words: list[str], chunk_size: int, overlap: int):
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be greater than overlap")

    start = 0

    while start < len(words):
        end = min(start + chunk_size, len(words))
        yield start, end, " ".join(words[start:end])

        if end == len(words):
            break

        start = end - overlap


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"{INPUT_PATH} not found. Run: python -m src.ingestion.load_scifact"
        )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    passage_count = 0
    doc_count = 0

    with OUTPUT_PATH.open("w", encoding="utf-8") as out:
        for doc in read_jsonl(INPUT_PATH):
            doc_count += 1

            title = doc["title"].strip()
            text = doc["text"].strip()

            combined_text = combine_title_and_text(title, text)
            words = combined_text.split()

            if not words:
                continue

            for chunk_index, (start, end, passage_text) in enumerate(
                chunk_words(
                    words,
                    chunk_size=CHUNK_SIZE_WORDS,
                    overlap=CHUNK_OVERLAP_WORDS,
                )
            ):
                passage = {
                    "passage_id": f"{doc['doc_id']}::p{chunk_index}",
                    "doc_id": doc["doc_id"],
                    "title": doc["title"],
                    "text": passage_text,
                    "source": doc["source"],
                    "metadata": {
                        **doc["metadata"],
                        "chunk_index": chunk_index,
                        "start_word": start,
                        "end_word": end,
                        "chunk_size_words": CHUNK_SIZE_WORDS,
                        "chunk_overlap_words": CHUNK_OVERLAP_WORDS,
                    },
                }

                out.write(json.dumps(passage, ensure_ascii=False) + "\n")
                passage_count += 1

    print("Done.")
    print(f"Documents processed: {doc_count}")
    print(f"Passages created:    {passage_count}")
    print(f"Output file:         {OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    main()