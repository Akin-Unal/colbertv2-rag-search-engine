from __future__ import annotations

import argparse
from pathlib import Path

from colbert import Indexer
from colbert.infra import ColBERTConfig, Run, RunConfig


DEFAULT_COLLECTION_PATH = Path("data/processed/colbert/collection.tsv")
DEFAULT_EXPERIMENT_NAME = "scifact_colbertv2"
DEFAULT_INDEX_NAME = "scifact.nbits=2"
DEFAULT_ROOT = Path("colbert_experiments")
DEFAULT_CHECKPOINT = "colbert-ir/colbertv2.0"


def build_index(
    collection_path: Path,
    checkpoint: str,
    root: Path,
    experiment_name: str,
    index_name: str,
    nbits: int,
    doc_maxlen: int,
    nranks: int,
    kmeans_niters: int,
) -> None:
    if not collection_path.exists():
        raise FileNotFoundError(
            f"Collection file not found: {collection_path}\n"
            "Run: python -m src.retrieval.prepare_colbert_official"
        )

    root.mkdir(parents=True, exist_ok=True)

    print("Building official Stanford ColBERT index")
    print(f"Collection:     {collection_path.resolve()}")
    print(f"Checkpoint:     {checkpoint}")
    print(f"Root:           {root.resolve()}")
    print(f"Experiment:     {experiment_name}")
    print(f"Index name:     {index_name}")
    print(f"nbits:          {nbits}")
    print(f"doc_maxlen:     {doc_maxlen}")
    print(f"nranks:         {nranks}")
    print(f"kmeans_niters:  {kmeans_niters}")

    with Run().context(
        RunConfig(
            nranks=nranks,
            experiment=experiment_name,
            root=str(root),
        )
    ):
        config = ColBERTConfig(
            nbits=nbits,
            doc_maxlen=doc_maxlen,
            kmeans_niters=kmeans_niters,
            root=str(root),
        )

        indexer = Indexer(
            checkpoint=checkpoint,
            config=config,
        )

        indexer.index(
            name=index_name,
            collection=str(collection_path),
            overwrite=True,
        )

    print()
    print("ColBERT index build finished.")
    print("Expected index location:")
    print(root / experiment_name / "indexes" / index_name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an official Stanford ColBERTv2 index."
    )

    parser.add_argument(
        "--collection",
        type=Path,
        default=DEFAULT_COLLECTION_PATH,
        help="Path to ColBERT collection.tsv.",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=DEFAULT_CHECKPOINT,
        help="ColBERT checkpoint name or local path.",
    )

    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help="Root directory for ColBERT experiments.",
    )

    parser.add_argument(
        "--experiment",
        type=str,
        default=DEFAULT_EXPERIMENT_NAME,
        help="ColBERT experiment name.",
    )

    parser.add_argument(
        "--index-name",
        type=str,
        default=DEFAULT_INDEX_NAME,
        help="Name of the ColBERT index.",
    )

    parser.add_argument(
        "--nbits",
        type=int,
        default=2,
        help="Number of bits used for ColBERT residual compression.",
    )

    parser.add_argument(
        "--doc-maxlen",
        type=int,
        default=180,
        help="Maximum number of document tokens encoded by ColBERT.",
    )

    parser.add_argument(
        "--nranks",
        type=int,
        default=1,
        help="Number of GPU ranks. Use 1 for single GPU.",
    )

    parser.add_argument(
        "--kmeans-niters",
        type=int,
        default=4,
        help=(
            "Number of FAISS k-means iterations. "
            "Lower values are faster for small experimental corpora."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    build_index(
        collection_path=args.collection,
        checkpoint=args.checkpoint,
        root=args.root,
        experiment_name=args.experiment,
        index_name=args.index_name,
        nbits=args.nbits,
        doc_maxlen=args.doc_maxlen,
        nranks=args.nranks,
        kmeans_niters=args.kmeans_niters,
    )


if __name__ == "__main__":
    main()