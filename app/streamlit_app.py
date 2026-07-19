from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_PASSAGES_PATH = PROJECT_ROOT / "data" / "processed" / "passages.jsonl"
DEFAULT_DENSE_INDEX_DIR = PROJECT_ROOT / "indexes" / "dense"

DEFAULT_COLBERT_ROOT = PROJECT_ROOT / "colbert_experiments"
DEFAULT_COLBERT_EXPERIMENT_NAME = "scifact_colbertv2"
DEFAULT_COLBERT_INDEX_NAME = "scifact.dmax128.nbits2"


st.set_page_config(
    page_title="ColBERTv2 RAG Search Engine",
    page_icon="🔎",
    layout="wide",
)


def path_exists(path: Path) -> bool:
    return path.exists()


@st.cache_resource(show_spinner="Loading BM25 retriever...")
def load_bm25(passages_path: str) -> Any:
    from src.retrieval.bm25 import BM25Retriever

    return BM25Retriever.from_jsonl(Path(passages_path))


@st.cache_resource(show_spinner="Loading dense retriever...")
def load_dense(index_dir: str, device: str) -> Any:
    from src.retrieval.dense import DenseRetriever

    return DenseRetriever(
        index_dir=Path(index_dir),
        device=device,
    )


@st.cache_resource(show_spinner="Loading official Stanford ColBERTv2 retriever...")
def load_colbert(
    root: str,
    experiment_name: str,
    index_name: str,
) -> Any:
    from src.retrieval.colbert_official import OfficialColBERTRetriever

    return OfficialColBERTRetriever(
        root=Path(root),
        experiment_name=experiment_name,
        index_name=index_name,
    )


def truncate_text(text: str, max_chars: int = 900) -> str:
    text = text.strip()

    if len(text) <= max_chars:
        return text

    return text[:max_chars].rstrip() + "..."


def render_result(result: dict[str, Any], fallback_rank: int) -> None:
    rank = result.get("rank", fallback_rank)
    score = result.get("score", 0.0)
    passage_id = result.get("passage_id", "")
    doc_id = result.get("doc_id", "")
    title = result.get("title", "")
    source = result.get("source", "")
    text = result.get("text", "")

    with st.container(border=True):
        st.markdown(f"### Rank {rank}")

        col_a, col_b, col_c = st.columns(3)

        with col_a:
            st.metric("Score", f"{float(score):.4f}")

        with col_b:
            st.write("**Document**")
            st.code(str(doc_id), language=None)

        with col_c:
            st.write("**Passage ID**")
            st.code(str(passage_id), language=None)

        if title:
            st.write("**Title**")
            st.write(title)

        if source:
            st.write("**Source**")
            st.write(source)

        if text:
            with st.expander("Passage text", expanded=rank == 1):
                st.write(truncate_text(text))


def run_search(
    method: str,
    query: str,
    top_k: int,
    device: str,
    passages_path: Path,
    dense_index_dir: Path,
    colbert_root: Path,
    colbert_experiment_name: str,
    colbert_index_name: str,
) -> list[dict[str, Any]]:
    if method == "BM25":
        retriever = load_bm25(str(passages_path))
        return retriever.search(query=query, top_k=top_k)

    if method == "Dense":
        retriever = load_dense(str(dense_index_dir), device)
        return retriever.search(query=query, top_k=top_k)

    if method == "ColBERTv2":
        retriever = load_colbert(
            root=str(colbert_root),
            experiment_name=colbert_experiment_name,
            index_name=colbert_index_name,
        )
        return retriever.search(query=query, top_k=top_k)

    raise ValueError(f"Unsupported method: {method}")


def main() -> None:
    st.title("🔎 ColBERTv2 RAG Search Engine")
    st.caption(
        "Compare BM25, dense FAISS retrieval, and Official Stanford ColBERTv2 "
        "on the SciFact scientific retrieval dataset."
    )

    with st.sidebar:
        st.header("Search Settings")

        query = st.text_area(
            "Query",
            value="Vitamin D deficiency increases the risk of multiple sclerosis",
            height=110,
        )

        selected_methods = st.multiselect(
            "Retrieval methods",
            options=["BM25", "Dense", "ColBERTv2"],
            default=["BM25", "Dense", "ColBERTv2"],
        )

        top_k = st.slider(
            "Top-k results",
            min_value=1,
            max_value=20,
            value=5,
        )

        device = st.selectbox(
            "Dense retrieval device",
            options=["auto", "cuda", "cpu"],
            index=0,
        )

        st.divider()

        st.subheader("Paths")

        passages_path = Path(
            st.text_input(
                "Passages JSONL",
                value=str(DEFAULT_PASSAGES_PATH),
            )
        )

        dense_index_dir = Path(
            st.text_input(
                "Dense index directory",
                value=str(DEFAULT_DENSE_INDEX_DIR),
            )
        )

        colbert_root = Path(
            st.text_input(
                "ColBERT root",
                value=str(DEFAULT_COLBERT_ROOT),
            )
        )

        colbert_experiment_name = st.text_input(
            "ColBERT experiment",
            value=DEFAULT_COLBERT_EXPERIMENT_NAME,
        )

        colbert_index_name = st.text_input(
            "ColBERT index",
            value=DEFAULT_COLBERT_INDEX_NAME,
        )

        search_clicked = st.button("Run Search", type="primary")

    st.subheader("Project Status")

    status_cols = st.columns(3)

    with status_cols[0]:
        st.write("**Passages**")
        st.success("Found" if path_exists(passages_path) else "Missing")

    with status_cols[1]:
        st.write("**Dense Index**")
        st.success("Found" if path_exists(dense_index_dir) else "Missing")

    with status_cols[2]:
        st.write("**ColBERT Index**")
        expected_colbert_index = (
            colbert_root
            / colbert_experiment_name
            / "indexes"
            / colbert_index_name
        )
        st.success("Found" if path_exists(expected_colbert_index) else "Missing")

    st.divider()

    if not query.strip():
        st.warning("Please enter a query.")
        return

    if not selected_methods:
        st.warning("Please select at least one retrieval method.")
        return

    if not search_clicked:
        st.info("Enter a query and click **Run Search**.")
        return

    tabs = st.tabs(selected_methods)

    for tab, method in zip(tabs, selected_methods):
        with tab:
            st.subheader(method)

            try:
                results = run_search(
                    method=method,
                    query=query,
                    top_k=top_k,
                    device=device,
                    passages_path=passages_path,
                    dense_index_dir=dense_index_dir,
                    colbert_root=colbert_root,
                    colbert_experiment_name=colbert_experiment_name,
                    colbert_index_name=colbert_index_name,
                )

                if not results:
                    st.warning("No results returned.")
                    continue

                for index, result in enumerate(results, start=1):
                    render_result(result, fallback_rank=index)

            except Exception as exc:
                st.error(f"{method} failed.")
                st.exception(exc)


if __name__ == "__main__":
    main()