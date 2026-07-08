# ColBERTv2-Based Semantic Search and RAG System

This project implements a research-paper-backed retrieval system inspired by ColBERTv2 and late interaction neural retrieval.

The goal is to compare multiple retrieval approaches on a scientific document corpus:

- BM25 keyword retrieval
- Dense retrieval with sentence-transformers and FAISS
- ColBERTv2 / late interaction retrieval
- Optional retrieval-augmented generation with source-grounded answers

## Dataset

The MVP uses the BEIR SciFact dataset, a scientific claim verification retrieval benchmark. The system retrieves relevant scientific paper abstracts/passages for a given claim or query.

## Planned Evaluation

| Method | Recall@5 | Recall@10 | MRR@10 | nDCG@10 | Notes |
|---|---:|---:|---:|---:|---|
| BM25 | TBD | TBD | TBD | TBD | Keyword baseline |
| Dense Retrieval | TBD | TBD | TBD | TBD | Sentence-transformers + FAISS |
| ColBERTv2 | TBD | TBD | TBD | TBD | Late interaction retrieval |

## Project Status

- [x] Dataset loading
- [x] Passage/chunk preprocessing
- [ ] BM25 baseline
- [ ] Dense retrieval with FAISS
- [ ] ColBERTv2 retrieval
- [ ] Evaluation metrics
- [ ] FastAPI backend
- [ ] Streamlit demo
- [ ] Optional RAG answer generation