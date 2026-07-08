# ColBERTv2-Based Semantic Search and RAG System

A research-paper-based information retrieval project that compares **BM25**, **dense retrieval**, and **ColBERTv2 late interaction retrieval** on a scientific document collection.

The main purpose of this project is not simply to call an existing language model or build another chatbot interface. My goal is to understand how modern retrieval systems work internally, implement multiple retrieval approaches, evaluate them using standard information retrieval metrics, and expose the resulting system through an API and an interactive demo.

The project is built around the ideas presented in the following research papers:

* **ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT**
* **ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction**
* **BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models**
* **Fact or Fiction: Verifying Scientific Claims**

---

## Project Motivation

Retrieval is one of the most important components of modern question-answering and Retrieval-Augmented Generation systems.

A language model may be capable of producing fluent answers, but it does not automatically know which external documents are relevant to a user query. Before an LLM can generate a grounded answer, the system must first retrieve useful evidence from a document collection.

This project focuses primarily on that retrieval stage.

Instead of starting with a large generative model, I first build and compare three different search approaches:

1. **BM25**, representing traditional lexical keyword retrieval.
2. **Dense retrieval**, representing single-vector semantic retrieval.
3. **ColBERTv2**, representing token-level multi-vector retrieval with late interaction.

By comparing these methods on the same dataset, queries, and relevance judgments, I can observe how retrieval architecture affects search quality.

The central technical question of the project is:

> How do lexical retrieval, single-vector dense retrieval, and token-level late interaction retrieval differ in their ability to find relevant scientific passages?

---

## Project Goals

The system is designed to:

* Accept a natural-language scientific query or claim.
* Search a collection of scientific documents.
* Retrieve the most relevant passages.
* Display ranked results with retrieval scores.
* Compare BM25, dense retrieval, and ColBERTv2.
* Evaluate each method using standard information retrieval metrics.
* Expose retrieval functionality through a FastAPI backend.
* Provide an interactive Streamlit interface.
* Optionally generate source-grounded answers with a local or external LLM.

The final system is intended to demonstrate that I can:

* Read and understand a research paper.
* Translate a research idea into a working software project.
* Build data ingestion and preprocessing pipelines.
* Implement classical and neural retrieval baselines.
* Work with vector embeddings and similarity search.
* Evaluate retrieval systems quantitatively.
* Design a modular backend API.
* Build a usable search interface.
* Structure an ML/NLP project as a maintainable repository.

---

## Why Retrieval Comes Before RAG

A typical RAG system contains two main components:

```text
User Query
    │
    ▼
Retriever
    │
    ▼
Relevant Passages
    │
    ▼
Generator / LLM
    │
    ▼
Grounded Answer
```

The **retriever** searches an external knowledge collection and selects passages that may answer the query.

The **generator** receives those passages as context and produces a natural-language response.

A RAG system can only generate a reliable answer when the retriever provides useful evidence. If retrieval fails, even a powerful language model may generate an incomplete, unsupported, or incorrect response.

For that reason, this project is intentionally retrieval-first.

The optional answer-generation component will only be added after the retrieval methods have been implemented and evaluated.

---

## Retrieval Methods

### 1. BM25

BM25 is a sparse lexical retrieval algorithm based primarily on keyword overlap.

It assigns higher scores to documents that contain important query terms while considering:

* Term frequency within the document.
* Rarity of the term across the collection.
* Document-length normalization.

BM25 does not generate neural embeddings and does not directly understand semantic similarity.

For example:

```text
Query:
heart attack treatment

Document:
treatment options for heart attack patients
```

BM25 can perform well because the query and document share important words.

However:

```text
Query:
heart attack treatment

Document:
therapy for myocardial infarction
```

The two texts are semantically related, but they have limited exact word overlap. A lexical retriever may therefore rank the second document lower than expected.

BM25 is included because it is:

* Fast.
* Interpretable.
* Easy to reproduce.
* Strong on rare technical terminology.
* A standard baseline for information retrieval experiments.

In this project, BM25 is implemented using `rank-bm25`.

---

### 2. Dense Retrieval

Dense retrieval represents the entire query and each passage as fixed-size numerical vectors called embeddings.

```text
Query text
    │
    ▼
Embedding model
    │
    ▼
Single query vector

Passage text
    │
    ▼
Embedding model
    │
    ▼
Single passage vector
```

The query vector is compared with passage vectors using a similarity measure such as cosine similarity or inner product.

Documents whose vectors are close to the query vector are considered semantically relevant.

This allows dense retrieval to match texts even when they do not contain the same exact words.

For example:

```text
heart attack
```

and:

```text
myocardial infarction
```

may receive similar vector representations because the embedding model has learned that these expressions are semantically related.

In this project:

* `sentence-transformers` generates embeddings.
* FAISS stores and searches passage vectors.
* The same SciFact queries are used for comparison with BM25.

A limitation of standard dense retrieval is that an entire passage is compressed into a single vector. Some fine-grained token-level information may be lost during this compression.

---

### 3. ColBERTv2 and Late Interaction

ColBERTv2 uses a different architecture from standard dense retrieval.

Instead of representing a query or document with one vector, it generates contextualized embeddings for individual tokens.

A simplified query representation may look like:

```text
Query:
"vitamin D deficiency"

Token embeddings:
vitamin     → vector q1
D           → vector q2
deficiency  → vector q3
```

A document is represented in the same way:

```text
Document tokens:
low         → vector d1
vitamin     → vector d2
levels      → vector d3
increase    → vector d4
disease     → vector d5
risk        → vector d6
```

During scoring, every query-token vector searches for its strongest matching document-token vector.

A simplified ColBERT score is:

```text
Score(query, document)
    = sum over query tokens(
        maximum similarity with any document token
      )
```

This operation is commonly described as **MaxSim**.

Conceptually:

```text
vitamin     → best matching document token
D           → best matching document token
deficiency  → best matching document token

Final score = sum of the best token-level similarities
```

This is called **late interaction** because:

* Queries and documents are encoded independently.
* Their token representations are stored separately.
* Fine-grained interaction happens later, during retrieval scoring.

This architecture attempts to combine two useful properties:

* The semantic understanding of transformer-based representations.
* The fine-grained matching behavior of token-level retrieval.

ColBERTv2 extends the original ColBERT architecture with techniques such as residual compression and denoised supervision, aiming to retain retrieval quality while reducing the storage cost of multi-vector representations.

---

## BM25 vs Dense Retrieval vs ColBERTv2

| Property                  | BM25                   | Dense Retrieval             | ColBERTv2                                 |
| ------------------------- | ---------------------- | --------------------------- | ----------------------------------------- |
| Representation            | Sparse term statistics | One vector per passage      | Multiple token vectors per passage        |
| Main matching signal      | Keyword overlap        | Global semantic similarity  | Fine-grained semantic token matching      |
| Understands synonyms      | Limited                | Usually strong              | Usually strong                            |
| Exact technical terms     | Strong                 | Model-dependent             | Strong token-level potential              |
| Storage requirement       | Low                    | Moderate                    | Higher than single-vector dense retrieval |
| GPU requirement           | No                     | Optional for small datasets | Usually recommended for indexing          |
| Interpretability          | Relatively high        | Lower                       | Medium                                    |
| Implementation complexity | Low                    | Medium                      | High                                      |
| Role in this project      | Lexical baseline       | Semantic baseline           | Main paper-based method                   |

The goal is not to assume that one method is always better.

For example:

* BM25 may outperform neural retrieval on rare scientific identifiers.
* Dense retrieval may perform better when queries and documents use different terminology.
* ColBERTv2 may preserve more fine-grained matching information than single-vector dense retrieval.
* Neural methods may also retrieve semantically related but non-relevant passages.

The project measures these differences instead of relying only on qualitative examples.

---

## Dataset

### BEIR SciFact

The MVP uses the **SciFact** dataset through the BEIR information retrieval benchmark.

SciFact is designed around scientific claims and evidence-containing research abstracts. Given a scientific claim, a retrieval system must identify documents containing relevant evidence.

This makes the dataset appropriate for the project because:

* It represents a real scientific retrieval task.
* It contains technical and domain-specific language.
* It provides relevance judgments for evaluation.
* It is small enough for local experimentation.
* It can be used to compare several retrieval architectures consistently.

The dataset is loaded through:

```python
ir_datasets.load("beir/scifact/test")
```

### Local Dataset Statistics

The current preprocessing pipeline produced:

| Item                               | Count |
| ---------------------------------- | ----: |
| Scientific documents               | 5,183 |
| Test queries                       |   300 |
| Query-document relevance judgments |   339 |
| Processed passages                 | 8,854 |

These values are generated directly from the local ingestion pipeline.

---

## Dataset Structure

The processed data is stored in JSON Lines format.

### Documents

```json
{
  "doc_id": "document-identifier",
  "title": "Scientific document title",
  "text": "Scientific abstract or document text",
  "source": "beir/scifact/test",
  "metadata": {
    "dataset": "SciFact",
    "task": "scientific claim verification",
    "original_doc_id": "document-identifier",
    "text_length": 1250
  }
}
```

### Queries

```json
{
  "query_id": "query-identifier",
  "text": "A scientific claim used as a retrieval query",
  "source": "beir/scifact/test",
  "metadata": {
    "dataset": "SciFact",
    "split": "test"
  }
}
```

### Relevance Judgments

A relevance judgment, commonly called a **qrel**, indicates which documents are considered relevant for a query.

```json
{
  "query_id": "query-identifier",
  "doc_id": "relevant-document-identifier",
  "relevance": 1
}
```

Qrels provide the ground truth required for retrieval evaluation.

### Passages

```json
{
  "passage_id": "document-identifier::p0",
  "doc_id": "document-identifier",
  "title": "Scientific document title",
  "text": "A chunk extracted from the document",
  "source": "beir/scifact/test",
  "metadata": {
    "chunk_index": 0,
    "start_word": 0,
    "end_word": 180,
    "chunk_size_words": 180,
    "chunk_overlap_words": 40
  }
}
```

---

## Passage Chunking

Retrieval systems often search passages rather than complete documents.

Long documents may contain several unrelated sections. Representing the entire document as one unit can dilute the relevant information, especially when using a single dense embedding.

The current preprocessing configuration uses:

```text
Chunk size:    180 words
Chunk overlap: 40 words
```

The overlap reduces the chance that an important sentence or concept is separated at a chunk boundary.

Each passage maintains a connection to its original document through:

```text
passage_id → unique passage identifier
doc_id     → original document identifier
```

This mapping is important because:

* Retrieval operates at passage level.
* SciFact relevance judgments are defined at document level.
* Evaluation must map retrieved passages back to their parent documents.

Chunking is treated as an experimental design choice rather than a universally optimal configuration. Future experiments may compare different chunk sizes and overlaps.

---

## System Architecture

```text
                         ┌─────────────────────┐
                         │    SciFact Dataset  │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Data Ingestion      │
                         │ JSONL Normalization │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Passage Chunking    │
                         │ 180 words / 40 ovlp │
                         └──────────┬──────────┘
                                    │
                ┌───────────────────┼───────────────────┐
                │                   │                   │
                ▼                   ▼                   ▼
        ┌───────────────┐   ┌───────────────┐   ┌───────────────┐
        │ BM25 Index    │   │ FAISS Index   │   │ ColBERT Index │
        │ Sparse Search │   │ Dense Search  │   │ Late Interact.│
        └───────┬───────┘   └───────┬───────┘   └───────┬───────┘
                │                   │                   │
                └───────────────────┼───────────────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Ranked Top-k Results│
                         │ Passage + Score     │
                         └──────────┬──────────┘
                                    │
                     ┌──────────────┴──────────────┐
                     │                             │
                     ▼                             ▼
          ┌─────────────────────┐       ┌─────────────────────┐
          │ Retrieval Evaluation│       │ Optional RAG Answer │
          │ Recall / MRR / nDCG │       │ Source-grounded LLM │
          └─────────────────────┘       └─────────────────────┘
```

---

## Current Pipeline

### 1. Dataset ingestion

The ingestion module:

* Downloads SciFact through `ir-datasets`.
* Loads documents.
* Loads test queries.
* Loads query-document relevance judgments.
* Converts the data into a consistent JSONL structure.
* Preserves source and metadata information.

Run:

```bash
python -m src.ingestion.load_scifact
```

Generated files:

```text
data/processed/documents.jsonl
data/processed/queries.jsonl
data/processed/qrels.jsonl
```

### 2. Passage generation

The chunking module:

* Reads normalized documents.
* Combines the title and abstract text.
* Splits documents into overlapping word-based passages.
* Assigns unique passage identifiers.
* Preserves parent-document mappings.

Run:

```bash
python -m src.ingestion.chunk_docs
```

Generated file:

```text
data/processed/passages.jsonl
```

### 3. BM25 indexing and retrieval

The BM25 module:

* Reads processed passages.
* Normalizes and tokenizes passage text.
* Builds an in-memory BM25 index.
* Scores every passage for an input query.
* Sorts results by BM25 score.
* Returns ranked passages with metadata.

Example:

```bash
python -m src.retrieval.bm25 \
  --query "Vitamin D deficiency increases the risk of multiple sclerosis" \
  --top-k 5
```

The returned result structure contains:

```text
rank
score
passage_id
doc_id
title
text
source
metadata
```

BM25 scores are ranking scores, not probabilities. They are primarily meaningful when comparing documents returned for the same query.

---

## Planned Dense Retrieval Pipeline

The dense retrieval baseline will use a sentence-transformer model to encode passages and queries.

The planned pipeline is:

```text
Passages
    │
    ▼
Sentence Transformer
    │
    ▼
Passage Embeddings
    │
    ▼
FAISS Index
```

At search time:

```text
Query
    │
    ▼
Sentence Transformer
    │
    ▼
Query Embedding
    │
    ▼
FAISS Similarity Search
    │
    ▼
Top-k Passages
```

The implementation will separate offline indexing from online search.

Offline indexing:

* Load passages.
* Encode passage text in batches.
* Normalize embeddings when cosine similarity is used.
* Create a FAISS index.
* Save the index and passage mapping to disk.

Online search:

* Load the saved index.
* Encode the query.
* Search the FAISS index.
* Return ranked passages with similarity scores.

---

## Planned ColBERTv2 Pipeline

The ColBERTv2 phase will:

* Convert processed passages into a ColBERT-compatible collection.
* Load a pretrained ColBERTv2 checkpoint.
* Generate token-level passage representations.
* Build a compressed multi-vector index.
* Encode incoming queries.
* Apply late-interaction scoring.
* Return ranked passage results.
* Compare results with BM25 and dense retrieval.

Because ColBERT indexing is more resource-intensive than BM25 or small-scale dense retrieval, this phase may use:

* WSL2 with a compatible Python environment.
* A dedicated Conda environment.
* A CUDA-enabled GPU.
* Google Colab for the initial indexing experiment.

The produced ColBERT index will remain compatible with the same processed SciFact collection used by the other retrieval methods.

---

## Retrieval Evaluation

The project uses qrels to compare retrieved documents with known relevant documents.

The main metrics are:

### Recall@5

Recall@5 measures how many relevant documents were retrieved within the first five results.

```text
Recall@5 =
relevant documents found in top 5
─────────────────────────────────
total relevant documents
```

A high Recall@5 means the retriever frequently places relevant evidence near the top of the result list.

### Recall@10

Recall@10 applies the same idea to the first ten results.

It shows whether increasing the retrieval depth allows the system to recover additional relevant evidence.

### MRR@10

Mean Reciprocal Rank focuses on the position of the first relevant result.

```text
Reciprocal Rank = 1 / rank of first relevant result
```

Examples:

```text
First relevant result at rank 1 → 1.00
First relevant result at rank 2 → 0.50
First relevant result at rank 5 → 0.20
No relevant result in top 10    → 0.00
```

MRR@10 is the average reciprocal rank across all evaluated queries, considering only the first ten results.

### nDCG@10

Normalized Discounted Cumulative Gain evaluates ranking quality while giving more importance to relevant results placed near the top.

The contribution of a result is discounted as its rank becomes lower.

nDCG is especially useful when relevance judgments contain multiple relevance levels. It is also commonly reported in retrieval benchmarks, allowing results to be compared with other systems.

---

## Planned Evaluation Table

The final measured results will be added after all retrieval methods are executed on the same query set.

| Method          | Recall@5 | Recall@10 | MRR@10 | nDCG@10 | Notes                         |
| --------------- | -------: | --------: | -----: | ------: | ----------------------------- |
| BM25            |      TBD |       TBD |    TBD |     TBD | Lexical keyword baseline      |
| Dense Retrieval |      TBD |       TBD |    TBD |     TBD | Sentence Transformers + FAISS |
| ColBERTv2       |      TBD |       TBD |    TBD |     TBD | Token-level late interaction  |

No metric will be added manually or copied from another implementation. All values in this table will be generated by this repository's evaluation pipeline.

---

## Passage-Level Retrieval and Document-Level Evaluation

One important evaluation detail is that the system retrieves passages, while SciFact qrels identify relevant documents.

A single document may produce multiple passages:

```text
Document 123
├── 123::p0
├── 123::p1
└── 123::p2
```

Before calculating document-level metrics:

1. Retrieved passages are mapped to their parent `doc_id`.
2. Duplicate document IDs are removed.
3. The highest-ranked passage determines the document's rank.
4. The resulting document ranking is compared with qrels.

Without this step, retrieving multiple passages from the same document could incorrectly occupy several ranking positions and distort evaluation results.

---

## Planned API

The backend will be implemented with FastAPI.

### Search endpoint

```http
GET /search?q=<query>&method=<method>&top_k=<number>
```

Supported retrieval methods:

```text
bm25
dense
colbert
```

Example:

```http
GET /search?q=vitamin%20d%20deficiency&method=bm25&top_k=5
```

Planned response:

```json
{
  "query": "vitamin d deficiency",
  "method": "bm25",
  "top_k": 5,
  "results": [
    {
      "rank": 1,
      "passage_id": "123::p0",
      "doc_id": "123",
      "title": "Example scientific paper",
      "text": "Retrieved passage text",
      "score": 8.4217,
      "source": "beir/scifact/test"
    }
  ]
}
```

### Optional RAG endpoint

```http
POST /rag/answer
```

The optional RAG endpoint will:

1. Receive a question.
2. Retrieve the top passages.
3. Construct a grounded prompt.
4. Ask an LLM to answer only from the supplied context.
5. Return the answer together with its source passages.

The retrieval response will remain available independently of the LLM so that the project can run without answer-generation latency.

---

## Planned Streamlit Interface

The Streamlit application will provide:

* A search input.
* Retrieval-method selection.
* Top-k selection.
* Ranked result cards.
* Passage text.
* Document title.
* Retrieval score.
* Source and document ID.
* Optional generated answer.
* Optional side-by-side method comparison.

The UI is intended as a demonstration layer. Retrieval logic remains inside reusable Python modules rather than being implemented directly in the Streamlit file.

---

## Optional RAG Answer Generation

Answer generation is intentionally optional.

The first objective is to build a measurable retrieval system. After retrieval quality is established, the top passages can be provided to an LLM.

A simplified prompt format will be:

```text
You are given a question and a collection of retrieved scientific passages.

Answer the question using only the supplied passages.
Do not introduce unsupported information.
If the passages do not contain sufficient evidence, state that the available
sources are insufficient.

Question:
{query}

Sources:
[1] {passage_1}
[2] {passage_2}
[3] {passage_3}
[4] {passage_4}
[5] {passage_5}

Answer:
```

The final response will show both the answer and the passages used to construct it.

Possible generator backends include:

* Ollama.
* A small local instruction model.
* A hosted language-model API.

The retrieval system will not depend on a specific generator.

---

## Repository Structure

```text
colbertv2-rag-search-engine/
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
│
├── data/
│   ├── raw/
│   └── processed/
│       ├── documents.jsonl
│       ├── queries.jsonl
│       ├── qrels.jsonl
│       └── passages.jsonl
│
├── indexes/
│   ├── dense/
│   └── colbert/
│
├── src/
│   ├── __init__.py
│   │
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── load_scifact.py
│   │   └── chunk_docs.py
│   │
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── bm25.py
│   │   ├── dense.py
│   │   └── colbert.py
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py
│   │   └── run_eval.py
│   │
│   ├── api/
│   │   ├── __init__.py
│   │   └── main.py
│   │
│   └── rag/
│       ├── __init__.py
│       └── answer_generator.py
│
├── app/
│   └── streamlit_app.py
│
├── configs/
│   ├── retrieval.yaml
│   └── models.yaml
│
├── experiments/
├── notebooks/
└── tests/
    ├── test_ingestion.py
    ├── test_bm25.py
    ├── test_dense.py
    └── test_metrics.py
```

Generated datasets, indexes, model files, and local environment files are excluded from Git through `.gitignore`.

---

## Installation

### Requirements

The initial BM25 and dense-retrieval phases are designed to run locally.

Recommended environment:

```text
Python 3.11 or newer
Windows, Linux, or WSL2
CPU for BM25
CPU or GPU for dense embedding generation
GPU recommended for ColBERTv2 indexing
```

### Clone the repository

```bash
git clone <repository-url>
cd colbertv2-rag-search-engine
```

### Create a virtual environment

Windows CMD:

```cmd
python -m venv .venv
.venv\Scripts\activate.bat
```

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Linux or macOS:

```bash
python -m venv .venv
source .venv/bin/activate
```

### Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Prepare the dataset

```bash
python -m src.ingestion.load_scifact
python -m src.ingestion.chunk_docs
```

### Run BM25 search

```bash
python -m src.retrieval.bm25 \
  --query "Vitamin D deficiency increases the risk of multiple sclerosis" \
  --top-k 5
```

---

## Technology Stack

| Area                 | Technology                    |
| -------------------- | ----------------------------- |
| Programming language | Python                        |
| Sparse retrieval     | rank-bm25                     |
| Embedding models     | sentence-transformers         |
| Vector search        | FAISS                         |
| Late interaction     | ColBERT / ColBERTv2           |
| Model execution      | PyTorch                       |
| Data processing      | pandas, NumPy                 |
| Dataset access       | ir-datasets                   |
| Backend API          | FastAPI                       |
| Demo interface       | Streamlit                     |
| Testing              | pytest                        |
| Optional deployment  | Docker                        |
| Optional generation  | Ollama or another LLM backend |

---

## Engineering Principles

The repository follows several design principles.

### Reproducibility

* Dataset preparation is script-based.
* Evaluation uses fixed queries and qrels.
* Generated metrics will come from executable code.
* Model and indexing configuration will be stored explicitly.

### Separation of concerns

* Ingestion prepares data.
* Retrieval modules implement search methods.
* Evaluation modules calculate metrics.
* API modules expose retrieval services.
* Streamlit provides only the presentation layer.
* RAG generation remains optional and independent.

### Comparable experiments

All methods will use:

* The same document collection.
* The same passage representation where technically appropriate.
* The same queries.
* The same qrels.
* The same evaluation metrics.
* The same top-k definitions.

### Honest reporting

The README distinguishes between:

* Implemented features.
* Planned features.
* Measured results.
* Expected behavior.

Incomplete components are not presented as finished, and evaluation values are not reported before the experiment is run.

---

## Current Project Status

* [x] Repository structure
* [x] Python virtual environment
* [x] SciFact dataset loading
* [x] Document normalization
* [x] Query extraction
* [x] Qrel extraction
* [x] Passage chunking
* [x] BM25 retrieval baseline
* [ ] BM25 automated tests
* [ ] Dense passage embedding generation
* [ ] FAISS vector index
* [ ] Dense retrieval search
* [ ] ColBERTv2-compatible collection
* [ ] ColBERTv2 index
* [ ] Late-interaction search
* [ ] Recall@5
* [ ] Recall@10
* [ ] MRR@10
* [ ] nDCG@10
* [ ] FastAPI search service
* [ ] Streamlit demo
* [ ] Optional source-grounded RAG generation
* [ ] Docker configuration
* [ ] Final experiment report

---

## Development Roadmap

### Phase 1 — Dataset and preprocessing

* Load SciFact.
* Normalize documents, queries, and qrels.
* Divide documents into passages.
* Preserve passage-to-document mappings.

**Status: Completed**

### Phase 2 — BM25 baseline

* Build a keyword-based passage index.
* Run top-k retrieval.
* Return scores and metadata.
* Add BM25 tests.
* Evaluate BM25 on all test queries.

**Status: Core retrieval completed**

### Phase 3 — Dense retrieval

* Select a sentence-transformer model.
* Generate passage embeddings.
* Build a FAISS index.
* Implement query encoding and nearest-neighbor search.
* Compare results with BM25.

**Status: Next phase**

### Phase 4 — ColBERTv2

* Prepare a ColBERT-compatible collection.
* Build a token-level index.
* Implement late-interaction retrieval.
* Inspect qualitative differences between methods.

**Status: Planned**

### Phase 5 — Evaluation

* Map passages to document IDs.
* Calculate Recall@5.
* Calculate Recall@10.
* Calculate MRR@10.
* Calculate nDCG@10.
* Generate a Markdown result table.

**Status: Planned**

### Phase 6 — API

* Initialize FastAPI.
* Create method-independent search schemas.
* Add BM25, dense, and ColBERT search options.
* Add validation and error handling.
* Generate OpenAPI documentation.

**Status: Planned**

### Phase 7 — User interface

* Build the Streamlit search interface.
* Display ranked result cards.
* Add method and top-k controls.
* Display scores and metadata.

**Status: Planned**

### Phase 8 — Optional RAG

* Retrieve top passages.
* Construct a grounded prompt.
* Connect an optional LLM backend.
* Return answers with sources.

**Status: Planned**

---

## Planned Experiments

In addition to the main method comparison, the following experiments may be included:

### Chunk-size comparison

```text
100 words / 20 overlap
180 words / 40 overlap
250 words / 50 overlap
```

This experiment will measure how passage size affects retrieval quality.

### Dense-model comparison

Candidate sentence-transformer models may be compared using the same FAISS pipeline.

The final model will not be selected only by popularity. Selection will consider:

* Retrieval quality.
* Embedding dimension.
* Index size.
* Encoding speed.
* CPU performance.
* Domain suitability.

### Qualitative error analysis

Queries will be inspected where:

* BM25 succeeds and dense retrieval fails.
* Dense retrieval succeeds and BM25 fails.
* ColBERTv2 improves over both baselines.
* All methods fail.

This analysis will help explain why metric differences occur.

### Latency comparison

The project may record:

```text
Index build time
Query encoding time
Average retrieval latency
Index size
Memory usage
```

This allows retrieval quality to be evaluated together with computational cost.

---

## Limitations

The initial project has several known limitations:

* SciFact is relatively small compared with production-scale corpora.
* Most source documents are scientific abstracts rather than full papers.
* The initial tokenizer used for BM25 is intentionally simple.
* Word-based chunking does not understand sentence or section boundaries.
* Dense retrieval quality depends heavily on the selected embedding model.
* ColBERTv2 indexing may require a separate GPU-compatible environment.
* SciFact qrels may not identify every passage that is semantically related to a query.
* Retrieval relevance does not automatically imply that a generated answer is factually correct.
* The optional RAG component will require separate faithfulness evaluation.

These limitations are treated as opportunities for further experimentation rather than hidden implementation details.

---

## Future Improvements

Possible extensions include:

* Hybrid BM25 and dense retrieval.
* Reciprocal Rank Fusion.
* Neural reranking with a cross-encoder.
* Query expansion.
* Sentence-aware chunking.
* Full-paper PDF ingestion.
* Additional BEIR datasets.
* NFCorpus evaluation.
* Scientific-domain embedding models.
* Retrieval latency benchmarks.
* Index persistence and caching.
* Asynchronous FastAPI endpoints.
* Dockerized API and UI services.
* Source citation extraction.
* RAG faithfulness evaluation.
* Experiment tracking.
* Automated CI tests with GitHub Actions.

A particularly useful extension would be a hybrid pipeline:

```text
BM25 candidates
        +
Dense candidates
        │
        ▼
Rank fusion or reranker
        │
        ▼
Final ranked results
```

This would test whether lexical and semantic retrieval signals complement each other.

---

## Research References

1. Keshav Santhanam, Omar Khattab, Jon Saad-Falcon, Christopher Potts, and Matei Zaharia.
   **ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction.**
   NAACL 2022.

2. Omar Khattab and Matei Zaharia.
   **ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT.**
   SIGIR 2020.

3. Nandan Thakur, Nils Reimers, Andreas Rücklé, Abhishek Srivastava, and Iryna Gurevych.
   **BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models.**
   NeurIPS 2021 Datasets and Benchmarks Track.

4. David Wadden, Shanchuan Lin, Kyle Lo, Lucy Lu Wang, Madeleine van Zuylen, Arman Cohan, and Hannaneh Hajishirzi.
   **Fact or Fiction: Verifying Scientific Claims.**
   EMNLP 2020.

---

## Author's Note

I developed this project as a practical study of modern information retrieval.

Rather than treating RAG as a single model or API call, I wanted to separate its components and understand the retrieval stage in detail. The project therefore starts with a traditional lexical baseline, progresses to dense semantic retrieval, and then implements the token-level late interaction approach introduced by ColBERT and improved by ColBERTv2.

The final goal is not only to produce a working search application, but also to create a reproducible comparison showing where different retrieval architectures succeed, where they fail, and what engineering trade-offs they introduce.
