# ColBERTv2-Based Semantic Search and RAG System

A research-paper-based information retrieval project that compares **BM25**, **single-vector dense retrieval**, and **ColBERTv2 late interaction retrieval** on a scientific document collection.

The main purpose of this project is not to build another application that only sends a prompt to a language model. My goal is to understand how modern retrieval systems work, implement their core pipelines, compare different retrieval architectures, and evaluate them using reproducible information retrieval metrics.

The system is being developed in progressive stages:

1. Scientific document ingestion and passage preparation
2. BM25 lexical retrieval
3. Dense semantic retrieval with Sentence Transformers and FAISS
4. ColBERTv2 token-level late interaction retrieval
5. Quantitative retrieval evaluation
6. FastAPI search backend
7. Streamlit demo interface
8. Optional source-grounded RAG answer generation

---

## Project Motivation

Retrieval is a central component of modern search engines, question-answering systems, and Retrieval-Augmented Generation applications.

A language model may generate fluent text, but it does not automatically know which external documents are relevant to a query. A RAG pipeline must first search a document collection and retrieve useful evidence before asking a generator to produce an answer.

A simplified RAG pipeline looks like this:

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

The quality of the final answer depends heavily on the retrieval stage.

If the retriever fails to find the relevant evidence, even a powerful language model may produce an incomplete, unsupported, or incorrect answer. For this reason, this project follows a **retrieval-first** development strategy.

The optional language-model component will be added only after the retrieval methods have been implemented and evaluated.

---

## Main Research Question

This project investigates the following question:

> How do lexical retrieval, single-vector dense retrieval, and token-level late interaction retrieval differ when searching scientific documents?

The three retrieval approaches represent different ways of matching a query with a document:

* **BM25** relies mainly on lexical and keyword overlap.
* **Dense retrieval** maps an entire query and an entire passage into one vector each.
* **ColBERTv2** preserves token-level representations and applies late interaction during scoring.

The project compares these methods using the same corpus, queries, relevance judgments, and evaluation metrics.

---

## Project Objectives

The completed system is intended to:

* Accept a natural-language scientific query or claim.
* Search a collection of scientific documents.
* Retrieve the most relevant passages.
* Display ranked results with retrieval scores.
* Compare lexical and neural retrieval approaches.
* Evaluate retrieval quality using standard metrics.
* Expose the search system through a FastAPI backend.
* Provide an interactive Streamlit interface.
* Optionally generate answers grounded in retrieved passages.
* Display the sources used for answer generation.

From an engineering perspective, the project demonstrates:

* Research paper comprehension
* Data ingestion and normalization
* Passage chunking
* Sparse retrieval
* Neural embedding generation
* Vector indexing
* Similarity search
* Retrieval evaluation
* Experiment reporting
* API development
* Demo application development
* Modular project architecture

---

# Retrieval Methods

## 1. BM25 Lexical Retrieval

BM25 is a traditional sparse retrieval algorithm based primarily on term matching.

It considers:

* How often a query term occurs in a document
* How rare or informative the term is across the corpus
* The length of the document relative to the average document length

A simplified example:

```text
Query:
vitamin D deficiency

Document:
Vitamin D deficiency may affect immune regulation.
```

The document receives a strong score because it contains important query terms.

BM25 is often effective for:

* Exact terminology
* Scientific identifiers
* Rare domain-specific words
* Product codes
* Gene names
* Abbreviations
* Proper nouns

However, BM25 does not directly understand semantic equivalence.

For example:

```text
Query:
heart attack treatment

Document:
therapy for myocardial infarction
```

These expressions are semantically related, but their lexical overlap is limited.

BM25 is included because it provides a strong, efficient, and interpretable baseline against which neural retrieval methods can be compared.

The implementation uses:

```text
rank-bm25
BM25Okapi
```

---

## 2. Dense Semantic Retrieval

Dense retrieval represents a query and a passage as fixed-size numerical vectors called embeddings.

```text
Query
  │
  ▼
Embedding Model
  │
  ▼
Single Query Vector
```

```text
Passage
  │
  ▼
Embedding Model
  │
  ▼
Single Passage Vector
```

The vectors are compared using a similarity function.

In this project:

* The embedding model is `sentence-transformers/all-mpnet-base-v2`
* Each query is represented by one 768-dimensional vector
* Each passage is represented by one 768-dimensional vector
* Embeddings are normalized
* FAISS `IndexFlatIP` performs exact nearest-neighbor search
* Inner product over normalized vectors is interpreted as cosine similarity

Dense retrieval can match semantically related expressions even when exact words differ.

For example:

```text
heart attack
```

and:

```text
myocardial infarction
```

may receive similar vector representations.

### Single-vector limitation

A complete passage is compressed into one vector:

```text
Passage containing many concepts
              │
              ▼
       One 768D vector
```

This is efficient, but some fine-grained token-level information may be lost during compression.

This limitation provides the motivation for the later ColBERTv2 phase.

---

## 3. ColBERTv2 and Late Interaction

ColBERTv2 uses a different representation strategy from standard dense retrieval.

Instead of creating one vector for an entire query and one vector for an entire passage, it creates contextualized vectors for individual tokens.

Simplified query representation:

```text
Query:
vitamin D deficiency

Token representations:
vitamin     → q1
D           → q2
deficiency  → q3
```

Simplified passage representation:

```text
low         → d1
vitamin     → d2
levels      → d3
immune      → d4
risk        → d5
```

During scoring, each query-token vector searches for its strongest matching passage-token vector.

A simplified ColBERT scoring function is:

```text
Score(query, passage)
    =
sum over query tokens(
    maximum similarity with any passage token
)
```

This is commonly described as a **MaxSim** operation.

The process is called **late interaction** because:

1. Queries and passages are encoded independently.
2. Their token representations are stored separately.
3. Fine-grained query-document interaction happens later during scoring.

ColBERTv2 aims to combine:

* Transformer-based semantic understanding
* Token-level matching
* Independent query and document encoding
* Efficient retrieval through compressed multi-vector representations

The ColBERTv2 implementation is planned after the BM25 and dense baselines have been fully evaluated.

---

# Method Comparison

| Property                  | BM25                   | Dense Retrieval            | ColBERTv2                           |
| ------------------------- | ---------------------- | -------------------------- | ----------------------------------- |
| Representation            | Sparse term statistics | One vector per passage     | Multiple token vectors per passage  |
| Primary signal            | Keyword overlap        | Global semantic similarity | Fine-grained semantic matching      |
| Synonym handling          | Limited                | Generally strong           | Generally strong                    |
| Exact technical terms     | Strong                 | Model-dependent            | Token-level potential               |
| Storage requirement       | Low                    | Moderate                   | Higher than single-vector retrieval |
| Query processing          | Very fast              | Requires embedding model   | Requires token encoding and MaxSim  |
| GPU requirement           | No                     | Optional                   | Usually recommended                 |
| Interpretability          | Relatively high        | Lower                      | Medium                              |
| Implementation complexity | Low                    | Medium                     | High                                |
| Project role              | Lexical baseline       | Semantic baseline          | Main paper-based method             |

The project does not assume that a neural method is always superior.

For example:

* BM25 may perform better on rare identifiers.
* Dense retrieval may perform better on paraphrases.
* Dense retrieval may retrieve semantically related but irrelevant documents.
* ColBERTv2 may preserve details that are lost in a single passage vector.
* ColBERTv2 introduces additional storage and indexing complexity.

These trade-offs will be measured rather than assumed.

---

# Dataset

## BEIR SciFact

The MVP uses the **SciFact** dataset through the BEIR retrieval benchmark.

SciFact contains scientific claims and research abstracts that may provide evidence related to those claims.

This dataset was selected because:

* It represents a real scientific retrieval task.
* It contains domain-specific terminology.
* It includes queries and relevance judgments.
* It supports quantitative evaluation.
* It is small enough for local development.
* It is suitable for comparing multiple retrieval methods.

The dataset is loaded with:

```python
ir_datasets.load("beir/scifact/test")
```

---

## Local Dataset Statistics

The current ingestion pipeline produced:

| Data type                          | Count |
| ---------------------------------- | ----: |
| Scientific documents               | 5,183 |
| Test queries                       |   300 |
| Query-document relevance judgments |   339 |
| Processed passages                 | 8,854 |

These values were generated by the local preprocessing scripts rather than copied from an external benchmark table.

---

# Data Structure

## Documents

Each normalized document contains:

```json
{
  "doc_id": "22843838",
  "title": "Scientific article title",
  "text": "Scientific abstract text",
  "source": "beir/scifact/test",
  "metadata": {
    "dataset": "SciFact",
    "task": "scientific claim verification",
    "original_doc_id": "22843838",
    "text_length": 1250
  }
}
```

## Queries

```json
{
  "query_id": "query-identifier",
  "text": "A scientific claim or search query",
  "source": "beir/scifact/test",
  "metadata": {
    "dataset": "SciFact",
    "split": "test"
  }
}
```

## Relevance Judgments

A relevance judgment, or qrel, indicates which document is considered relevant for a query.

```json
{
  "query_id": "query-identifier",
  "doc_id": "relevant-document-identifier",
  "relevance": 1
}
```

Qrels provide the ground truth required for retrieval evaluation.

## Passages

```json
{
  "passage_id": "22843838::p0",
  "doc_id": "22843838",
  "title": "Scientific article title",
  "text": "A passage extracted from the article",
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

# Passage Chunking

The retrieval system searches passages rather than only complete documents.

Long documents may contain several topics. Encoding or indexing a complete document as one unit can dilute the relevant information. Passage retrieval allows the system to return a more focused section.

The current configuration uses:

```text
Chunk size:    180 words
Chunk overlap: 40 words
```

The overlap reduces the chance that an important sentence or concept will be lost at a chunk boundary.

Each passage preserves its relationship with the source document:

```text
passage_id → unique passage
doc_id     → original document
```

This distinction is important because retrieval occurs at passage level, while SciFact qrels are defined at document level.

---

# System Architecture

```text
                         ┌──────────────────────┐
                         │   SciFact Dataset    │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │ Data Ingestion       │
                         │ JSONL Normalization  │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │ Passage Chunking     │
                         │ 180 words / 40 ovlp  │
                         └──────────┬───────────┘
                                    │
                 ┌──────────────────┼───────────────────┐
                 │                  │                   │
                 ▼                  ▼                   ▼
        ┌────────────────┐ ┌────────────────┐ ┌────────────────┐
        │ BM25 Index     │ │ FAISS Index    │ │ ColBERT Index  │
        │ Sparse Search  │ │ Dense Search   │ │ Late Interaction│
        └───────┬────────┘ └───────┬────────┘ └───────┬────────┘
                │                  │                   │
                └──────────────────┼───────────────────┘
                                   │
                                   ▼
                         ┌──────────────────────┐
                         │ Ranked Top-k Results │
                         │ Passage + Score      │
                         └──────────┬───────────┘
                                    │
                   ┌────────────────┴────────────────┐
                   │                                 │
                   ▼                                 ▼
        ┌──────────────────────┐          ┌──────────────────────┐
        │ Retrieval Evaluation │          │ Optional RAG Answer  │
        │ Recall / MRR / nDCG  │          │ Source-grounded LLM  │
        └──────────────────────┘          └──────────────────────┘
```

---

# Implemented Pipeline

## 1. SciFact ingestion

The ingestion module:

* Downloads SciFact through `ir-datasets`
* Loads scientific documents
* Loads test queries
* Loads relevance judgments
* Converts the data into consistent JSONL files
* Preserves source and metadata fields

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

---

## 2. Passage generation

The chunking module:

* Reads normalized documents
* Combines document titles and abstract text
* Splits documents into overlapping word-based passages
* Assigns unique passage IDs
* Preserves original document IDs
* Saves chunk metadata

Run:

```bash
python -m src.ingestion.chunk_docs
```

Generated file:

```text
data/processed/passages.jsonl
```

Current output:

```text
Documents processed: 5,183
Passages created:    8,854
```

---

## 3. BM25 retrieval

The BM25 module:

* Loads processed passages
* Applies lowercase regex tokenization
* Builds an in-memory BM25 index
* Scores every passage for a query
* Sorts results by BM25 score
* Returns passage text, title, document ID, source, and metadata

Run:

```bash
python -m src.retrieval.bm25 \
  --query "Vitamin D deficiency increases the risk of multiple sclerosis" \
  --top-k 5
```

BM25 scores are ranking scores.

They are:

* Not probabilities
* Not percentages
* Not directly comparable across unrelated queries
* Not directly comparable with dense cosine similarity scores

---

## 4. Dense index construction

The dense indexing pipeline:

1. Loads all processed passages
2. Encodes passage text with `all-mpnet-base-v2`
3. Generates normalized 768-dimensional embeddings
4. Creates a FAISS `IndexFlatIP` index
5. Adds all passage vectors
6. Saves the FAISS index
7. Saves passage-to-vector mappings
8. Saves index metadata

Run:

```bash
python -m src.retrieval.build_dense_index --batch-size 32
```

Generated files:

```text
indexes/dense/index.faiss
indexes/dense/passages.jsonl
indexes/dense/metadata.json
```

Generated index files are excluded from Git because they can be reproduced from the source data and code.

---

## 5. Dense retrieval

At query time:

1. The FAISS index is loaded
2. The same embedding model is loaded
3. The query is encoded
4. The query vector is normalized
5. FAISS performs exact inner-product search
6. The top passages are returned

Run:

```bash
python -m src.retrieval.dense \
  --query "Vitamin D deficiency increases the risk of multiple sclerosis" \
  --top-k 5
```

Because both passage and query embeddings are normalized, the inner-product score corresponds to cosine similarity.

The score is not a probability or correctness percentage.

---

## 6. BM25 and dense comparison

The comparison module runs both methods for the same query.

```bash
python -m src.retrieval.compare \
  --query "Vitamin D deficiency increases the risk of multiple sclerosis" \
  --top-k 5
```

The module reports:

* BM25 top-k results
* Dense top-k results
* Common passages
* Common documents
* Unique document counts
* Document-level Jaccard overlap

Results can also be saved:

```bash
python -m src.retrieval.compare \
  --query "Vitamin D deficiency increases the risk of multiple sclerosis" \
  --top-k 5 \
  --output experiments/vitamin_d_comparison.json
```

The overlap values are qualitative analysis tools. They are not replacements for Recall, MRR, or nDCG.

---

# Initial Dense Index Experiment

The first dense index was generated locally using a CPU environment.

| Property                   |                                        Value |
| -------------------------- | -------------------------------------------: |
| Indexed passages           |                                        8,854 |
| Saved passage mappings     |                                        8,854 |
| Embedding model            |    `sentence-transformers/all-mpnet-base-v2` |
| Embedding dimension        |                                          768 |
| FAISS index                |                                `IndexFlatIP` |
| Similarity                 | Normalized inner product / cosine similarity |
| Embedding device           |                                          CPU |
| Batch size                 |                                           32 |
| Build time                 |                               934.09 seconds |
| Approximate build duration |                        15 minutes 34 seconds |

This is an offline indexing cost. Passage embeddings do not need to be regenerated for every query.

---

# Initial Qualitative Retrieval Experiment

The following query was tested:

```text
Vitamin D deficiency increases the risk of multiple sclerosis
```

## BM25 top result

```text
Document ID: 22843838
Passage ID:  22843838::p0
Title:       [Vitamin D and latitude as environmental factors in multiple sclerosis].
BM25 score:  36.6362
```

## Dense top result

```text
Document ID: 22843838
Passage ID:  22843838::p0
Title:       [Vitamin D and latitude as environmental factors in multiple sclerosis].
Dense score: 0.7737
```

Both methods returned the same highly relevant passage at rank one.

The dense retriever also returned passages related to:

* Sunlight exposure
* Ultraviolet radiation
* Vitamin D production
* Multiple sclerosis risk

This demonstrates the ability of dense retrieval to capture concepts that are semantically related even when the exact query wording is not repeated.

However, lower-ranked dense results also included passages primarily focused on general vitamin D research. These passages were semantically close to the query but did not fully address multiple sclerosis.

This is an example of **semantic drift**:

> A passage may be semantically related to the general topic while failing to satisfy the complete information need.

---

## Top-5 overlap

The first comparison produced:

| Measurement              |  Value |
| ------------------------ | -----: |
| Common passages          |      2 |
| Common documents         |      2 |
| Unique BM25 documents    |      4 |
| Unique dense documents   |      4 |
| Document Jaccard overlap | 0.3333 |

Common document IDs:

```text
22843838
23267371
```

A Jaccard value of `0.3333` shows that the two retrieval methods shared some results while also producing meaningfully different rankings.

This is expected because:

* BM25 prioritizes lexical overlap
* Dense retrieval prioritizes embedding similarity
* Both methods can find relevant evidence through different signals

No conclusion about the superior method will be made from one query. The complete test set will be evaluated using qrels.

---

# Passage-Level Retrieval and Document-Level Evaluation

The system retrieves passages:

```text
22843838::p0
22843838::p1
```

However, SciFact qrels identify relevant documents:

```text
22843838
```

Multiple retrieved passages may therefore belong to the same source document.

Before evaluation:

1. Each retrieved passage will be mapped to its parent `doc_id`
2. Duplicate document IDs will be removed
3. The highest-ranked passage will define the document rank
4. The deduplicated document ranking will be compared with qrels

Example:

```text
Passage ranking:
1. A::p1
2. A::p0
3. B::p0
4. C::p2
```

Document ranking:

```text
1. A
2. B
3. C
```

Without deduplication, one document could occupy several positions and distort the evaluation.

---

# Planned Evaluation Metrics

## Recall@5

Recall@5 measures how many relevant documents are found in the first five retrieved documents.

```text
Recall@5 =
relevant documents found in top 5
─────────────────────────────────
total relevant documents
```

A high Recall@5 indicates that relevant evidence is frequently retrieved near the top.

---

## Recall@10

Recall@10 uses the same calculation for the first ten results.

It shows whether increasing retrieval depth recovers additional relevant documents.

---

## MRR@10

Mean Reciprocal Rank focuses on the rank of the first relevant result.

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

MRR@10 averages this value across all evaluated queries.

---

## nDCG@10

Normalized Discounted Cumulative Gain evaluates ranking quality while giving higher importance to relevant documents appearing near the top.

Relevant results at lower positions contribute less because their gain is discounted by rank.

nDCG is especially useful when relevance judgments contain graded relevance levels.

---

## Planned evaluation table

All values will be generated by this repository's own evaluation pipeline.

| Method          | Recall@5 | Recall@10 | MRR@10 | nDCG@10 | Notes                         |
| --------------- | -------: | --------: | -----: | ------: | ----------------------------- |
| BM25            |      TBD |       TBD |    TBD |     TBD | Lexical baseline              |
| Dense Retrieval |      TBD |       TBD |    TBD |     TBD | Sentence Transformers + FAISS |
| ColBERTv2       |      TBD |       TBD |    TBD |     TBD | Late interaction retrieval    |

No metric will be manually estimated or copied from another experiment.

---

# Planned ColBERTv2 Pipeline

The ColBERTv2 phase will:

* Convert passages into a ColBERT-compatible collection
* Load a pretrained ColBERTv2 checkpoint
* Encode passages into contextualized token vectors
* Build a compressed multi-vector index
* Encode incoming queries
* Apply late-interaction MaxSim scoring
* Return ranked passages
* Compare the results with BM25 and dense retrieval
* Evaluate all methods on the same test queries

ColBERTv2 may require a separate environment because of:

* PyTorch compatibility
* FAISS compatibility
* CUDA requirements
* Windows support limitations
* GPU memory requirements

Possible environments include:

* WSL2
* Conda
* Linux
* Google Colab
* CUDA-enabled local environment

---

# Planned FastAPI Backend

The backend will expose a method-independent search API.

## Search endpoint

```http
GET /search?q=<query>&method=<method>&top_k=<number>
```

Supported methods:

```text
bm25
dense
colbert
```

Example:

```http
GET /search?q=vitamin%20d%20deficiency&method=dense&top_k=5
```

Planned response:

```json
{
  "query": "vitamin d deficiency",
  "method": "dense",
  "top_k": 5,
  "results": [
    {
      "rank": 1,
      "passage_id": "22843838::p0",
      "doc_id": "22843838",
      "title": "Scientific article title",
      "text": "Retrieved passage",
      "score": 0.7737,
      "source": "beir/scifact/test"
    }
  ]
}
```

Retriever instances will be loaded once when the API starts. They will not be reloaded for every request.

---

# Planned Streamlit Interface

The Streamlit application will include:

* Search input
* Retrieval method selection
* Top-k selection
* Ranked result cards
* Passage title
* Passage text
* Retrieval score
* Source
* Passage ID
* Document ID
* Optional side-by-side comparison
* Optional RAG answer section

The interface will remain separate from retrieval logic.

---

# Optional RAG Generation

The RAG stage will be optional because retrieval can be tested independently of an LLM.

The planned process is:

```text
Question
   │
   ▼
Retriever
   │
   ▼
Top 5 passages
   │
   ▼
Grounded prompt
   │
   ▼
LLM
   │
   ▼
Answer + sources
```

A simplified grounding instruction:

```text
Answer the question using only the supplied scientific passages.

Do not introduce unsupported information.

If the passages do not contain sufficient evidence, state that the
available sources are insufficient.

Question:
{query}

Sources:
[1] {passage_1}
[2] {passage_2}
[3] {passage_3}
[4] {passage_4}
[5] {passage_5}
```

Potential generator backends:

* Ollama
* A small local instruction model
* A hosted language-model API

The retrieval architecture will not depend on a particular generator.

---

# Repository Structure

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
│   │   ├── index.faiss
│   │   ├── passages.jsonl
│   │   └── metadata.json
│   └── colbert/
│
├── experiments/
│   └── retrieval_comparisons/
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
│   │   ├── build_dense_index.py
│   │   ├── dense.py
│   │   ├── compare.py
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
│   ├── models.yaml
│   └── retrieval.yaml
│
├── notebooks/
│
└── tests/
    ├── test_ingestion.py
    ├── test_bm25.py
    ├── test_dense.py
    └── test_metrics.py
```

Generated data, model files, and indexes are excluded from version control.

---

# Installation

## Clone the repository

```bash
git clone <repository-url>
cd colbertv2-rag-search-engine
```

## Create a virtual environment

### Windows CMD

```cmd
python -m venv .venv
.venv\Scripts\activate.bat
```

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Linux or macOS

```bash
python -m venv .venv
source .venv/bin/activate
```

## Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Verify the environment

```bash
python -c "import ir_datasets, rank_bm25, faiss, torch, sentence_transformers; print('Environment OK')"
```

---

# Usage

## Prepare SciFact

```bash
python -m src.ingestion.load_scifact
python -m src.ingestion.chunk_docs
```

## Run BM25

```bash
python -m src.retrieval.bm25 \
  --query "Vitamin D deficiency increases the risk of multiple sclerosis" \
  --top-k 5
```

## Build the dense index

```bash
python -m src.retrieval.build_dense_index --batch-size 32
```

## Run dense retrieval

```bash
python -m src.retrieval.dense \
  --query "Vitamin D deficiency increases the risk of multiple sclerosis" \
  --top-k 5
```

## Compare BM25 and dense retrieval

```bash
python -m src.retrieval.compare \
  --query "Vitamin D deficiency increases the risk of multiple sclerosis" \
  --top-k 5
```

## Save a comparison experiment

```bash
python -m src.retrieval.compare \
  --query "Vitamin D deficiency increases the risk of multiple sclerosis" \
  --top-k 5 \
  --output experiments/vitamin_d_comparison.json
```

---

# Technology Stack

| Area                | Technology            |
| ------------------- | --------------------- |
| Language            | Python                |
| Dataset access      | ir-datasets           |
| Sparse retrieval    | rank-bm25             |
| Embeddings          | sentence-transformers |
| Dense model         | all-mpnet-base-v2     |
| Vector search       | FAISS                 |
| Late interaction    | ColBERT / ColBERTv2   |
| Model execution     | PyTorch               |
| Data processing     | NumPy, pandas         |
| Backend             | FastAPI               |
| Demo UI             | Streamlit             |
| Testing             | pytest                |
| Optional generation | Ollama or hosted LLM  |
| Optional deployment | Docker                |

---

# Engineering Decisions

## Retrieval-first development

The generator is intentionally postponed until retrieval quality has been measured.

## Offline indexing

Passage embeddings are generated once and stored in a FAISS index. Query search does not re-encode the complete corpus.

## Shared corpus

BM25, dense retrieval, and ColBERTv2 use the same source documents and passage preparation pipeline where technically appropriate.

## Reproducibility

Dataset preparation, indexing, retrieval, comparison, and evaluation are script-based.

## Honest experiment reporting

The README separates:

* Implemented components
* Planned components
* Qualitative observations
* Measured results
* Metrics that have not yet been calculated

## Modular architecture

Data ingestion, retrieval, evaluation, API, UI, and RAG generation are implemented as separate modules.

---

# Current Project Status

* [x] Git repository initialization
* [x] Python virtual environment
* [x] SciFact dataset loading
* [x] Document normalization
* [x] Query extraction
* [x] Qrel extraction
* [x] Passage chunking
* [x] BM25 retrieval baseline
* [x] Dense passage embedding generation
* [x] FAISS vector index
* [x] Dense retrieval search
* [x] BM25 and dense comparison script
* [x] Initial qualitative comparison
* [ ] Automated ingestion tests
* [ ] Automated BM25 tests
* [ ] Automated dense retrieval tests
* [ ] Document-level retrieval evaluation
* [ ] Recall@5
* [ ] Recall@10
* [ ] MRR@10
* [ ] nDCG@10
* [ ] ColBERTv2-compatible collection
* [ ] ColBERTv2 index
* [ ] Late-interaction retrieval
* [ ] Three-method comparison
* [ ] FastAPI backend
* [ ] Streamlit interface
* [ ] Optional RAG answer generation
* [ ] Docker configuration
* [ ] GitHub Actions
* [ ] Final experimental report

---

# Development Roadmap

## Phase 1 — Dataset and preprocessing

* Load SciFact
* Normalize documents
* Extract queries
* Extract qrels
* Generate passages
* Preserve passage-document mappings

**Status: Completed**

## Phase 2 — BM25 baseline

* Build BM25 passage index
* Implement top-k search
* Return scores and metadata
* Test representative queries

**Status: Completed**

## Phase 3 — Dense retrieval

* Select embedding model
* Generate normalized passage vectors
* Build FAISS index
* Save index metadata
* Implement query encoding
* Implement nearest-neighbor search
* Compare with BM25

**Status: Completed**

## Phase 4 — Evaluation

* Load all test queries
* Load qrels
* Retrieve top results
* Convert passage rankings to document rankings
* Deduplicate documents
* Calculate Recall@5
* Calculate Recall@10
* Calculate MRR@10
* Calculate nDCG@10
* Export Markdown and JSON result reports

**Status: Next phase**

## Phase 5 — ColBERTv2

* Prepare collection
* Configure environment
* Load pretrained checkpoint
* Build compressed multi-vector index
* Run late-interaction retrieval
* Compare all methods
* Add ColBERT metrics

**Status: Planned**

## Phase 6 — FastAPI

* Add retriever registry
* Load models once at startup
* Implement search endpoint
* Add request validation
* Add health endpoint
* Add OpenAPI documentation

**Status: Planned**

## Phase 7 — Streamlit

* Add query input
* Add method selection
* Add top-k selection
* Add result cards
* Add comparison view
* Display scores and sources

**Status: Planned**

## Phase 8 — Optional RAG

* Retrieve top passages
* Construct grounded prompt
* Connect optional LLM backend
* Generate source-based answer
* Display cited passages

**Status: Planned**

---

# Limitations

The current project has several limitations:

* SciFact is small compared with production-scale corpora.
* Documents are primarily abstracts rather than complete papers.
* BM25 currently uses a simple regex tokenizer.
* Chunking is word-based rather than sentence-aware.
* Dense retrieval quality depends on the selected embedding model.
* A single-vector embedding may lose fine-grained information.
* Dense retrieval may experience semantic drift.
* Passage-level duplicates must be handled during evaluation.
* ColBERTv2 may require a separate GPU-compatible environment.
* Retrieval relevance does not guarantee answer factuality.
* RAG generation requires separate faithfulness evaluation.

These limitations are reported explicitly and may be used as directions for future experiments.

---

# Future Improvements

Potential extensions include:

* Hybrid BM25 and dense retrieval
* Reciprocal Rank Fusion
* Cross-encoder reranking
* Sentence-aware chunking
* Chunk-size experiments
* Embedding-model comparison
* Query expansion
* Full-paper PDF ingestion
* Additional BEIR datasets
* NFCorpus evaluation
* Scientific-domain embedding models
* Search latency benchmarking
* Memory and index-size benchmarking
* Cached model loading
* Dockerized services
* GitHub Actions
* Source citation extraction
* RAG faithfulness metrics
* Experiment tracking

A potential hybrid architecture:

```text
BM25 candidates
        +
Dense candidates
        │
        ▼
Reciprocal Rank Fusion
        │
        ▼
Optional cross-encoder reranking
        │
        ▼
Final result list
```

---

# Research Background

The project is primarily inspired by:

1. **ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT**
2. **ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction**
3. **BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models**
4. **Fact or Fiction: Verifying Scientific Claims**

These works provide the research foundation for the dataset, retrieval architectures, and evaluation approach used in this repository.

---

# Author's Note

I created this project to study modern information retrieval through implementation rather than only theoretical reading.

The project begins with a classical lexical baseline, continues with single-vector semantic retrieval, and then progresses toward token-level late interaction with ColBERTv2.

My objective is not simply to use pretrained models as black-box APIs. I want to understand the complete retrieval workflow:

```text
Raw documents
    → preprocessing
    → passages
    → indexing
    → query encoding
    → similarity scoring
    → ranking
    → evaluation
    → API
    → user interface
    → optional RAG generation
```

## Retrieval Evaluation Results

The retrieval methods were evaluated on the SciFact test split using document-level relevance judgments.

Although the system retrieves passages, SciFact relevance labels are defined at document level. Therefore, retrieved passages are mapped back to their parent `doc_id`, duplicate documents are removed, and the resulting document ranking is evaluated against qrels.

The evaluation uses:

```text
Candidate passage depth: 50
Evaluation document depth: 10
Evaluated queries: 300
```

| Method | Recall@5 | Recall@10 | MRR@10 | nDCG@10 | Notes                         |
| ------ | -------: | --------: | -----: | ------: | ----------------------------- |
| BM25   |   0.7112 |    0.7640 | 0.6162 |  0.6471 | Lexical keyword baseline      |
| DENSE  |   0.7127 |    0.7981 | 0.6062 |  0.6480 | Sentence Transformers + FAISS |

### Evaluation Analysis

The dense retrieval baseline achieves a higher `Recall@10` than BM25, which means it retrieves more relevant documents when the system is allowed to inspect the top ten results.

BM25 achieves a slightly higher `MRR@10`, showing that it remains very competitive at ranking the first relevant document near the top. This is expected in a scientific retrieval dataset because lexical overlap, rare terminology, and exact scientific expressions are highly informative.

The `nDCG@10` scores are very close, with dense retrieval slightly ahead. This suggests that both methods produce comparable ranking quality on this dataset, but they may retrieve relevant evidence through different signals.

Overall, the result supports the motivation for comparing multiple retrieval architectures instead of assuming that a neural method will always dominate a lexical baseline.

The next step is to add ColBERTv2, which may preserve more token-level matching information than single-vector dense retrieval.
