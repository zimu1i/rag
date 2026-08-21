# LegalMind

A retrieval-augmented question answering system over the **Canada Business
Corporations Act** (R.S.C., 1985, c. C-44), built without a RAG framework.

Retrieval, ranking, and chunking are hand-written against NumPy and the OpenAI
API rather than delegated to LangChain or LlamaIndex. The goal is to be able to
explain and change every ranking decision, which is difficult when scoring lives
inside a framework abstraction.

## Status

This project is **in progress**. The table below is the honest current state,
not the intended design.

| Component | Status |
| --- | --- |
| PDF ingestion (PyMuPDF) | Working — but English and French text are interleaved, see below |
| Chunking (NLTK sentence packing) | Working — size limit is not reliably enforced |
| Embedding + vector cache | Working — sequential, unbatched, no resume |
| Semantic retrieval (cosine) | Working |
| Generation (GPT-4o-mini) | Working |
| BM25 keyword retrieval | **Not built** |
| Hybrid merge / ranking | **Not built** |
| Groundedness / low-support refusal | **Not built** |
| Evaluation set + retrieval metrics | **Not built** |

### Known issues

- **The source PDF is bilingual.** C-44 is published as a two-column document
  with English on the left and French on the right. The current extraction
  flattens both columns into one string, so roughly half the search index is
  French text. This degrades every retrieval and is the next thing being fixed.
- **No citations.** Chunks carry no section or page metadata, so answers cannot
  cite a provision (e.g. "s. 102(1)").
- **First run is slow.** With no embedding cache present, the pipeline makes
  ~2,500 sequential embedding requests (roughly 12–25 minutes). There is no
  batching, retry, or resume, so an interrupted run restarts from zero.

## Setup

Requires Python 3.13 and an OpenAI API key.

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

> **Note:** the `venv/` directory currently checked out on the original
> development machine was created at a different path and its `pip` is broken.
> Create a fresh virtual environment as above.

Set your API key in the environment. It is never read from a file:

```bash
export OPENAI_API_KEY="sk-..."
```

## Usage

```bash
python rag.py
```

On first run this builds `embeddings.json` (~75 MB, gitignored). Subsequent runs
load the cache. Ask questions at the prompt; type `quit` to exit.

## Running tests

```bash
pip install -r requirements-dev.txt
pytest
```

Tests mock the OpenAI client, so they are deterministic, offline, and free —
no API key or network access required.

## Data

`C-44.pdf` is the consolidated Act as published by the Department of Justice
Canada, current to May 26, 2026. It is included so the project is runnable
without an external download.
