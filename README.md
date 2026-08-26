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
| PDF ingestion (PyMuPDF) | Working — separates the bilingual columns geometrically |
| Text cleanup | Working — strips boilerplate and front matter, repairs hyphenation |
| Chunking | Working — one chunk per provision, carrying its citation |
| Embedding + vector cache | Working — batched, and invalidated when chunking changes |
| Semantic retrieval (cosine) | Working |
| Generation (GPT-4o-mini) | Working — answers cite the provisions they rely on |
| BM25 keyword retrieval | Working — hand-built, indexes citations and headings |
| Hybrid merge / ranking | Working — reciprocal rank fusion + structured citation lookup |
| Groundedness / low-support refusal | **Not built** |
| Evaluation set + retrieval metrics | Working — 26 labelled questions in five categories |

The index holds 1,125 chunks, one per provision, each labelled with its section
(e.g. `s. 122(1)`), marginal note, Part and source pages.

### Known issues

- **`rag.py` still uses semantic retrieval only.** Hybrid retrieval is built,
  tested and measured, but the interactive question loop has not been switched
  over to it yet.
- **One evaluation question still fails.** "What can I do if the company is
  treating me unfairly as a minority shareholder?" does not retrieve s. 241.
  The distinctive words ("oppressive", "unfairly prejudicial") sit 70% of the
  way through a 675-character provision, diluted in its embedding.
- **Nothing verifies groundedness.** The model is instructed to answer only from
  the retrieved excerpts and to decline otherwise, but no code checks whether it
  did. Until that exists, treat answers as unverified.
- **A provision's substance can be split from its hook.** Chunking follows the
  Act's subsections, but a question is usually about a *section*. s. 241(1) says
  only that "a complainant may apply to a court for an order under this section";
  the grounds a reader is looking for — "oppressive", "unfairly prejudicial" —
  are in s. 241(2). Retrieving one subsection therefore does not guarantee the
  neighbouring subsection that explains it. Whether retrieval should expand to
  sibling subsections is an open design question, deferred until hybrid
  retrieval is measured.
- **About 1% of chunks exceed the size budget.** These are single provisions
  with no internal list to split at; they are left whole rather than cut
  mid-sentence.
- **The French column is extracted but not indexed.** Only the English text is
  embedded.

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
./venv/bin/python rag.py
```

Calling the venv's interpreter directly is deliberate: if you have Anaconda or
another Python on your `PATH`, a bare `python rag.py` will pick that one up and
fail with `ModuleNotFoundError: No module named 'openai'`.

On first run this embeds the Act and writes `embeddings.json` (gitignored) —
about 107,000 tokens, roughly one minute and well under a cent. Subsequent runs
load the cache. The cache records a fingerprint of the chunks it was built from,
so changing the chunking automatically triggers a rebuild rather than silently
pairing chunks with the wrong vectors.

Ask questions at the prompt; type `quit` to exit.

## Measuring retrieval

```bash
./venv/bin/python evaluation.py
```

Runs 26 labelled questions from `eval_set.json` and reports recall and MRR per
question category. Categories are reported separately on purpose: retrieval that
is strong on paraphrased questions and blind to section numbers scores
respectably overall while failing the query a lawyer is most likely to type.

`./venv/bin/python evaluation.py compare` runs the ablation:

| retriever | hit@1 | hit@3 | hit@5 | MRR |
| --- | --- | --- | --- | --- |
| semantic only | 0.54 | 0.65 | 0.73 | 0.61 |
| BM25 only | 0.15 | 0.58 | 0.65 | 0.37 |
| rank fusion | 0.46 | 0.77 | 0.77 | 0.61 |
| **+ structured lookup** | **0.69** | **0.96** | **0.96** | **0.82** |

Per category, for the full system:

| category | n | hit@1 | hit@3 | MRR |
| --- | --- | --- | --- | --- |
| definition | 4 | 1.00 | 1.00 | 1.00 |
| multi_provision | 4 | 0.50 | 1.00 | 0.75 |
| paraphrase | 7 | 0.43 | 0.86 | 0.62 |
| section_number | 6 | 1.00 | 1.00 | 1.00 |
| statutory_term | 5 | 0.60 | 1.00 | 0.80 |

Two results worth reading carefully:

- **Fusing semantic and BM25 made section-number queries worse than BM25 alone**
  (0.33 → 0.17 hit@3). Semantic retrieval returns the same generic
  cross-reference chunks for every numbered query, and fusion reinforces them.
  What fixed those queries was not a better scorer but a lookup: "section 122"
  is answered from the section metadata attached during chunking.
- **Fusion trades top-1 precision for top-3 recall.** Paraphrase hit@1 fell from
  0.57 to 0.43 while hit@3 rose from 0.57 to 0.86.

### Caveats on these numbers

- 26 questions is a small set. One case moves a category by 0.14–0.25.
- Fusion parameters were chosen by sweeping against this same set, so the
  reported figures are optimistic. RRF is kept at its standard k=60 with equal
  weights rather than the best-scoring configuration, to limit that. A proper
  dev/test split would be better and needs a larger question set.

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
