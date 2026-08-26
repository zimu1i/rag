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
| Citation validation | Working — every answer's citations are audited against what was retrieved |
| Low-support refusal threshold | **Tested and rejected** — no signal separates in from out of scope |
| Evaluation set + retrieval metrics | Working — 26 labelled questions in five categories |

The index holds 1,125 chunks, one per provision, each labelled with its section
(e.g. `s. 122(1)`), marginal note, Part and source pages.

### Known issues

- **One evaluation question still fails.** "What can I do if the company is
  treating me unfairly as a minority shareholder?" does not retrieve s. 241.
  The distinctive words ("oppressive", "unfairly prejudicial") sit 70% of the
  way through a 675-character provision, diluted in its embedding.
- **Nothing detects a jurisdictional error.** If the system answered a question
  about Delaware or Ontario corporate law using correctly-cited CBCA provisions,
  every citation would be real and genuinely retrieved, and no guard would fire.
  This has not been observed, but it is not covered.
- **Nothing checks that an answer faithfully paraphrases what it cites.** A model
  can cite s. 122(1) accurately and still misdescribe it. Catching that needs a
  verifier pass, which is not built.
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

## Refusing to answer

The original design called for a groundedness check that declined "low-support"
queries. **That threshold could not be built, and the reason is worth recording.**

`evaluation.py support` scores every question in `eval_set.json` against the 11
out-of-scope questions in `eval_negatives.json`. Five candidate support signals
were tested; all of them overlap:

| signal | lowest answerable | highest out-of-scope | separation |
| --- | --- | --- | --- |
| max cosine | 0.487 | 0.633 | −0.147 |
| top-1 minus top-5 mean | 0.009 | 0.120 | −0.111 |
| peak vs corpus mean | 0.160 | 0.305 | −0.145 |
| mean of top 3 | 0.465 | 0.588 | −0.123 |
| top BM25 score | 6.71 | 13.89 | −7.18 |

The cause is structural, not a matter of tuning. "What are the requirements to
incorporate under the *Ontario* Business Corporations Act?" is semantically
almost identical to a legitimate CBCA question — retrieval similarity measures
topical closeness, not which statute governs. Worse, the best signal is
*anti-correlated* with answer quality on the one category the system handles
perfectly: section-number queries score 0.354–0.485, the lowest of any answerable
group, because a citation query is lexically unlike the provision it names.

So no threshold ships. What is relied on instead:

1. **A prompt restricting the model to the retrieved excerpts.** Measured with
   `evaluation.py refusal`: 11 of 11 out-of-scope questions were declined,
   including near-misses about Delaware, Ontario, securities law and bankruptcy.
2. **Deterministic citation validation** (`groundedness.py`): every provision an
   answer cites must appear in what was retrieved.

Caveats, because 11/11 is easy to over-read: the negative set is small and was
not written adversarially, and neither guard detects a jurisdictional error or an
unfaithful paraphrase of a correctly-cited provision.

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
