# LegalMind

RAG over the Canada Business Corporations Act. Portfolio project for ML/AI internship
applications — I must be able to explain every design decision in an interview.

## Deliberate choices (do not "improve" these without asking)
- Hand-built retrieval and ranking, no LangChain/LlamaIndex — the point is control
  over relevance and understanding the mechanics.
- Hybrid BM25 + semantic retrieval, merged with reciprocal rank fusion. Semantic
  handles paraphrase; BM25 contributes rare-term matches that embeddings dilute.
- Section-number queries are answered by **structured lookup** against the section
  metadata attached during chunking, not by similarity. This was not the original
  plan — see below.
- For legal information a confident wrong answer is worse than no answer. What
  enforces that is (a) a prompt restricting the model to the retrieved excerpts,
  and (b) deterministic validation that every provision the answer cites was
  actually retrieved.

## Measured findings that overrode the original design
Both of these came from `evaluation.py`, and both contradict what this file used
to say. Keep them in mind before "fixing" the code back.

- **BM25 did not rescue section-number queries, and fusion made them worse.**
  hit@3 on that category: semantic 0.00, BM25 alone 0.33, fused 0.17, structured
  lookup 1.00. Semantic retrieval returns the same generic cross-reference chunks
  for every numbered query and fusion reinforces them. "Section 122" turned out to
  be a lookup, not a similarity problem.
- **A groundedness *threshold* could not be built.** Five candidate support
  signals were tested against 11 out-of-scope questions; every one overlaps the
  answerable set. The cause is structural — an Ontario Business Corporations Act
  question is semantically near-identical to a CBCA one, and retrieval similarity
  measures topical closeness, not jurisdictional applicability. The best signal is
  also anti-correlated with answer quality on section-number queries. No threshold
  is shipped; do not add one without new evidence.

## Conventions
- pytest for tests; mock the model client so tests are deterministic and free
- Small commits with clear messages
- No `Co-Authored-By` AI trailers in commit messages
- No secrets in the repo, ever

## Current state
**Check the status table in README.md before assuming any component works.**

As of 2026-08-26: ingestion, cleanup, chunking, embedding, semantic retrieval,
BM25, rank fusion, structured citation lookup, generation and the evaluation
harness are built and tested. Retrieval measures hit@3 0.96 / MRR 0.82 on 26
labelled questions.

Measured on 26 answerable + 11 out-of-scope questions: 11/11 declined, 26/26
answered, 0 citations absent from the excerpts.

Not yet done:
- No vocabulary/synonym layer. A question using "shareholder" does not reach a
  provision written in terms of "complainant".
- One evaluation question still fails (s. 241, the oppression remedy).

## Working style
- One component at a time, then stop for review
- Explain the reasoning behind non-obvious decisions as the work happens
- Ask before adding any dependency
- Ask before any refactor touching more than one file; don't rewrite working code
