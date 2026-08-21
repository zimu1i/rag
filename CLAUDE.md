# LegalMind

RAG over the Canada Business Corporations Act. Portfolio project for ML/AI internship
applications — I must be able to explain every design decision in an interview.

## Deliberate choices (do not "improve" these without asking)
- Hand-built retrieval and ranking, no LangChain/LlamaIndex — the point is control
  over relevance and understanding the mechanics.
- Hybrid BM25 + semantic retrieval: BM25 catches exact statutory terms and section
  numbers that embeddings blur; semantic catches paraphrase.
- Groundedness check declines low-support queries — for legal information a confident
  wrong answer is worse than no answer.

## Conventions
- pytest for tests; mock the model client so tests are deterministic and free
- Small commits with clear messages
- No `Co-Authored-By` AI trailers in commit messages
- No secrets in the repo, ever

## Current state
The section above describes the intended design, not what exists today. Several
components are still unbuilt — **check the status table in README.md before
assuming any component works.** As of 2026-08-21, BM25, hybrid merge, and the
groundedness check are design intent with no implementation.

## Working style
- One component at a time, then stop for review
- Explain the reasoning behind non-obvious decisions as the work happens
- Ask before adding any dependency
- Ask before any refactor touching more than one file; don't rewrite working code
