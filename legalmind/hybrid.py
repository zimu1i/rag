"""
Combine semantic, keyword and structured retrieval.

Measured on the 26-question evaluation set, each retriever fails somewhere the
others do not:

    retriever          hit@3   section_number   paraphrase
    semantic            0.65        0.00           0.57
    bm25                0.58        0.33           0.29
    rrf fusion          0.77        0.17           0.86
    + structured        0.96        1.00           0.86

Two findings shaped this module, and neither was what I expected going in.

1.  Fusing semantic and BM25 made section-number queries *worse* than BM25
    alone (0.33 -> 0.17). Semantic retrieval returns the same handful of generic
    cross-reference chunks for every numbered query, and because those chunks
    also score on BM25's side, fusion reinforces them and buries the provision
    actually being asked about.

2.  What fixed those queries was not a better scorer. "Section 122" is not a
    similarity problem -- it is a lookup. Chunks already carry the section number
    extracted during chunking, so a query naming a provision is answered
    directly from that metadata, and only questions that are genuinely about
    meaning are handed to the retrievers.

Fusion still earns its place: it took paraphrased questions from 0.57 to 0.86 by
rescuing provisions that BM25 ranked around 10th and semantic could not see at
all.
"""

from __future__ import annotations

import re
from collections import defaultdict

# Rank-fusion constant. A document's contribution is 1/(RRF_K + rank), so a
# larger value flattens the difference between ranks. 60 is the standard value
# from the original RRF paper and is deliberately left untuned: with only 26
# evaluation questions, fitting it here would be overfitting, not tuning.
RRF_K = 60

# How many candidates to pull from each retriever before fusing. BM25 ranks
# genuinely relevant provisions as deep as 28th, so a shallow pool discards the
# results fusion exists to rescue.
CANDIDATE_DEPTH = 100

# Matches a query naming a provision: "section 122", "subsection 190(1)",
# "s. 146". Requires a digit, so the bare word "section" -- common in ordinary
# questions -- never triggers a structured lookup.
_CITATION_QUERY = re.compile(
    r"\b(?:sub)?section\s+(\d+(?:\.\d+)?)(?:\s*\((\d+)\))?" r"|\bs\.\s*(\d+(?:\.\d+)?)",
    re.IGNORECASE,
)


def parse_citation_query(question: str) -> tuple[str, str | None] | None:
    """Extract (section, subsection) from a question naming a provision."""
    match = _CITATION_QUERY.search(question)
    if not match:
        return None
    section = match.group(1) or match.group(3)
    return section, match.group(2)


def _subsection_order(chunk) -> int:
    """Sort key placing (1) before (2) before (10)."""
    if not chunk.subsection:
        return 0
    try:
        return int(chunk.subsection)
    except ValueError:
        return 999


def structured_lookup(chunks, section: str, subsection: str | None = None) -> list:
    """Return every chunk of a named section, in reading order.

    The Schedule is excluded because it restarts numbering from 1, so its items
    would otherwise collide with sections of the Act itself -- Schedule item 1
    is not s. 1.

    When the question names a specific subsection, that one leads; the rest of
    the section follows, because neighbouring subsections usually carry the
    context that explains it.
    """
    matches = [c for c in chunks if c.section == section and c.part != "SCHEDULE"]
    if subsection is None:
        matches.sort(key=_subsection_order)
    else:
        matches.sort(key=lambda c: (c.subsection != subsection, _subsection_order(c)))
    return matches


def reciprocal_rank_fusion(rankings, k: int = RRF_K) -> list:
    """Merge ranked lists by summing 1/(k + rank) for each document.

    Rank-based rather than score-based on purpose: cosine similarity is bounded
    in [-1, 1] while BM25 is unbounded and its scale shifts with the query, so
    combining the raw numbers would let one retriever dominate for reasons that
    have nothing to do with relevance. Ranks are directly comparable, and there
    is no normalisation scheme to justify or tune.
    """
    scores: dict = defaultdict(float)
    order: dict = {}
    for ranking in rankings:
        for rank, chunk in enumerate(ranking, start=1):
            scores[chunk] += 1.0 / (k + rank)
            order.setdefault(chunk, len(order))
    # Ties break on first-seen order so results are stable across runs.
    return sorted(scores, key=lambda c: (-scores[c], order[c]))


def build_retriever(
    chunks,
    semantic,
    keyword,
    depth: int = CANDIDATE_DEPTH,
    use_structured: bool = True,
):
    """Assemble the full retriever.

    `semantic` and `keyword` are each callables taking (question, k) and
    returning ranked chunks, matching the contract the evaluation harness uses.
    Taking them as arguments rather than constructing them here keeps this
    module free of any dependency on the embedding cache or the API client,
    which is what lets it be tested with stubs.
    """

    def retrieve(question: str, k: int = 5) -> list:
        fused = reciprocal_rank_fusion(
            [semantic(question, depth), keyword(question, depth)]
        )

        parsed = parse_citation_query(question) if use_structured else None
        if parsed is None:
            return fused[:k]

        section, subsection = parsed
        ranked = structured_lookup(chunks, section, subsection) + fused

        seen, results = set(), []
        for chunk in ranked:
            if chunk not in seen:
                seen.add(chunk)
                results.append(chunk)
        return results[:k]

    return retrieve
