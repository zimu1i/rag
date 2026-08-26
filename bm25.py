"""
BM25 keyword retrieval over the Act's provisions.

Embeddings are blind to exact tokens. Measured on this corpus, the section
number "122" appears in only 4 of 1,125 chunks -- it is among the most
discriminative terms available -- yet semantic retrieval scores 0.00 hit@3 on
questions that name a section by number, returning the same two generic
cross-reference chunks ("Section 242 applies to an application under this
section") no matter which number is asked about. It has learned the *shape* of a
statutory cross-reference and discarded the digits.

BM25 weights a term by how rare it is, which is exactly the signal being thrown
away. The same mechanism helps where a distinctive word is buried in a long
provision: "oppressive" (3 chunks) and "unfairly" (5 chunks) sit 70% of the way
through s. 241(2), diluted in the averaged embedding but full-strength here.

Implemented directly rather than through a library so the ranking stays
inspectable -- the point of the project is to be able to explain why a chunk
scored what it did.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from dataclasses import dataclass, field

# Controls how quickly repeated occurrences of a term stop adding score. A
# provision that says "director" six times is not six times more about directors
# than one that says it once, and this saturates that. 1.5 is the standard
# default; it is deliberately untuned, since fitting it to 26 eval questions
# would be overfitting rather than tuning.
K1 = 1.5

# Controls length normalisation, from 0 (ignore length) to 1 (fully normalise).
# Chunks here range from ~40 to ~1,750 characters, so some correction is needed
# or long provisions win by sheer surface area. 0.75 is the standard default.
B = 0.75

# Digits are kept because section numbers are the whole point. Hyphenated
# compounds are kept whole because cleanup.py deliberately preserved them:
# splitting "by-laws" into "by" and "laws" would discard the term of art it
# worked to protect. Note that a decimal section such as "2.1" tokenises to
# "2" and "1"; that applies equally to documents and queries, so it stays
# consistent on both sides.
_TOKEN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def indexed_text(chunk) -> str:
    """The text BM25 searches for a chunk: citation, heading, then body.

    The citation is included because only a section's *first* subsection carries
    its own number in the body. s. 122(2) reads "(2) Every director and officer
    shall comply..." -- the digits 122 appear nowhere in it. Without this, a
    search for "section 122" could never reach 122(2) or 122(3), even though
    that is exactly how people cite provisions.
    """
    parts = [chunk.citation]
    if chunk.heading:
        parts.append(chunk.heading)
    parts.append(chunk.text)
    return " ".join(parts)


@dataclass
class BM25Index:
    """An inverted index over chunks, scored with Okapi BM25."""

    chunks: list = field(default_factory=list)
    postings: dict[str, list[tuple[int, int]]] = field(default_factory=dict)
    idf: dict[str, float] = field(default_factory=dict)
    lengths: list[int] = field(default_factory=list)
    average_length: float = 0.0

    def score_terms(self, query: str) -> dict[int, float]:
        """Accumulate BM25 scores per document for one query.

        Only documents containing a query term are touched, which is the reason
        for the inverted index: scoring walks a handful of postings lists rather
        than all 1,125 chunks.
        """
        scores: dict[int, float] = defaultdict(float)
        for term in tokenize(query):
            postings = self.postings.get(term)
            if not postings:
                continue
            idf = self.idf[term]
            for document, frequency in postings:
                length_norm = 1 - B + B * self.lengths[document] / self.average_length
                scores[document] += idf * frequency * (K1 + 1) / (
                    frequency + K1 * length_norm
                )
        return scores

    def search(self, query: str, top_k: int = 5) -> list[tuple[float, object]]:
        """Return the top_k (score, chunk) pairs for a query."""
        scores = self.score_terms(query)
        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        return [(score, self.chunks[document]) for document, score in ranked[:top_k]]


def build_index(chunks) -> BM25Index:
    """Build an inverted index with BM25 term weights."""
    postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
    lengths: list[int] = []

    for position, chunk in enumerate(chunks):
        tokens = tokenize(indexed_text(chunk))
        lengths.append(len(tokens))
        frequencies: dict[str, int] = defaultdict(int)
        for token in tokens:
            frequencies[token] += 1
        for term, frequency in frequencies.items():
            postings[term].append((position, frequency))

    total = len(chunks)
    # Robertson-Sparck-Jones IDF with the +1 smoothing, which keeps the value
    # non-negative. Without it, a term appearing in more than half the corpus
    # scores negatively and actively pushes matching documents down.
    idf = {
        term: math.log(1 + (total - len(entries) + 0.5) / (len(entries) + 0.5))
        for term, entries in postings.items()
    }

    return BM25Index(
        chunks=list(chunks),
        postings=dict(postings),
        idf=idf,
        lengths=lengths,
        average_length=(sum(lengths) / total) if total else 0.0,
    )
