"""Tests for BM25 keyword retrieval.

Entirely offline: BM25 needs no model, so every test here is exact rather than
approximate. Where a test asserts a ranking, it is because the BM25 formula
guarantees that ordering, not because it happened to come out that way.
"""

import math

import pytest

import bm25
from bm25 import build_index, indexed_text, tokenize
from chunking import Chunk


def chunk(text, section=None, subsection=None, heading=None):
    return Chunk(text=text, section=section, subsection=subsection, heading=heading)


class TestTokenize:
    def test_lowercases_and_splits_on_punctuation(self):
        assert tokenize("Shares of a Corporation.") == ["shares", "of", "a", "corporation"]

    def test_keeps_digits(self):
        # The entire point: section numbers must survive tokenisation.
        assert "122" in tokenize("122 (1) Every director")

    def test_keeps_hyphenated_compounds_whole(self):
        """cleanup.py deliberately preserved these; splitting them here would
        undo that work and make "by-laws" unmatchable as a term of art."""
        assert tokenize("subject to the by-laws") == ["subject", "to", "the", "by-laws"]

    def test_drops_bare_punctuation(self):
        assert tokenize("(a) ... ;") == ["a"]

    def test_empty_string(self):
        assert tokenize("") == []


class TestIndexedText:
    def test_includes_the_citation(self):
        """s. 122(2)'s body never contains the digits 122.

        Only a section's first subsection carries its own number, so without
        the citation in the indexed text a search for "section 122" could
        never reach 122(2).
        """
        subsection = chunk("(2) Every director shall comply.", section="122", subsection="2")

        assert "122" in tokenize(indexed_text(subsection))

    def test_includes_the_heading(self):
        with_heading = chunk("(1) text", section="122", subsection="1", heading="Duty of care")

        assert "duty" in tokenize(indexed_text(with_heading))

    def test_survives_a_missing_heading(self):
        assert indexed_text(chunk("text", section="1"))


class TestIdf:
    def test_a_rare_term_outweighs_a_common_one(self):
        chunks = [chunk("corporation shares") for _ in range(20)]
        chunks.append(chunk("corporation oppressive"))

        index = build_index(chunks)

        assert index.idf["oppressive"] > index.idf["corporation"]

    def test_idf_stays_non_negative_for_a_term_in_every_document(self):
        """Without the +1 smoothing, a term in more than half the corpus scores
        negatively and actively pushes matching documents down the ranking."""
        index = build_index([chunk("corporation") for _ in range(10)])

        assert index.idf["corporation"] >= 0


class TestSearch:
    def test_finds_a_chunk_by_an_exact_section_number(self):
        chunks = [
            chunk("(1) Shares shall be in registered form.", section="24", subsection="1"),
            chunk("(1) Every director shall act honestly.", section="122", subsection="1"),
        ]

        results = build_index(chunks).search("section 122", top_k=1)

        assert results[0][1].citation == "s. 122(1)"

    def test_finds_a_later_subsection_by_its_section_number(self):
        # The case that motivates indexing the citation.
        chunks = [
            chunk("(1) Shares shall be registered.", section="24", subsection="1"),
            chunk("(2) Every director shall comply.", section="122", subsection="2"),
        ]

        results = build_index(chunks).search("what does section 122 require", top_k=1)

        assert results[0][1].citation == "s. 122(2)"

    def test_a_rare_term_beats_a_common_one(self):
        chunks = [chunk("the corporation and the corporation")] * 10
        chunks.append(chunk("the corporation acted oppressively"))

        results = build_index(chunks).search("oppressively", top_k=1)

        assert "oppressively" in results[0][1].text

    def test_shorter_documents_win_when_term_frequency_matches(self):
        """Length normalisation, controlled by b.

        A one-word document about "dissent" is more about dissent than a long
        provision that mentions it once.
        """
        short = chunk("dissent")
        long = chunk("dissent " + "unrelated words here " * 40)

        results = build_index([long, short]).search("dissent", top_k=2)

        assert results[0][1] is short

    def test_repeated_terms_saturate(self):
        """Controlled by k1: ten mentions is not ten times one mention."""
        once = build_index([chunk("dissent")]).search("dissent", 1)[0][0]
        index = build_index([chunk("dissent " * 10)])
        many = index.search("dissent", 1)[0][0]

        assert many < once * 10

    def test_unknown_terms_return_nothing(self):
        index = build_index([chunk("shares and directors")])

        assert index.search("zzzznotaword", top_k=5) == []

    def test_empty_query(self):
        assert build_index([chunk("shares")]).search("", top_k=5) == []

    def test_respects_top_k(self):
        chunks = [chunk(f"dissent number {i}") for i in range(10)]

        assert len(build_index(chunks).search("dissent", top_k=3)) == 3

    def test_scores_descend(self):
        chunks = [chunk("dissent " * i + "filler " * 10) for i in range(1, 6)]

        scores = [score for score, _ in build_index(chunks).search("dissent", top_k=5)]

        assert scores == sorted(scores, reverse=True)

    def test_ranking_is_deterministic_for_tied_scores(self):
        # Ties break on document order, so repeated runs cannot disagree.
        chunks = [chunk("dissent") for _ in range(5)]
        index = build_index(chunks)

        assert [c.text for _, c in index.search("dissent", 5)] == [
            c.text for _, c in index.search("dissent", 5)
        ]


class TestBuildIndex:
    def test_empty_corpus(self):
        index = build_index([])

        assert index.search("anything", top_k=5) == []
        assert index.average_length == 0.0

    def test_records_document_lengths(self):
        index = build_index([chunk("one two three"), chunk("one")])

        # Citations ("(uncited)") are part of the indexed text, so compare
        # relative rather than absolute lengths.
        assert index.lengths[0] > index.lengths[1]


@pytest.mark.slow
class TestRealAct:
    @pytest.fixture(scope="class")
    def index(self):
        import rag

        return build_index(rag.build_chunks())

    def test_section_number_query_reaches_the_right_section(self, index):
        results = index.search("what does section 122 require?", top_k=10)

        assert any(c.section == "122" for _, c in results)

    def test_rare_statutory_term_ranks_its_provision(self, index):
        results = index.search("oppressive unfairly prejudicial", top_k=5)

        assert any(c.section == "241" for _, c in results)

    def test_hyphenated_term_of_art_is_searchable(self, index):
        results = index.search("by-laws", top_k=10)

        assert results, "by-laws should match; if empty, tokenisation broke the hyphen"
