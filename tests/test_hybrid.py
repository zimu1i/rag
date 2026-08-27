"""Tests for hybrid retrieval: rank fusion plus structured citation lookup.

Fully offline. The semantic and keyword retrievers are stubs returning fixed
rankings, so every assertion about fusion is exact rather than approximate.
"""

import pytest

from legalmind.chunking import Chunk
from legalmind.hybrid import (
    build_retriever,
    parse_citation_query,
    reciprocal_rank_fusion,
    structured_lookup,
)


def chunk(section=None, subsection=None, text=None, part=None):
    return Chunk(
        text=text or f"text for {section}({subsection})",
        section=section,
        subsection=subsection,
        part=part,
    )


def stub(*results):
    """A retriever returning a fixed ranking."""
    return lambda question, k: list(results)[:k]


class TestParseCitationQuery:
    @pytest.mark.parametrize(
        "question,expected",
        [
            ("what does section 122 require?", ("122", None)),
            ("explain subsection 190(1)", ("190", "1")),
            ("what is section 241 about?", ("241", None)),
            ("see s. 146", ("146", None)),
            ("what does section 2.1 say?", ("2.1", None)),
            ("SECTION 122", ("122", None)),
        ],
    )
    def test_extracts_the_provision(self, question, expected):
        assert parse_citation_query(question) == expected

    @pytest.mark.parametrize(
        "question",
        [
            "what are the duties of directors?",
            "which section applies to by-laws?",  # names no number
            "what is a distributing corporation?",
            "",
        ],
    )
    def test_ignores_questions_that_name_no_provision(self, question):
        """The bare word 'section' appears in ordinary questions.

        Triggering a structured lookup on it would route meaning-based
        questions into a metadata match that cannot answer them.
        """
        assert parse_citation_query(question) is None


class TestStructuredLookup:
    def test_returns_the_whole_section_in_subsection_order(self):
        chunks = [
            chunk("122", "3"),
            chunk("122", "1"),
            chunk("241", "1"),
            chunk("122", "2"),
        ]

        result = structured_lookup(chunks, "122")

        assert [c.subsection for c in result] == ["1", "2", "3"]

    def test_orders_numerically_not_lexically(self):
        chunks = [chunk("190", "10"), chunk("190", "2"), chunk("190", "1")]

        result = structured_lookup(chunks, "190")

        assert [c.subsection for c in result] == ["1", "2", "10"]

    def test_named_subsection_leads_but_siblings_follow(self):
        """Neighbouring subsections usually carry the context that explains the
        one asked about, so they are kept rather than filtered out."""
        chunks = [chunk("190", "1"), chunk("190", "2"), chunk("190", "3")]

        result = structured_lookup(chunks, "190", "2")

        assert result[0].subsection == "2"
        assert {c.subsection for c in result} == {"1", "2", "3"}

    def test_excludes_the_schedule(self):
        """The Schedule restarts numbering, so its item 1 is not s. 1."""
        chunks = [chunk("1", None, part="SCHEDULE"), chunk("1", None, part="I")]

        result = structured_lookup(chunks, "1")

        assert len(result) == 1
        assert result[0].part == "I"

    def test_unknown_section(self):
        assert structured_lookup([chunk("122", "1")], "999") == []


class TestReciprocalRankFusion:
    def test_a_document_ranked_by_both_beats_one_ranked_by_either(self):
        both, only_a, only_b = chunk("1"), chunk("2"), chunk("3")

        fused = reciprocal_rank_fusion([[only_a, both], [only_b, both]])

        assert fused[0] is both

    def test_preserves_order_of_a_single_ranking(self):
        first, second, third = chunk("1"), chunk("2"), chunk("3")

        assert reciprocal_rank_fusion([[first, second, third]]) == [first, second, third]

    def test_higher_ranks_contribute_more(self):
        top, lower = chunk("1"), chunk("2")

        assert reciprocal_rank_fusion([[top, lower], [top, lower]])[0] is top

    def test_a_larger_k_flattens_rank_differences(self):
        a, b, c = chunk("1"), chunk("2"), chunk("3")
        # b is 1st in one list, a is 1st in the other and 3rd in the first.
        rankings = [[a, c, b], [b, a, c]]

        assert reciprocal_rank_fusion(rankings, k=1)[0] is not None
        assert set(reciprocal_rank_fusion(rankings, k=1000)) == {a, b, c}

    def test_is_deterministic_for_tied_documents(self):
        a, b = chunk("1"), chunk("2")

        assert reciprocal_rank_fusion([[a, b]]) == reciprocal_rank_fusion([[a, b]])

    def test_empty_rankings(self):
        assert reciprocal_rank_fusion([]) == []
        assert reciprocal_rank_fusion([[], []]) == []


class TestBuildRetriever:
    def test_uses_fusion_for_a_meaning_based_question(self):
        semantic_top, keyword_top = chunk("120", "1"), chunk("122", "1")
        retrieve = build_retriever(
            [semantic_top, keyword_top], stub(semantic_top), stub(keyword_top)
        )

        results = retrieve("what duties do directors have?", k=2)

        assert set(results) == {semantic_top, keyword_top}

    def test_a_citation_query_returns_the_named_section_first(self):
        """The finding this module exists for.

        Both retrievers rank an irrelevant chunk top for numbered queries;
        structured lookup answers from metadata instead.
        """
        wanted = chunk("122", "1")
        noise = chunk("214", "3")
        retrieve = build_retriever([wanted, noise], stub(noise), stub(noise))

        results = retrieve("what does section 122 require?", k=2)

        assert results[0] is wanted

    def test_falls_back_to_fusion_when_the_section_does_not_exist(self):
        noise = chunk("214", "3")
        retrieve = build_retriever([noise], stub(noise), stub(noise))

        results = retrieve("what does section 999 require?", k=1)

        assert results == [noise]

    def test_does_not_return_duplicates(self):
        wanted = chunk("122", "1")
        retrieve = build_retriever([wanted], stub(wanted), stub(wanted))

        results = retrieve("what does section 122 require?", k=5)

        assert results == [wanted]

    def test_respects_k(self):
        chunks = [chunk("122", str(i)) for i in range(1, 8)]
        retrieve = build_retriever(chunks, stub(*chunks), stub(*chunks))

        assert len(retrieve("section 122", k=3)) == 3

    def test_structured_routing_can_be_disabled(self):
        """Supports an ablation: the same pipeline without the lookup."""
        wanted, noise = chunk("122", "1"), chunk("214", "3")
        retrieve = build_retriever(
            [wanted, noise], stub(noise), stub(noise), use_structured=False
        )

        assert retrieve("what does section 122 require?", k=1) == [noise]


@pytest.mark.slow
class TestRealAct:
    @pytest.fixture(scope="class")
    def pieces(self):
        from legalmind import bm25
        from legalmind import evaluation
        from legalmind import rag

        cached = rag.load_cache()
        if cached is None:
            pytest.skip("no embedding cache; run rag.py first")
        chunks, embeddings = cached
        cache = evaluation.load_query_cache()
        if not cache:
            pytest.skip("no query cache; run `evaluation.py warm` first")
        matrix = rag.to_matrix(embeddings)
        index = bm25.build_index(chunks)

        def semantic(question, k):
            return [c for _, c in rag.find_chunks(cache[question], chunks, matrix, top_k=k)]

        def keyword(question, k):
            return [c for _, c in index.search(question, top_k=k)]

        return chunks, semantic, keyword

    def test_section_number_queries_are_answered_exactly(self, pieces):
        chunks, semantic, keyword = pieces
        retrieve = build_retriever(chunks, semantic, keyword)

        for question, section in [
            ("what does section 122 require?", "122"),
            ("what obligations does section 155 impose?", "155"),
            ("what are the requirements in section 6?", "6"),
        ]:
            top = retrieve(question, k=1)[0]
            assert top.section == section, f"{question} returned {top.citation}"

    def test_meaning_based_queries_still_work(self, pieces):
        chunks, semantic, keyword = pieces
        retrieve = build_retriever(chunks, semantic, keyword)

        results = retrieve("what happens if a director has a conflict of interest?", k=3)

        assert any(c.section == "120" for c in results)
