"""Tests for the retrieval evaluation harness.

The harness decides whether every future retrieval change looks like an
improvement, so its arithmetic needs to be right. These tests use a stub
retriever with hand-chosen rankings -- no embeddings, no API, no cache.
"""

import json
from pathlib import Path

import pytest

import evaluation
from chunking import Chunk
from evaluation import (
    EvalCase,
    Expectation,
    Target,
    evaluate,
    evaluate_case,
    format_report,
    load_eval_set,
    summarise,
)


def chunk(section, subsection=None, heading=None):
    return Chunk(text="text", section=section, subsection=subsection, heading=heading)


def case(question="q", category="paraphrase", expected=(Expectation.of("s. 122(1)"),)):
    return EvalCase(question=question, category=category, expected=tuple(expected))


class TestExpectation:
    def test_matches_on_citation(self):
        assert Expectation.of("s. 122(1)").matches(chunk("122", "1"))

    def test_rejects_a_different_citation(self):
        assert not Expectation.of("s. 122(1)").matches(chunk("123", "1"))

    def test_definition_requires_the_right_term(self):
        """~39 definition chunks share the citation s. 2(1).

        Matching on citation alone would score a question about "affiliate" as
        correct when the retriever returned the definition of "auditor".
        """
        affiliate = chunk("2", "1", heading="affiliate")
        auditor = chunk("2", "1", heading="auditor")
        expectation = Expectation.of("s. 2(1)", term="affiliate")

        assert expectation.matches(affiliate)
        assert not expectation.matches(auditor)

    def test_alternatives_are_satisfied_by_either_target(self):
        """A provision's hook and its substance are often separate subsections.

        s. 241(1) carries the right to apply; s. 241(2) carries the grounds.
        Either answers the question.
        """
        expectation = Expectation.either([Target("s. 241(1)"), Target("s. 241(2)")])

        assert expectation.matches(chunk("241", "1"))
        assert expectation.matches(chunk("241", "2"))
        assert not expectation.matches(chunk("239", "1"))

    def test_alternatives_differ_from_multiple_expectations(self):
        """'Either will do' must not be confused with 'both are needed'.

        One expectation with two alternatives is fully covered by one chunk; two
        separate expectations are only half covered by one chunk.
        """
        either = case(expected=(Expectation.either([Target("s. 241(1)"), Target("s. 241(2)")]),))
        both = case(expected=(Expectation.of("s. 102(1)"), Expectation.of("s. 122(1)")))
        retrieved_one = [chunk("241", "2"), chunk("102", "1")]

        assert evaluate_case(either, retrieved_one).coverage_at(3) == pytest.approx(1.0)
        assert evaluate_case(both, retrieved_one).coverage_at(3) == pytest.approx(0.5)

    def test_describe_lists_alternatives(self):
        expectation = Expectation.either([Target("s. 241(1)"), Target("s. 241(2)")])

        assert expectation.describe() == "s. 241(1) or s. 241(2)"

    def test_describe_includes_a_definition_term(self):
        assert Expectation.of("s. 2(1)", "affiliate").describe() == "s. 2(1) [affiliate]"


class TestEvaluateCase:
    def test_finds_the_rank_of_an_expected_chunk(self):
        retrieved = [chunk("99"), chunk("122", "1"), chunk("241", "1")]

        result = evaluate_case(case(), retrieved)

        assert result.first_rank == 2
        assert result.hit_at(3)
        assert not result.hit_at(1)

    def test_records_a_miss(self):
        result = evaluate_case(case(), [chunk("99"), chunk("87")])

        assert result.first_rank is None
        assert not result.hit_at(5)
        assert result.reciprocal_rank == 0.0

    def test_reciprocal_rank(self):
        retrieved = [chunk("99"), chunk("122", "1")]

        assert evaluate_case(case(), retrieved).reciprocal_rank == pytest.approx(0.5)

    def test_coverage_is_partial_when_one_of_two_is_missing(self):
        """A multi-provision question half answered is not a success.

        hit@k would call this a win; coverage records that half the required law
        was never retrieved.
        """
        multi = case(expected=(Expectation.of("s. 102(1)"), Expectation.of("s. 122(1)")))
        retrieved = [chunk("102", "1"), chunk("99")]

        result = evaluate_case(multi, retrieved)

        assert result.hit_at(3)
        assert result.coverage_at(3) == pytest.approx(0.5)

    def test_coverage_is_full_when_both_are_found(self):
        multi = case(expected=(Expectation.of("s. 102(1)"), Expectation.of("s. 122(1)")))
        retrieved = [chunk("102", "1"), chunk("122", "1")]

        assert evaluate_case(multi, retrieved).coverage_at(3) == pytest.approx(1.0)

    def test_rank_is_the_earliest_match_not_the_last(self):
        retrieved = [chunk("122", "1"), chunk("99"), chunk("122", "1")]

        assert evaluate_case(case(), retrieved).first_rank == 1


class TestEvaluate:
    def test_runs_every_case_through_the_retriever(self):
        asked = []

        def retriever(question, k):
            asked.append((question, k))
            return [chunk("122", "1")]

        cases = [case(question="one"), case(question="two")]
        results = evaluate(cases, retriever, k_values=(1, 3))

        assert len(results) == 2
        assert [q for q, _ in asked] == ["one", "two"]

    def test_retrieval_depth_defaults_to_the_largest_k(self):
        depths = []

        def retriever(question, k):
            depths.append(k)
            return []

        evaluate([case()], retriever, k_values=(1, 3, 10))

        assert depths == [10]


class TestSummarise:
    def test_reports_overall_and_per_category(self):
        results = [
            evaluate_case(case(category="paraphrase"), [chunk("122", "1")]),
            evaluate_case(case(category="section_number"), [chunk("99")]),
        ]

        summary = summarise(results, k_values=(1, 3))

        assert summary["overall"]["n"] == 2
        assert summary["overall"]["hit@1"] == pytest.approx(0.5)
        assert summary["by_category"]["paraphrase"]["hit@1"] == pytest.approx(1.0)
        assert summary["by_category"]["section_number"]["hit@1"] == pytest.approx(0.0)

    def test_category_averages_do_not_hide_a_failing_category(self):
        """The reason results are broken out by category at all."""
        results = [evaluate_case(case(category="paraphrase"), [chunk("122", "1")])] * 9
        results.append(evaluate_case(case(category="section_number"), [chunk("99")]))

        summary = summarise(results, k_values=(3,))

        assert summary["overall"]["hit@3"] == pytest.approx(0.9)
        assert summary["by_category"]["section_number"]["hit@3"] == pytest.approx(0.0)

    def test_empty_results(self):
        assert summarise([], k_values=(1,))["overall"] == {}


class TestFormatReport:
    def test_includes_categories_and_overall(self):
        results = [evaluate_case(case(category="paraphrase"), [chunk("122", "1")])]

        report = format_report(results, k_values=(1, 3))

        assert "paraphrase" in report
        assert "OVERALL" in report

    def test_lists_misses_with_what_was_expected(self):
        results = [evaluate_case(case(question="what does section 122 require?"), [chunk("99")])]

        report = format_report(results, k_values=(1, 3))

        assert "Misses" in report
        assert "what does section 122 require?" in report
        assert "s. 122(1)" in report

    def test_no_miss_section_when_everything_is_found(self):
        results = [evaluate_case(case(), [chunk("122", "1")])]

        assert "Misses" not in format_report(results, k_values=(1, 3))

    def test_a_miss_shows_what_was_retrieved_instead(self):
        """A miss alone does not say whether the right chunk ranked just outside
        k or whether retrieval went somewhere unrelated. Those need different
        fixes, so the report has to show what came back."""
        retrieved = [chunk("99", heading="Duty of care"), chunk("87", heading="Evidence")]
        results = [evaluate_case(case(), retrieved)]

        report = format_report(results, k_values=(1, 3))

        assert "got 1: s. 99" in report
        assert "Duty of care" in report


class TestQueryCache:
    """The cache that makes retrieval experiments free, offline and repeatable."""

    @pytest.fixture
    def client(self):
        import sys

        sys.path.insert(0, str(Path(__file__).parent))
        from test_rag import FakeClient

        return FakeClient()

    def test_missing_file_is_an_empty_cache(self, tmp_path):
        assert evaluation.load_query_cache(tmp_path / "absent.json") == {}

    def test_round_trip(self, tmp_path):
        path = tmp_path / "queries.json"
        evaluation.save_query_cache({"a question": [0.1, 0.2]}, path)

        assert evaluation.load_query_cache(path) == {"a question": [0.1, 0.2]}

    def test_cache_from_a_different_model_is_discarded(self, tmp_path):
        """Vectors from two embedding models are not comparable.

        Mixing them would produce meaningless similarity scores with no error.
        """
        path = tmp_path / "queries.json"
        path.write_text(json.dumps({"model": "some-older-model", "embeddings": {"q": [1.0]}}))

        assert evaluation.load_query_cache(path) == {}

    def test_warming_embeds_only_what_is_missing(self, client, tmp_path):
        path = tmp_path / "queries.json"
        evaluation.warm_query_cache(client, ["one", "two"], path)
        client.embedding_calls.clear()

        cache = evaluation.warm_query_cache(client, ["one", "two", "three"], path)

        assert client.embedding_calls == [["three"]]
        assert set(cache) == {"one", "two", "three"}

    def test_warming_nothing_makes_no_calls(self, client, tmp_path):
        path = tmp_path / "queries.json"
        evaluation.warm_query_cache(client, ["one"], path)
        client.embedding_calls.clear()

        evaluation.warm_query_cache(client, ["one"], path)

        assert client.embedding_calls == []

    def test_retriever_raises_rather_than_calling_the_api(self, client, tmp_path):
        """An uncached question must fail loudly.

        A silent fallback to a live call would turn an offline experiment into
        an unexpected bill, and would break reproducibility invisibly.
        """
        import rag

        path = tmp_path / "queries.json"
        evaluation.warm_query_cache(client, ["known question"], path)
        chunks = [chunk("122", "1")]
        matrix = rag.to_matrix([[1.0, 0.0, 0.0, 1.0]])

        retrieve = evaluation.cached_semantic_retriever(chunks, matrix, path)

        assert retrieve("known question", 1)
        with pytest.raises(KeyError, match="No cached embedding"):
            retrieve("an unknown question", 1)


class TestEvalSetFile:
    """The shipped eval set is data, and wrong data invalidates every number."""

    @pytest.fixture(scope="class")
    def cases(self):
        return load_eval_set()

    def test_loads(self, cases):
        assert len(cases) >= 20

    def test_every_case_has_at_least_one_expectation(self, cases):
        assert all(c.expected for c in cases)

    def test_questions_are_unique(self, cases):
        questions = [c.question for c in cases]
        assert len(set(questions)) == len(questions)

    def test_covers_every_category(self, cases):
        expected = {
            "paraphrase",
            "statutory_term",
            "section_number",
            "definition",
            "multi_provision",
        }
        assert {c.category for c in cases} == expected

    def test_definition_cases_name_a_term(self, cases):
        # Otherwise they would match any of the ~39 chunks citing s. 2(1).
        for eval_case in cases:
            if eval_case.category == "definition":
                for expectation in eval_case.expected:
                    assert all(t.term for t in expectation.targets), eval_case.question

    def test_citations_are_well_formed(self, cases):
        import re

        for eval_case in cases:
            for expectation in eval_case.expected:
                for target in expectation.targets:
                    assert re.fullmatch(r"s\. \d+(\.\d+)?(\(\d+\))?", target.citation), (
                        target.citation
                    )

    def test_every_expectation_has_at_least_one_target(self, cases):
        for eval_case in cases:
            assert all(e.targets for e in eval_case.expected), eval_case.question


@pytest.mark.slow
class TestEvalSetAgainstTheRealIndex:
    """Every expected provision must actually exist, or the case is unpassable."""

    @pytest.fixture(scope="class")
    def chunks(self):
        import rag

        return rag.build_chunks()

    def test_every_expected_provision_exists_in_the_index(self, chunks):
        missing = []
        for eval_case in load_eval_set():
            for expectation in eval_case.expected:
                if not any(expectation.matches(c) for c in chunks):
                    missing.append((eval_case.question, expectation.citation, expectation.term))
        assert not missing, f"eval set references chunks that do not exist: {missing}"
