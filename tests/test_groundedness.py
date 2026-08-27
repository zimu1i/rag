"""Tests for the groundedness guards.

Both mechanisms are deterministic, so every test here is exact: no model, no
embeddings from an API, no tolerance windows.
"""

import numpy as np
import pytest

from legalmind.chunking import Chunk
from legalmind.groundedness import (
    audit_citations,
    cited_provisions,
    support_score,
    unsupported_citations,
)


def chunk(section, subsection=None, text="text"):
    return Chunk(text=text, section=section, subsection=subsection)


class TestAuditCitations:
    """The three-way split, which a real caught case forced.

    Asked to explain s. 190(1), the model cited s. 173. It had not been
    retrieved -- but s. 190(1)'s own text reads "amend its articles under
    section 173 or 174", so the model was repeating the Act, not inventing.
    """

    def test_a_retrieved_provision_is_grounded(self):
        audit = audit_citations("see s. 122(1)", [chunk("122", "1")])

        assert audit.grounded == ["s. 122(1)"]
        assert audit.unsupported == []

    def test_a_cross_reference_inside_retrieved_text_is_not_a_fabrication(self):
        retrieved = [chunk("190", "1", text="amend its articles under section 173 or 174")]

        audit = audit_citations("dissent applies to amendments under s. 173", retrieved)

        assert audit.cross_referenced == ["s. 173"]
        assert audit.unsupported == []

    def test_a_provision_mentioned_nowhere_is_unsupported(self):
        audit = audit_citations("see s. 45(2)", [chunk("122", "1")])

        assert audit.unsupported == ["s. 45(2)"]
        assert audit.cross_referenced == []

    def test_an_invented_subsection_of_a_retrieved_section_is_unsupported(self):
        """Holding s. 122(1) and citing s. 122(9) is a fabricated subsection,
        not a reference to somewhere else -- even though "122" appears in the
        retrieved text."""
        retrieved = [chunk("122", "1", text="122 (1) Every director shall act honestly")]

        audit = audit_citations("see s. 122(9)", retrieved)

        assert audit.unsupported == ["s. 122(9)"]
        assert audit.cross_referenced == []

    def test_sorts_each_category(self):
        retrieved = [chunk("122", "1"), chunk("120", "1")]

        audit = audit_citations("s. 122(1), s. 120(1), s. 300, s. 45", retrieved)

        assert audit.grounded == ["s. 120(1)", "s. 122(1)"]
        assert audit.unsupported == ["s. 300", "s. 45"]

    def test_an_answer_with_no_citations_is_empty_everywhere(self):
        audit = audit_citations("I cannot answer that.", [chunk("122", "1")])

        assert audit.grounded == audit.cross_referenced == audit.unsupported == []


class TestSupportScore:
    def test_identical_vectors_score_one(self):
        assert support_score([1.0, 0.0], [[1.0, 0.0]]) == pytest.approx(1.0)

    def test_orthogonal_vectors_score_zero(self):
        assert support_score([1.0, 0.0], [[0.0, 1.0]]) == pytest.approx(0.0)

    def test_takes_the_closest_of_several(self):
        score = support_score([1.0, 0.0], [[0.0, 1.0], [0.9, 0.1], [0.0, 1.0]])

        assert score == pytest.approx(0.994, abs=1e-3)

    def test_is_insensitive_to_vector_magnitude(self):
        """Cosine, not dot product: a longer vector is not better supported."""
        short = support_score([1.0, 0.0], [[1.0, 0.0]])
        long = support_score([1.0, 0.0], [[50.0, 0.0]])

        assert short == pytest.approx(long)

    def test_nothing_retrieved_means_no_support(self):
        assert support_score([1.0, 0.0], []) == 0.0

    def test_zero_vector_does_not_divide_by_zero(self):
        assert not np.isnan(support_score([0.0, 0.0], [[0.0, 0.0]]))


class TestCitedProvisions:
    def test_extracts_section_and_subsection(self):
        assert cited_provisions("as required by s. 122(1)") == {("122", "1")}

    def test_extracts_a_bare_section(self):
        assert cited_provisions("see s. 190") == {("190", None)}

    def test_extracts_a_decimal_section(self):
        assert cited_provisions("under s. 2.1(3)") == {("2.1", "3")}

    def test_extracts_several(self):
        found = cited_provisions("s. 122(1) and s. 122(2), plus s. 241")

        assert found == {("122", "1"), ("122", "2"), ("241", None)}

    def test_ignores_prose_without_citations(self):
        assert cited_provisions("The directors shall act honestly.") == set()

    def test_tolerates_missing_space(self):
        assert cited_provisions("see s.122(1)") == {("122", "1")}

    def test_ignores_cross_references_written_out_in_full(self):
        """Deliberately narrow, and this is the reason.

        The Act's own text spells cross-references out -- "as required by
        section 86", "subject to subsection 146(5)" -- while the citation
        labels given to the model use the "s. 122(1)" form. Matching the
        spelled-out form too would flag quoted statutory text as a fabricated
        citation: an answer quoting s. 87 would be accused of inventing s. 86.
        """
        quoted = "Evidence of compliance as required by section 86 shall consist of"

        assert cited_provisions(quoted) == set()

    def test_a_quoted_cross_reference_is_not_reported_as_fabricated(self):
        answer = "Under s. 87, evidence of compliance as required by section 86 is needed."

        assert unsupported_citations(answer, [chunk("87")]) == []


class TestUnsupportedCitations:
    def test_a_retrieved_provision_is_supported(self):
        assert unsupported_citations("see s. 122(1)", [chunk("122", "1")]) == []

    def test_a_fabricated_provision_is_reported(self):
        """The failure this exists for: a citation to something never retrieved
        did not come from the excerpts."""
        result = unsupported_citations("see s. 45(2)", [chunk("122", "1")])

        assert result == ["s. 45(2)"]

    def test_a_section_level_citation_is_satisfied_by_any_subsection(self):
        """An answer saying "s. 122" while relying on the retrieved s. 122(1) is
        citing accurately, just less precisely."""
        assert unsupported_citations("see s. 122", [chunk("122", "1")]) == []

    def test_the_wrong_subsection_is_not_supported(self):
        # s. 122(1) was retrieved; s. 122(9) was not and does not exist.
        result = unsupported_citations("see s. 122(9)", [chunk("122", "1")])

        assert result == ["s. 122(9)"]

    def test_reports_every_unsupported_citation(self):
        result = unsupported_citations(
            "s. 122(1), s. 45(2) and s. 300", [chunk("122", "1")]
        )

        assert result == ["s. 300", "s. 45(2)"]

    def test_an_answer_with_no_citations_is_vacuously_clean(self):
        assert unsupported_citations("I cannot answer that.", [chunk("122", "1")]) == []

    def test_nothing_retrieved_makes_every_citation_unsupported(self):
        assert unsupported_citations("see s. 122(1)", []) == ["s. 122(1)"]


class TestEvalNegatives:
    """The out-of-scope set is data; wrong data invalidates the calibration."""

    @pytest.fixture(scope="class")
    def negatives(self):
        from legalmind import evaluation

        return evaluation.load_negatives()

    def test_loads(self, negatives):
        assert len(negatives) >= 8

    def test_every_case_explains_why_it_is_out_of_scope(self, negatives):
        assert all(n["why"].strip() for n in negatives)

    def test_questions_are_unique(self, negatives):
        questions = [n["question"] for n in negatives]
        assert len(set(questions)) == len(questions)

    def test_includes_hard_near_misses(self, negatives):
        """A threshold tuned only on obvious negatives will wave through the
        corporate-law questions this statute happens not to govern."""
        near = [n for n in negatives if n["difficulty"] == "near_miss"]

        assert len(near) >= 4

    def test_does_not_overlap_the_answerable_set(self, negatives):
        from legalmind import evaluation

        answerable = {case.question for case in evaluation.load_eval_set()}

        assert not answerable & {n["question"] for n in negatives}
