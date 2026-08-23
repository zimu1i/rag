"""Tests for structure-aware chunking.

Fast tests build Paragraph records directly. The slow tests assert properties of
the real Act, including the two that matter most for a legal tool: every chunk
can be cited, and nothing that is not currently law gets indexed.
"""

import pytest

from chunking import (
    Chunk,
    MAX_CHUNK_CHARS,
    chunk_paragraphs,
    is_heading,
    merge_page_breaks,
)
from cleanup import Paragraph, to_paragraphs
from ingest import extract_bilingual


def para(text, page=20, block=0):
    return Paragraph(page=page, block=block, text=text)


class TestCitation:
    def test_section_and_subsection(self):
        assert Chunk(text="x", section="24", subsection="1").citation == "s. 24(1)"

    def test_section_only(self):
        assert Chunk(text="x", section="24").citation == "s. 24"

    def test_schedule_item_is_not_cited_as_a_section(self):
        # The Schedule restarts numbering at 1. Citing its first item as "s. 1"
        # would point at the short title of the Act instead.
        chunk = Chunk(text="x", section="1", part="SCHEDULE")

        assert chunk.citation == "Schedule, item 1"

    def test_uncited(self):
        assert Chunk(text="x").citation == "(uncited)"


class TestMergePageBreaks:
    def test_merges_a_provision_split_across_a_page_boundary(self):
        paragraphs = [
            para("a corporation may be dissolved by an event that", page=20),
            para("has occurred and is continuing.", page=21),
        ]

        merged = merge_page_breaks(paragraphs)

        assert len(merged) == 1
        assert merged[0].text == (
            "a corporation may be dissolved by an event that has occurred and is continuing."
        )

    def test_does_not_merge_when_the_previous_sentence_ended(self):
        paragraphs = [para("a complete sentence.", page=20), para("more text", page=21)]

        assert len(merge_page_breaks(paragraphs)) == 2

    def test_does_not_merge_within_a_page(self):
        # Block segmentation is already correct inside a page.
        paragraphs = [para("text without a stop", page=20), para("more text", page=20)]

        assert len(merge_page_breaks(paragraphs)) == 2

    def test_does_not_swallow_a_new_subsection(self):
        paragraphs = [para("text without a stop", page=20), para("(2) A new one", page=21)]

        assert len(merge_page_breaks(paragraphs)) == 2

    def test_empty(self):
        assert merge_page_breaks([]) == []


class TestIsHeading:
    def test_marginal_note(self):
        assert is_heading("Duty of care of directors and officers")

    def test_prose_is_not_a_heading(self):
        assert not is_heading("The corporation shall send a notice to each shareholder.")

    def test_clause_fragment_is_not_a_heading(self):
        # Ends with a semicolon, so it is a piece of a list, not a title.
        assert not is_heading("whether or not a successor has been appointed;")

    def test_section_marker_is_not_a_heading(self):
        assert not is_heading("24 (1) Shares of a corporation")


class TestChunkParagraphs:
    def test_extracts_section_and_subsection(self):
        chunks = chunk_paragraphs([para("24 (1) Shares shall be in registered form.")])

        assert chunks[0].section == "24"
        assert chunks[0].subsection == "1"
        assert chunks[0].citation == "s. 24(1)"

    def test_subsection_inherits_the_current_section(self):
        chunks = chunk_paragraphs(
            [para("24 (1) Shares shall be in registered form."), para("(2) A second rule.")]
        )

        assert [c.citation for c in chunks] == ["s. 24(1)", "s. 24(2)"]

    def test_marginal_note_attaches_to_the_provision_it_introduces(self):
        """A marginal note precedes its provision, so it must attach forward.

        Attaching it to the paragraph just ended would label every provision
        with the previous provision's heading.
        """
        chunks = chunk_paragraphs(
            [
                para("24 (1) The first provision."),
                para("Issue of shares"),
                para("25 (1) The second provision."),
            ]
        )

        assert chunks[0].heading is None
        assert chunks[1].heading == "Issue of shares"
        assert chunks[1].text.startswith("Issue of shares\n")

    def test_drops_citation_trailers(self):
        chunks = chunk_paragraphs(
            [para("24 (1) A provision."), para("R.S., 1985, c. C-44, s. 23; 2001, c. 14, s. 12.")]
        )

        assert len(chunks) == 1
        assert "R.S., 1985" not in chunks[0].text

    @pytest.mark.parametrize(
        "marker",
        [
            "(2) [Repealed, 1991, c. 45, s. 551]",
            "(3) and (4) [Repealed, 2001, c. 14, s. 52]",  # a repealed range
            "24 [Repealed, 2001, c. 14, s. 52]",  # a whole repealed section
        ],
    )
    def test_drops_repealed_markers(self, marker):
        chunks = chunk_paragraphs([para("24 (1) A provision."), para(marker)])

        assert len(chunks) == 1
        assert "[Repealed" not in chunks[0].text

    def test_tracks_the_current_part(self):
        chunks = chunk_paragraphs([para("PART V"), para("24 (1) A provision.")])

        assert chunks[0].part == "V"

    def test_each_definition_becomes_its_own_chunk(self):
        chunks = chunk_paragraphs(
            [
                para("2 (1) In this Act,"),
                para("affairs means the relationships among a corporation;"),
                para("affiliate means an affiliated body corporate;"),
            ]
        )

        assert [c.heading for c in chunks] == [None, "affairs", "affiliate"]
        assert all(c.citation == "s. 2(1)" for c in chunks)

    def test_a_definition_does_not_repeat_its_own_term(self):
        chunks = chunk_paragraphs(
            [para("2 (1) In this Act,"), para("affiliate means an affiliated body corporate;")]
        )

        assert chunks[1].text == "affiliate means an affiliated body corporate;"

    def test_definitions_only_apply_inside_a_definitions_subsection(self):
        # "means" in ordinary prose must not start a new chunk.
        chunks = chunk_paragraphs(
            [para("24 (1) A provision."), para("the term means nothing here")]
        )

        assert len(chunks) == 1

    def test_splits_an_oversized_provision_at_item_boundaries(self):
        items = " ".join(f"({chr(97 + i)}) {'word ' * 40}" for i in range(8))
        chunks = chunk_paragraphs([para(f"5 (1) An offence under any of: {items}")])

        assert len(chunks) > 1
        # Every piece keeps the stem, so each stands on its own.
        assert all("An offence under any of:" in c.text for c in chunks)
        assert all(c.citation == "s. 5(1)" for c in chunks)

    def test_does_not_split_a_provision_that_fits(self):
        chunks = chunk_paragraphs([para("5 (1) " + "word " * 50)])

        assert len(chunks) == 1

    def test_excludes_amendments_not_in_force(self):
        """Text that is not yet law must never reach the index."""
        chunks = chunk_paragraphs(
            [
                para("24 (1) A provision in force."),
                para("AMENDMENTS NOT IN FORCE"),
                para("144 Subsection 261(1) of the Act is amended."),
            ]
        )

        assert len(chunks) == 1
        assert chunks[0].section == "24"

    def test_excludes_related_provisions(self):
        chunks = chunk_paragraphs(
            [para("24 (1) A provision."), para("RELATED PROVISIONS"), para("11 Transitional.")]
        )

        assert len(chunks) == 1

    def test_schedule_items_are_labelled_as_schedule(self):
        chunks = chunk_paragraphs(
            [para("24 (1) A provision."), para("SCHEDULE"), para("1 An offence under the Code.")]
        )

        assert chunks[1].part == "SCHEDULE"
        assert chunks[1].citation == "Schedule, item 1"

    def test_empty(self):
        assert chunk_paragraphs([]) == []


@pytest.mark.slow
class TestRealAct:
    @pytest.fixture(scope="class")
    def chunks(self):
        act = extract_bilingual("C-44.pdf")
        return chunk_paragraphs(to_paragraphs(act.left, act.page_height))

    def test_every_chunk_can_be_cited(self, chunks):
        assert all(c.citation != "(uncited)" for c in chunks)

    def test_nothing_from_the_end_matter_is_indexed(self, chunks):
        # "RELATED PROVISIONS" starts on page 246; nothing at or beyond it is law
        # currently in force.
        assert max(max(c.pages) for c in chunks) < 246

    def test_no_repealed_markers_survive(self, chunks):
        assert not any("[Repealed" in c.text for c in chunks)

    def test_schedule_is_not_cited_as_sections_of_the_act(self, chunks):
        schedule = [c for c in chunks if c.part == "SCHEDULE"]
        assert schedule, "expected the Schedule to be indexed"
        assert all(c.citation.startswith("Schedule") for c in schedule)

    def test_recovers_the_directors_duty_of_care(self, chunks):
        match = [c for c in chunks if c.section == "122" and c.subsection == "1"]
        assert len(match) == 1
        assert match[0].heading == "Duty of care of directors and officers"
        assert "act honestly and in good faith" in match[0].text

    def test_definitions_are_individually_addressable(self, chunks):
        match = [c for c in chunks if c.heading == "affiliate"]
        assert len(match) == 1
        assert match[0].citation == "s. 2(1)"
        assert len(match[0].text) < 300, "definition should be its own small chunk"

    def test_most_provisions_are_left_whole(self, chunks):
        oversized = [c for c in chunks if len(c.text) > MAX_CHUNK_CHARS]
        assert len(oversized) / len(chunks) < 0.05

    def test_chunks_are_a_sane_size(self, chunks):
        assert 900 < len(chunks) < 1_500
        assert max(len(c.text) for c in chunks) < 2_500
