"""Tests for turning extracted lines into clean prose.

All fast tests build synthetic Line records directly -- no PDF, no filesystem,
no network. The slow tests assert properties of the real Act.
"""

import pytest

import cleanup
from cleanup import (
    CleanupError,
    find_front_matter,
    hyphenated_vocabulary,
    join_wrapped_lines,
    strip_running_headers,
    to_paragraphs,
)
from ingest import Line, extract_bilingual

PAGE_HEIGHT = 792.0


def make_line(text, page=0, y0=100.0, block=0, x0=48.0):
    return Line(page=page, x0=x0, y0=y0, x1=x0 + 200, text=text, block=block)


def build_page(page, body_texts, body_start=100.0, header=True, footer=True):
    """A page shaped like C-44's: header at y=45-59, body from y=100, footer at y=758."""
    lines = []
    if header:
        # The title repeats on every page; the Part name does not, but is part
        # of the same visual header block.
        lines.append(make_line("Canada Business Corporations", page, 45.1, block=0))
        lines.append(make_line(f"PART {page} Something", page, 49.2, block=0))
    y = body_start
    for i, text in enumerate(body_texts):
        lines.append(make_line(text, page, y, block=i + 1))
        y += 11.5
    if footer:
        lines.append(make_line("Current to May 26, 2026", page, 758.9, block=90))
        lines.append(make_line(f"Page {page}", page, 770.4, block=91))
    return lines


class TestHyphenatedVocabulary:
    def test_collects_compounds_appearing_within_a_line(self):
        lines = [make_line("subject to the by-laws of the corporation")]

        assert "by-laws" in hyphenated_vocabulary(lines)

    def test_ignores_words_broken_at_a_line_end(self):
        # "Corpo-" ends the line, so there is no compound to learn here.
        lines = [make_line("the Canada Business Corpo-"), make_line("rations Act")]

        assert hyphenated_vocabulary(lines) == set()


class TestJoinWrappedLines:
    def test_repairs_a_word_broken_across_lines(self):
        result = join_wrapped_lines(["the Canada Business Corpo-", "rations Act"], set())

        assert result == "the Canada Business Corporations Act"

    def test_preserves_a_hyphen_the_document_uses_elsewhere(self):
        # "by-laws" is a term of art. Stripping the hyphen would produce
        # "bylaws", which no longer matches a query for "by-laws" under BM25.
        result = join_wrapped_lines(["subject to the by-", "laws of the"], {"by-laws"})

        assert result == "subject to the by-laws of the"

    def test_joins_ordinary_lines_with_a_space(self):
        result = join_wrapped_lines(["shares of a", "corporation"], set())

        assert result == "shares of a corporation"

    def test_leaves_hyphen_when_next_line_starts_a_new_word(self):
        # A capital letter after the break means this is not a split word.
        result = join_wrapped_lines(["the Governor-", "General acts"], set())

        assert result == "the Governor- General acts"

    def test_ignores_blank_lines(self):
        assert join_wrapped_lines(["shares", "   ", "issued"], set()) == "shares issued"


class TestStripRunningHeaders:
    def test_removes_repeated_header_and_footer(self):
        lines = []
        for page in range(10):
            lines += build_page(page, [f"Body text on page {page}"])

        kept = strip_running_headers(lines, PAGE_HEIGHT)

        assert [ln.text for ln in kept] == [f"Body text on page {p}" for p in range(10)]

    def test_keeps_first_body_line_despite_it_sitting_inside_the_margin_band(self):
        """The first body line falls inside the 15% margin band.

        It survives because it is not repeated across pages. Repetition is the
        real discriminator; the band is only a secondary safety filter.
        """
        lines = []
        for page in range(10):
            lines += build_page(page, [f"Unique first line {page}"], body_start=100.0)

        kept = strip_running_headers(lines, PAGE_HEIGHT)

        assert len(kept) == 10

    def test_identical_text_repeated_in_the_margin_is_boilerplate(self):
        """Documents the flip side of the test above.

        Text that both repeats on every page and sits in a margin is removed,
        even if it looks like prose. That is the correct reading: a sentence
        printed at the top of all 253 pages is boilerplate by definition.
        """
        lines = []
        for page in range(10):
            lines += build_page(page, ["Identical every page"], body_start=100.0)

        assert strip_running_headers(lines, PAGE_HEIGHT) == []

    def test_removes_header_lines_that_do_not_themselves_repeat(self):
        """The Part name appears on only one page but is still header text.

        It is removed because it is visually contiguous with a line that does
        repeat -- repetition alone would leave it behind.
        """
        lines = []
        for page in range(10):
            lines += build_page(page, ["Body text here"])

        kept = strip_running_headers(lines, PAGE_HEIGHT)

        assert not any("PART" in ln.text for ln in kept)

    def test_keeps_repeated_text_that_sits_in_the_body(self):
        # "Offence" is a genuine heading that recurs throughout the Act. It
        # repeats often enough to be a candidate, but is nowhere near a margin.
        lines = []
        for page in range(10):
            lines += build_page(page, ["Offence", "Body text here"], body_start=300.0)

        kept = strip_running_headers(lines, PAGE_HEIGHT)

        assert sum(1 for ln in kept if ln.text == "Offence") == 10

    def test_refuses_to_strip_implausibly_many_lines(self):
        # Every line is contiguous with the repeated header, so growth would
        # consume the whole page.
        lines = []
        for page in range(10):
            page_lines = [make_line("Canada Business Corporations", page, 45.1)]
            y = 56.0
            for i in range(12):
                page_lines.append(make_line(f"line {i}", page, y, block=i))
                y += 11.5
            lines += page_lines

        with pytest.raises(CleanupError, match="run into the body"):
            strip_running_headers(lines, PAGE_HEIGHT)

    def test_empty_input(self):
        assert strip_running_headers([], PAGE_HEIGHT) == []


class TestFindFrontMatter:
    def test_detects_table_of_contents_pages(self):
        lines = []
        for page in range(3):  # ToC: numbers paired with headings
            for i in range(10):
                lines.append(make_line(str(100 + i), page, 100.0 + i, block=i))
                lines.append(make_line(f"Heading {i}", page, 105.0 + i, block=i))
        for page in range(3, 20):
            lines += [make_line("Ordinary prose here.", page, 100.0)]

        assert find_front_matter(lines) == {0, 1, 2}

    def test_returns_nothing_when_there_is_no_table_of_contents(self):
        lines = [make_line("Ordinary prose here.", page, 100.0) for page in range(20)]

        assert find_front_matter(lines) == set()

    def test_ignores_number_heavy_pages_late_in_the_document(self):
        # A schedule or table near the end must not be mistaken for a ToC.
        lines = [make_line("Ordinary prose here.", page, 100.0) for page in range(20)]
        for i in range(10):
            lines.append(make_line(str(100 + i), 18, 100.0 + i, block=i))

        assert find_front_matter(lines) == set()

    def test_sweeps_up_the_cover_page_before_the_contents(self):
        lines = [make_line("CANADA", 0, 300.0)]  # cover, no numbers
        for i in range(10):
            lines.append(make_line(str(100 + i), 1, 100.0 + i, block=i))
            lines.append(make_line(f"Heading {i}", 1, 105.0 + i, block=i))
        for page in range(2, 20):
            lines.append(make_line("Ordinary prose here.", page, 100.0))

        assert find_front_matter(lines) == {0, 1}


class TestToParagraphs:
    def test_end_to_end(self):
        lines = []
        # ToC page
        for i in range(10):
            lines.append(make_line(str(100 + i), 0, 100.0 + i, block=i))
            lines.append(make_line(f"Heading {i}", 0, 105.0 + i, block=i))
        # Body pages, with a word broken across two lines of one block. Placed
        # clear of the margin band so this exercises reflow, not header removal.
        for page in range(1, 12):
            lines += build_page(page, [])
            lines.append(make_line(f"On page {page} the Corpo-", page, 300.0, block=5))
            lines.append(make_line("rations Act applies", page, 311.5, block=5))

        paragraphs = to_paragraphs(lines, PAGE_HEIGHT)

        assert len(paragraphs) == 11
        assert [p.text for p in paragraphs] == [
            f"On page {page} the Corporations Act applies" for page in range(1, 12)
        ]
        assert all(p.page >= 1 for p in paragraphs)

    def test_empty_input(self):
        assert to_paragraphs([], PAGE_HEIGHT) == []

    def test_raises_if_everything_is_removed(self, monkeypatch):
        """Exercises the last-resort guard.

        TOC_SEARCH_FRACTION normally makes this unreachable by capping front
        matter at the first fifth of the document, so the search window is
        widened here to confirm the guard actually fires rather than sitting
        in the file as untested reassurance.
        """
        monkeypatch.setattr(cleanup, "TOC_SEARCH_FRACTION", 1.0)
        # Numbers differ per page so that nothing is mistaken for boilerplate.
        lines = []
        for page in range(10):
            for i in range(10):
                lines.append(make_line(str(page * 100 + i), page, 100.0 + i, block=i))

        with pytest.raises(CleanupError, match="removed every line"):
            to_paragraphs(lines, PAGE_HEIGHT)


@pytest.mark.slow
class TestRealAct:
    @pytest.fixture(scope="class")
    def paragraphs(self):
        act = extract_bilingual("C-44.pdf")
        return to_paragraphs(act.left, act.page_height)

    @pytest.fixture(scope="class")
    def text(self, paragraphs):
        return "\n\n".join(p.text for p in paragraphs)

    def test_running_boilerplate_is_gone(self, text):
        for marker in ("Current to", "Last amended on March"):
            assert marker not in text, f"boilerplate survived: {marker}"

    def test_table_of_contents_is_gone(self, paragraphs):
        # The ToC occupies pages 0-12; the body starts after it.
        assert min(p.page for p in paragraphs) > 12

    def test_words_broken_across_lines_are_repaired(self, text):
        for broken in ("Corpo-", "unan-", "gov-", "incor-"):
            assert broken not in text, f"unrepaired line-break hyphen: {broken}"

    def test_legal_compounds_keep_their_hyphens(self, text):
        for compound in ("by-laws", "receiver-manager", "take-over"):
            assert compound in text, f"compound was flattened: {compound}"

    def test_known_provision_reads_correctly(self, text):
        assert "1 This Act may be cited as the Canada Business Corporations Act." in text

    def test_retains_most_of_the_document(self, text):
        # Cleanup should remove noise, not content. Raw English is ~500k chars.
        assert len(text) > 400_000
