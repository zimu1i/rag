"""Tests for column-based bilingual PDF ingestion.

Everything here except the tests marked "slow" runs without touching the
filesystem or the network: the classification rules are pure functions, and the
one test that needs a real PDF builds a synthetic one in memory.
"""

from pathlib import Path

import pymupdf
import pytest

from ingest import (
    ColumnLayout,
    Line,
    LayoutError,
    assign_column,
    detect_layout,
    extract_bilingual,
    extract_lines,
)

# The margins actually measured in C-44: English at x=48, French at x=318, on a
# 612pt page. Threshold is therefore 183.
C44_LAYOUT = ColumnLayout(left_margin=48.0, right_margin=318.0)
PAGE_WIDTH = 612.0


def make_line(x0: float, text: str = "text", page: int = 0, y0: float = 0.0) -> Line:
    return Line(page=page, x0=x0, y0=y0, x1=x0 + 100, text=text)


class TestAssignColumn:
    """The pure classification rule."""

    def test_left_margin_is_left_column(self):
        assert assign_column(48.0, C44_LAYOUT) == "left"

    def test_right_margin_is_right_column(self):
        assert assign_column(318.0, C44_LAYOUT) == "right"

    def test_deepest_english_indent_stays_left(self):
        # English sub-clauses indent to about x=100 in C-44, well short of 183.
        assert assign_column(100.0, C44_LAYOUT) == "left"

    def test_line_just_left_of_page_centre_is_right_column(self):
        # The case that motivates nearest-margin over page-centre splitting.
        # 252 lines in C-44 begin at x=288-312: French-column footers and
        # indented text. A page-centre split (306) would call x=300 "left".
        assert assign_column(300.0, C44_LAYOUT) == "right"

    def test_threshold_is_midway_between_margins_not_page_centre(self):
        assert C44_LAYOUT.threshold == 183.0

    @pytest.mark.parametrize(
        "x0,expected",
        [(182.9, "left"), (183.0, "right"), (183.1, "right")],
    )
    def test_boundary_is_exact_and_half_open(self, x0, expected):
        assert assign_column(x0, C44_LAYOUT) == expected


class TestDetectLayout:
    """Measuring the margins, and refusing documents that are not two-column."""

    def test_finds_the_two_dominant_margins(self):
        lines = [make_line(48.0) for _ in range(50)]
        lines += [make_line(318.0) for _ in range(50)]
        # Noise that must not become a margin.
        lines += [make_line(75.0), make_line(500.0)]

        layout = detect_layout(lines, PAGE_WIDTH)

        assert layout.left_margin == 48.0
        assert layout.right_margin == 318.0

    def test_ignores_outliers_in_favour_of_the_mode(self):
        lines = [make_line(48.0) for _ in range(50)]
        lines += [make_line(318.0) for _ in range(50)]
        lines += [make_line(52.0) for _ in range(5)]

        assert detect_layout(lines, PAGE_WIDTH).left_margin == 48.0

    def test_rejects_single_column_document(self):
        # A single-column document with a few centred headings on the right.
        lines = [make_line(72.0) for _ in range(100)]
        lines += [make_line(400.0) for _ in range(3)]

        with pytest.raises(LayoutError, match="does not look two-column"):
            detect_layout(lines, PAGE_WIDTH)

    def test_rejects_content_scattered_across_the_right_half(self):
        """Regression test for a real false positive.

        A single-column paper with tables and equations puts a fifth of its
        lines right of the page centre. An earlier version of this check
        counted lines per half-page and accepted such a document as
        two-column. What distinguishes a column is that ONE x-position
        attracts a dense cluster; scattered content does not.
        """
        lines = [make_line(108.0) for _ in range(100)]
        # 30 lines right of centre -- 23% of the document -- but spread across
        # 30 distinct positions, so no single one is a margin.
        lines += [make_line(350.0 + 5 * i) for i in range(30)]

        with pytest.raises(LayoutError, match="not a column"):
            detect_layout(lines, PAGE_WIDTH)

    def test_tolerates_sub_pixel_jitter_on_a_margin(self):
        # Rendering puts nominally identical margins at 48.0 / 48.4 / 47.6.
        # These must count as one margin, not three.
        lines = [make_line(48.0 + 0.4 * (i % 3) - 0.4) for i in range(50)]
        lines += [make_line(318.0) for _ in range(50)]

        layout = detect_layout(lines, PAGE_WIDTH)

        assert layout.left_margin == 48.0

    def test_rejects_margins_that_are_merely_indents(self):
        # Two clusters, but only 30pt apart -- indent levels, not columns.
        lines = [make_line(300.0) for _ in range(50)]
        lines += [make_line(330.0) for _ in range(50)]

        with pytest.raises(LayoutError, match="indent levels"):
            detect_layout(lines, PAGE_WIDTH)

    def test_rejects_empty_document(self):
        with pytest.raises(LayoutError, match="no text lines"):
            detect_layout([], PAGE_WIDTH)


def build_two_column_pdf(pairs, width=612, height=792):
    """Build a synthetic bilingual PDF in memory.

    Uses the same margins as C-44 so the fixture exercises the real geometry.
    """
    document = pymupdf.open()
    page = document.new_page(width=width, height=height)
    y = 100
    for left_text, right_text in pairs:
        page.insert_text((48, y), left_text)
        page.insert_text((318, y), right_text)
        y += 20
    return document


class TestExtractLines:
    def test_reads_position_and_text_for_every_line(self):
        document = build_two_column_pdf([("Hello", "Bonjour")])

        lines = extract_lines(document)

        assert len(lines) == 2
        assert {line.text for line in lines} == {"Hello", "Bonjour"}
        assert all(line.page == 0 for line in lines)

    def test_skips_blank_lines(self):
        document = build_two_column_pdf([("Hello", "Bonjour"), ("   ", "   ")])

        assert len(extract_lines(document)) == 2


class TestExtractBilingual:
    def test_separates_the_two_columns(self, tmp_path):
        pairs = [(f"English line {i}", f"Ligne francaise {i}") for i in range(20)]
        path = tmp_path / "bilingual.pdf"
        build_two_column_pdf(pairs).save(path)

        document = extract_bilingual(str(path))

        assert len(document.left) == 20
        assert len(document.right) == 20
        assert all("English" in line.text for line in document.left)
        assert all("francaise" in line.text for line in document.right)

    def test_columns_are_disjoint(self, tmp_path):
        pairs = [(f"English line {i}", f"Ligne francaise {i}") for i in range(20)]
        path = tmp_path / "bilingual.pdf"
        build_two_column_pdf(pairs).save(path)

        document = extract_bilingual(str(path))

        assert "francaise" not in document.text("left")
        assert "English" not in document.text("right")

    def test_preserves_reading_order(self, tmp_path):
        pairs = [(f"English line {i}", f"Ligne francaise {i}") for i in range(20)]
        path = tmp_path / "bilingual.pdf"
        build_two_column_pdf(pairs).save(path)

        document = extract_bilingual(str(path))

        assert [line.text for line in document.left] == [p[0] for p in pairs]


@pytest.mark.slow
class TestRealAct:
    """Properties measured against the real C-44 document.

    Marked slow: parses 253 pages. Run with `pytest -m slow`, skip with
    `pytest -m "not slow"`.
    """

    @pytest.fixture(scope="class")
    def act(self):
        return extract_bilingual("C-44.pdf")

    def test_detects_the_expected_margins(self, act):
        assert act.layout.left_margin == 48.0
        assert act.layout.right_margin == 318.0

    def test_split_is_roughly_balanced(self, act):
        # A bilingual statute carries each provision in both languages, so the
        # two columns should be close in size. A lopsided split means the
        # classification went wrong.
        ratio = len(act.left) / len(act.right)
        assert 0.8 < ratio < 1.2, f"columns unbalanced: {ratio:.2f}"

    def test_english_column_is_not_contaminated_with_french(self, act):
        # Marker phrases that appear only in the French text of the Act.
        english = act.text("left")
        for marker in ("Sociétés par actions", "Dernière modification", "L.R. (1985)"):
            assert marker not in english, f"French marker leaked into English: {marker}"

    def test_french_column_is_populated(self, act):
        # Guards against a split that quietly puts everything on one side.
        assert "Sociétés par actions" in act.text("right")

    def test_recovers_known_english_provision(self, act):
        # s. 24(1), verified by eye against the published Act.
        assert "Shares of a corporation shall be in registered" in act.text("left")


SINGLE_COLUMN_PDF = Path(__file__).parent / "fixtures" / "attention-is-all-you-need.pdf"


@pytest.mark.slow
@pytest.mark.skipif(
    not SINGLE_COLUMN_PDF.exists(), reason="single-column fixture not present"
)
def test_rejects_a_real_single_column_document():
    """The negative case, against a real document rather than a synthetic one.

    "Attention Is All You Need" is single-column, but its tables, figures and
    equations put 20% of its lines right of the page centre. An earlier version
    of the layout check accepted it as two-column and invented a split
    threshold. A synthetic negative fixture did not catch that; this does.
    """
    with pytest.raises(LayoutError, match="does not look two-column"):
        extract_bilingual(str(SINGLE_COLUMN_PDF))
