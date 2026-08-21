"""
PDF ingestion for bilingual two-column statutes.

The Canada Business Corporations Act is published as a two-column bilingual
document: English in the left column, French in the right. Extracting the page
as a flat string interleaves the two languages, which puts roughly half of the
search index in a language the user is not asking questions in.

This module separates the columns geometrically, using the x-coordinate at which
each line starts. Two decisions in here are load-bearing and worth stating:

1.  Splitting happens at the LINE level, not the block level. PyMuPDF returns
    headings as a single block spanning both columns, e.g.
        'PART IV Registered Office and Records\nPARTIE IV Siege social et livres'
    In C-44, 1,977 of 8,402 blocks (24%) straddle the two columns this way, so a
    block-level split corrupts a quarter of the document. Individual lines never
    straddle: of 23,963 lines, exactly one does (the centred word "CANADA" on
    the cover page).

2.  Lines are assigned to the NEAREST COLUMN MARGIN, not by comparing against
    the page centre. In C-44 the margins sit at x=48 and x=318 on a 612pt page,
    so the nearest-margin threshold is ~183. Splitting at the page centre (306)
    would misfile the 252 lines that begin at x=288-312 -- French-column footers
    and indented text that start just left of centre. The nearest-margin rule
    classifies those correctly and still leaves ~85pt of clearance on the
    English side, whose deepest indents only reach x=100.

The layout is measured from the document rather than hardcoded, and validated
before use, so a PDF that is not two-column fails loudly instead of silently
producing mangled text. The validation asks whether a single x-position attracts
a large, sharp cluster of lines -- which is what a column margin is -- rather
than how much content sits on each half of the page. The distinction matters: a
single-column research paper with tables and equations puts 20% of its lines
right of the page centre, but they are scattered across dozens of positions and
its densest right-hand cluster holds only 3.5% of the document.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import pymupdf

# A real column margin attracts a large, sharp cluster of lines. This is the
# gate that does the actual work of rejecting non-two-column documents.
#
# Measured: C-44's right margin holds 33% of all lines; a single-column paper's
# densest right-hand cluster holds 3.5%. Anything from roughly 8% to 25%
# separates those two cases, so this sits in the middle with 2.2x headroom above
# the true case and 4.3x below the false one.
MIN_MODE_CONCENTRATION = 0.15

# Lines within this many points of a margin count as sitting on it. Absorbs
# sub-pixel rendering jitter (48.0 / 48.02 / 47.98 are the same margin).
MODE_TOLERANCE = 2.0

# The two margins must be separated by at least this fraction of the page width.
# Defensive only: it catches two sharp clusters that straddle the page centre but
# are really indent levels (e.g. body text at x=300 and an indent at x=330),
# which the concentration gate above would accept. It has never fired on a real
# document -- C-44 measures 44%, a single-column paper 49% -- so treat it as
# cheap insurance rather than as load-bearing.
MIN_MARGIN_SEPARATION = 0.15


class LayoutError(ValueError):
    """Raised when a document does not look like a two-column layout."""


@dataclass(frozen=True)
class Line:
    """A single line of text with the position it was found at.

    Position is retained deliberately. Section numbers for citations (e.g.
    "s. 102(1)") are recovered downstream from line ordering and leading
    markers; flattening to a string here would force a re-parse of the PDF to
    get that information back.
    """

    page: int
    x0: float
    y0: float
    x1: float
    text: str


@dataclass(frozen=True)
class ColumnLayout:
    """The two column margins measured from a document."""

    left_margin: float
    right_margin: float

    @property
    def threshold(self) -> float:
        """The x-coordinate midway between the two margins.

        Lines starting left of this belong to the left column. This is a
        property of the two margins, not of the page, which is the whole point
        -- see the module docstring.
        """
        return (self.left_margin + self.right_margin) / 2


@dataclass(frozen=True)
class BilingualDocument:
    """Lines from a bilingual document, split by column."""

    left: list[Line]
    right: list[Line]
    layout: ColumnLayout

    def text(self, side: str) -> str:
        """Join one column's lines into a single string in reading order."""
        lines = self.left if side == "left" else self.right
        return "\n".join(line.text for line in lines)


def assign_column(x0: float, layout: ColumnLayout) -> str:
    """Return "left" or "right" for a line starting at x0.

    Pure function, no I/O, so the classification rule can be tested directly
    without constructing a PDF.
    """
    return "left" if x0 < layout.threshold else "right"


def detect_layout(lines: list[Line], page_width: float) -> ColumnLayout:
    """Measure the two column margins from the lines themselves.

    The page centre is used only as a coarse bootstrap for locating the two
    margins -- reliable for that, because both margins sit far from the centre
    -- and never for classifying an individual line.

    Raises LayoutError if the document does not look two-column.
    """
    if not lines:
        raise LayoutError("Cannot detect column layout: no text lines found.")

    centre = page_width / 2
    total = len(lines)
    # Round to whole points so that near-identical margins group together.
    starts = [round(line.x0) for line in lines]

    candidates = {
        "left": Counter(x for x in starts if x < centre),
        "right": Counter(x for x in starts if x >= centre),
    }
    for side, counts in candidates.items():
        if not counts:
            raise LayoutError(
                f"Document does not look two-column: no lines begin on the "
                f"{side} half of the page."
            )

    margins = {side: counts.most_common(1)[0][0] for side, counts in candidates.items()}

    # The real test: does that margin attract a large, sharp cluster of lines?
    # Counting lines per half-page instead would conflate "has content on the
    # right" with "has a column on the right".
    for side, margin in margins.items():
        on_margin = sum(1 for x in starts if abs(x - margin) <= MODE_TOLERANCE)
        if on_margin < total * MIN_MODE_CONCENTRATION:
            raise LayoutError(
                f"Document does not look two-column: the densest {side}-hand "
                f"cluster, at x={margin}, holds only {on_margin}/{total} lines "
                f"({on_margin / total:.1%}), below the "
                f"{MIN_MODE_CONCENTRATION:.0%} minimum. Content scattered across "
                f"many x-positions is tables, figures or equations, not a column."
            )

    left_margin = float(margins["left"])
    right_margin = float(margins["right"])
    separation = (right_margin - left_margin) / page_width
    if separation < MIN_MARGIN_SEPARATION:
        raise LayoutError(
            f"Column margins at x={left_margin:.0f} and x={right_margin:.0f} are "
            f"only {separation:.0%} of page width apart, below the "
            f"{MIN_MARGIN_SEPARATION:.0%} minimum. These are more likely to be "
            f"indent levels than separate columns."
        )

    return ColumnLayout(left_margin=left_margin, right_margin=right_margin)


def extract_lines(document: pymupdf.Document) -> list[Line]:
    """Pull every non-empty text line, with coordinates, in reading order.

    Takes an open Document rather than a path so that tests can build a
    synthetic PDF in memory and never touch the filesystem.
    """
    lines: list[Line] = []
    for page in document:
        for block in page.get_text("dict")["blocks"]:
            # Image blocks have no "lines" key.
            for line in block.get("lines", []):
                text = "".join(span["text"] for span in line["spans"]).strip()
                if not text:
                    continue
                x0, y0, x1, _ = line["bbox"]
                lines.append(
                    Line(page=page.number, x0=x0, y0=y0, x1=x1, text=text)
                )
    return lines


def extract_bilingual(path: str) -> BilingualDocument:
    """Open a two-column bilingual PDF and split it by column.

    Note that this returns "left" and "right", not "english" and "french".
    Which language occupies which column is a fact about a particular document,
    not about two-column layouts in general, so the mapping is made explicitly
    by the caller.
    """
    with pymupdf.open(path) as document:
        lines = extract_lines(document)
        page_width = document[0].rect.width

    layout = detect_layout(lines, page_width)

    # Sort into reading order within each column: down the page, then across.
    # Block order alone is not reliable once two columns interleave.
    left = sorted(
        (ln for ln in lines if assign_column(ln.x0, layout) == "left"),
        key=lambda ln: (ln.page, ln.y0, ln.x0),
    )
    right = sorted(
        (ln for ln in lines if assign_column(ln.x0, layout) == "right"),
        key=lambda ln: (ln.page, ln.y0, ln.x0),
    )
    return BilingualDocument(left=left, right=right, layout=layout)
