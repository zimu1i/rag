"""
Turn raw extracted lines into clean prose paragraphs.

`ingest.py` yields one record per printed line, which is not what should go into
a search index. A page of a consolidated statute carries three kinds of noise:

1.  Running headers and footers. Every one of C-44's 253 pages repeats
    "Current to May 26, 2026", "Last amended on March 26, 2026" and the document
    title, plus the current Part and section range. That is ~750 lines of text
    that answers no question but does match queries.

2.  A table of contents. Pages 2-12 are section numbers paired with headings and
    no prose at all. Left in the index it matches almost any query lexically
    while containing no actual law -- it is the worst kind of noise, because it
    looks relevant to a scorer and is useless to a reader.

3.  Words broken across lines. 1,844 lines (15.4%) end mid-word with a hyphen:
    "Corpo-/rations", "unan-/imous", "gov-/ernment". Left unrepaired these embed
    as nonsense and, worse, BM25 will index "corpo" and "rations" as terms.

Each is removed using a signal measured from the document rather than a
hardcoded rule, and each threshold below records the measurement that justifies
it along with the headroom on either side.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass

from ingest import Line

# A line whose exact text repeats on at least this share of pages is running
# header or footer boilerplate.
#
# Measured in C-44: "Current to May 26, 2026" appears on 100% of pages and the
# document title on 95%, while the most repeated genuine content ("PART VII
# Security Certificates...") appears on 10%. This sits 3x above the highest
# non-boilerplate value and 3x below the lowest true one.
REPEAT_SHARE = 0.30

# Having found a boilerplate line, the rest of its header block is whatever sits
# within this many points of it. Headers are a visual block: the title, the Part
# name and the section range are separate lines that must all go.
#
# Measured: gaps inside the header block are 4.1pt and 10.2pt, the gap from
# header to body is 28.3pt, and body line spacing is 11.5pt. 20pt sits 2x above
# the largest in-header gap and 1.4x below the header-to-body gap.
MAX_BLOCK_GAP = 20.0

# Boilerplate must sit within this fraction of the page height from the top or
# bottom edge. Repetition alone is not sufficient evidence: this makes it
# impossible for a phrase that recurs in body text to delete its surroundings.
#
# Measured: C-44's header sits at 5.7-7.5% of page height and its footers at
# 95.8-97.3%, so a 15% band contains both with roughly 2x room to spare.
#
# Note that the first body line, at 11.1%, also falls inside this band. That is
# harmless: the band is a secondary filter, and repetition is what actually
# identifies boilerplate. A page's first body line differs from page to page and
# so is never a seed. Widening the band would not break anything; narrowing it
# below ~8% would start missing real header lines.
MARGIN_BAND = 0.15

# Guard against the block-growth above running away into the body of the page.
# C-44's header is 3 lines and its footer 2.
MAX_MARGIN_LINES = 8

# A page whose lines are at least this often nothing but a number is a table of
# contents page.
#
# Measured: ToC pages score 33-44%, body pages score exactly 0.0%. The
# separation is total, so this threshold is uncontroversial -- anything from 5%
# to 30% behaves identically.
TOC_NUMBER_SHARE = 0.15

# Only look for a table of contents in the front of the document, so that a
# schedule or a table late in the body is never mistaken for one.
TOC_SEARCH_FRACTION = 0.20

_NUMBER_ONLY = re.compile(r"^\d+(\.\d+)?$")
_HYPHEN_COMPOUND = re.compile(r"\b([a-z]+-[a-z]+)\b")
_TRAILING_WORD = re.compile(r"(\S+)$")


class CleanupError(ValueError):
    """Raised when cleanup would remove implausibly much of the document."""


@dataclass(frozen=True)
class Paragraph:
    """A block of prose, rejoined from the lines it was printed across."""

    page: int
    block: int
    text: str


def hyphenated_vocabulary(lines: list[Line]) -> set[str]:
    """Collect hyphenated compounds that appear *within* a line.

    This is the evidence used to decide whether a word broken across lines
    should keep its hyphen. "by-laws", "take-over" and "receiver-manager" are
    legal terms of art that appear mid-line elsewhere in the same document, so
    the document tells us to preserve them; "Corpo-/rations" never appears
    hyphenated mid-line, so its hyphen is a typesetting artifact.

    Using the document's own vocabulary avoids both a dictionary dependency and
    a hand-maintained list that would be wrong for the next statute.
    """
    vocabulary: set[str] = set()
    for line in lines:
        vocabulary.update(_HYPHEN_COMPOUND.findall(line.text.lower()))
    return vocabulary


def join_wrapped_lines(texts: list[str], vocabulary: set[str]) -> str:
    """Join printed lines into one paragraph, repairing broken words."""
    joined = ""
    for text in texts:
        text = text.strip()
        if not text:
            continue
        if not joined:
            joined = text
            continue

        if joined.endswith("-") and text[:1].islower():
            stem = _TRAILING_WORD.search(joined)
            candidate = f"{stem.group(1) if stem else ''}{text.split(' ')[0]}"
            if candidate.lower() in vocabulary:
                joined += text  # a real compound: "by-" + "laws" -> "by-laws"
            else:
                joined = joined[:-1] + text  # "Corpo-" + "rations"
        else:
            joined += " " + text
    return joined


def _grow_margin_block(ordered: list[Line], seed_index: int) -> set[int]:
    """Expand from a boilerplate line through visually contiguous lines.

    Growth runs in both directions: a header block extends below its repeated
    title line to the Part name and section range, and a footer block extends
    below its first repeated line to the second one.
    """
    included = {seed_index}
    for step in (1, -1):
        index = seed_index
        while 0 <= index + step < len(ordered):
            gap = abs(ordered[index + step].y0 - ordered[index].y0)
            if gap > MAX_BLOCK_GAP:
                break
            index += step
            included.add(index)
    return included


def strip_running_headers(lines: list[Line], page_height: float) -> list[Line]:
    """Remove repeated headers and footers from every page.

    Boilerplate is identified by repetition across pages, then extended through
    the lines visually attached to it. Repetition alone is not enough: the Part
    name in the header ("PART IV Registered Office and Records") only appears on
    the pages of that Part, so it never clears the repetition bar on its own,
    but it is still header text and must go.
    """
    if not lines:
        return []

    pages_by_text: dict[str, set[int]] = defaultdict(set)
    for line in lines:
        pages_by_text[line.text].add(line.page)
    page_count = len({line.page for line in lines})
    repeated = {
        text
        for text, pages in pages_by_text.items()
        if len(pages) >= page_count * REPEAT_SHARE
    }

    by_page: dict[int, list[Line]] = defaultdict(list)
    for line in lines:
        by_page[line.page].append(line)

    kept: list[Line] = []
    for page in sorted(by_page):
        ordered = sorted(by_page[page], key=lambda ln: ln.y0)
        drop: set[int] = set()

        # A seed must both repeat across pages and sit in a page margin. The
        # position requirement means a phrase that happens to recur in body
        # text can never trigger deletion of the text around it.
        for index, line in enumerate(ordered):
            in_margin = (
                line.y0 < page_height * MARGIN_BAND
                or line.y0 > page_height * (1 - MARGIN_BAND)
            )
            if line.text in repeated and in_margin:
                drop |= _grow_margin_block(ordered, index)

        if len(drop) > MAX_MARGIN_LINES:
            raise CleanupError(
                f"Header/footer detection would remove {len(drop)} lines from "
                f"page {page}, above the {MAX_MARGIN_LINES}-line maximum. The "
                f"margin blocks have most likely run into the body text."
            )

        kept.extend(ordered[i] for i in range(len(ordered)) if i not in drop)

    return kept


def find_front_matter(lines: list[Line]) -> set[int]:
    """Return the pages of the cover and table of contents.

    A table of contents is recognised by its shape rather than its wording: it
    pairs bare section numbers with headings, so a large share of its lines are
    nothing but a number. Body pages never are.

    Everything up to and including the last ToC page is treated as front matter,
    which sweeps up the cover page and the publisher's notes ahead of it without
    needing a separate rule for them.
    """
    if not lines:
        return set()

    by_page: dict[int, list[Line]] = defaultdict(list)
    for line in lines:
        by_page[line.page].append(line)

    pages = sorted(by_page)
    search_limit = pages[0] + max(1, int(len(pages) * TOC_SEARCH_FRACTION))

    toc_pages = [
        page
        for page in pages
        if page < search_limit
        and sum(1 for ln in by_page[page] if _NUMBER_ONLY.match(ln.text.strip()))
        >= len(by_page[page]) * TOC_NUMBER_SHARE
    ]
    if not toc_pages:
        return set()

    return {page for page in pages if page <= max(toc_pages)}


def to_paragraphs(lines: list[Line], page_height: float) -> list[Paragraph]:
    """Full cleanup: strip margins and front matter, then rejoin wrapped lines.

    Vocabulary for de-hyphenation is built before the front matter is dropped,
    so that compounds appearing only in the table of contents still count as
    evidence.
    """
    if not lines:
        return []

    vocabulary = hyphenated_vocabulary(lines)
    body = strip_running_headers(lines, page_height)

    front_matter = find_front_matter(body)
    body = [line for line in body if line.page not in front_matter]
    if not body:
        raise CleanupError(
            "Cleanup removed every line of the document. The front-matter or "
            "header detection is matching body content."
        )

    grouped: dict[tuple[int, int], list[Line]] = defaultdict(list)
    for line in body:
        grouped[(line.page, line.block)].append(line)

    paragraphs: list[Paragraph] = []
    for (page, block), block_lines in sorted(grouped.items()):
        ordered = sorted(block_lines, key=lambda ln: (ln.y0, ln.x0))
        text = join_wrapped_lines([ln.text for ln in ordered], vocabulary)
        if text:
            paragraphs.append(Paragraph(page=page, block=block, text=text))
    return paragraphs
