"""
Chunk a statute along its own structure.

A statute is already divided into units by the people who drafted it. Sections
and subsections are written to be read, cited and amended independently, and
they are also the unit a legal question is asked about ("what does s. 122 require
of a director?"). Cutting that text into fixed 500-character windows severs
provisions mid-sentence and discards the section numbers that would let an answer
cite its source.

So chunking here follows the document: one chunk per provision, carrying the
section number, subsection number and marginal note that identify it.

Three complications the text forces, each handled explicitly below:

1.  Provisions run across page breaks. The extractor yields one paragraph per
    page-bound block, so 52 provisions in C-44 arrive as two fragments.

2.  A few provisions are enormous. The largest is 8,619 characters -- a list of
    129 Criminal Code cross-references. Those are split at their own lettered
    item boundaries, not at an arbitrary character offset, and each piece keeps
    the provision's opening words so it still makes sense alone.

3.  The definitions section is a special shape. s. 2(1) is one subsection
    containing ~100 defined terms, each self-contained. Each definition becomes
    its own chunk, which is exactly the granularity "what is an affiliate?"
    needs.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from legalmind.cleanup import Paragraph

# Provisions larger than this are split at their internal item boundaries.
#
# Measured over C-44's 1,156 provisions: median 286 characters, 90th percentile
# 683, 95th 880, 99th 1,587. A 1,200-character budget therefore leaves 97% of
# provisions whole, which is the point -- splitting a provision is the exception,
# not the normal path.
#
# This is a threshold for splitting, not a hard ceiling. A provision with no
# lettered items has no safe internal boundary, so it is left whole rather than
# cut mid-sentence: about 1% of chunks exceed the budget, the largest at ~1,750
# characters. A chunk that misstates the law is a worse outcome than a large one.
MAX_CHUNK_CHARS = 1_200

# End matter that must not be indexed.
#
# A consolidated Act closes with transitional provisions from past amending Acts
# and with amendments that have been passed but are NOT YET LAW. Retrieving the
# latter would let the system answer with text that does not currently apply,
# which for legal information is the most damaging error available to it.
EXCLUDED_SECTIONS = ("RELATED PROVISIONS", "AMENDMENTS NOT IN FORCE")

# The Schedule restarts numbering from 1, so its items must not be cited as
# sections of the Act -- without this, Schedule item 1 is indistinguishable from
# s. 1 of the Act itself.
SCHEDULE_HEADING = "SCHEDULE"

# A marginal note ("Issue of shares") is a heading, not prose. Real headings in
# C-44 run to about 60 characters; the limit is set above that and combined with
# a punctuation test, since headings never end in sentence or clause punctuation.
MAX_HEADING_CHARS = 70

_SECTION_WITH_SUB = re.compile(r"^(\d+(?:\.\d+)?)\s*\((\d+)\)\s+\S")
_SECTION = re.compile(r"^(\d+(?:\.\d+)?)\s+[A-Z(]")
_SUBSECTION = re.compile(r"^\((\d+)\)\s+\S")
_ITEM = re.compile(r"^\(([a-z]+(?:\.\d+)?)\)\s+\S")
_CITATION = re.compile(r"^(R\.S\.|S\.C\.|\d{4}), ")
_PART = re.compile(r"^PART\s+([IVXLC]+)")
# A repeal marker can cover a single provision ("(2) [Repealed, ...]"), a range
# ("(3) and (4) [Repealed, ...]"), or a whole section ("24 [Repealed, ...]").
_REPEALED = re.compile(r"^(?:\([^)]+\)|\d+(?:\.\d+)?).{0,40}?\[Repealed")
# A defined term followed by "means" or "includes". Only trusted inside a
# definitions subsection -- see _opens_definitions.
_DEFINITION = re.compile(r"^([A-Za-z][^.]{0,60}?)\s+(means|includes)\b")
_OPENS_DEFINITIONS = re.compile(
    r"\bIn this (Act|Part|section|subsection)\b|\bfollowing definitions apply\b"
)
_MID_SENTENCE_END = (".", ";", ":", "!", "?")


@dataclass(frozen=True)
class Chunk:
    """One provision, with the metadata needed to cite it."""

    text: str
    section: str | None = None
    subsection: str | None = None
    heading: str | None = None
    part: str | None = None
    pages: tuple[int, ...] = field(default_factory=tuple)

    @property
    def citation(self) -> str:
        """A human-readable citation, e.g. "s. 24(1)"."""
        if self.part == SCHEDULE_HEADING:
            return f"Schedule, item {self.section}" if self.section else "Schedule"
        if self.section is None:
            return "(uncited)"
        if self.subsection is None:
            return f"s. {self.section}"
        return f"s. {self.section}({self.subsection})"


def merge_page_breaks(paragraphs: list[Paragraph]) -> list[Paragraph]:
    """Rejoin provisions split across a page boundary.

    A fragment is recognised by position rather than wording: it is the first
    thing on the next page, the previous paragraph stopped mid-sentence, and it
    begins with a lowercase word that starts no new structure. Restricting this
    to page boundaries matters -- within a page, block segmentation is already
    correct, so a broader rule could only do harm.
    """
    if not paragraphs:
        return []

    merged = [paragraphs[0]]
    for current in paragraphs[1:]:
        previous = merged[-1]
        continues = (
            current.page == previous.page + 1
            and not previous.text.endswith(_MID_SENTENCE_END)
            and current.text[:1].islower()
            and not _ITEM.match(current.text)
            and not _SUBSECTION.match(current.text)
        )
        if continues:
            merged[-1] = Paragraph(
                page=previous.page,
                block=previous.block,
                text=f"{previous.text} {current.text}",
            )
        else:
            merged.append(current)
    return merged


def is_heading(text: str) -> bool:
    """True for a marginal note or division heading."""
    if len(text) > MAX_HEADING_CHARS:
        return False
    if text.endswith(_MID_SENTENCE_END) or text.endswith(","):
        return False
    return not (
        _SECTION_WITH_SUB.match(text)
        or _SECTION.match(text)
        or _SUBSECTION.match(text)
        or _ITEM.match(text)
        or _CITATION.match(text)
    )


def _split_oversized(text: str, limit: int) -> list[str]:
    """Split an over-long provision at its lettered item boundaries.

    Each piece keeps the provision's opening words, because a bare "(z.001)
    subsection 121(1) (frauds on the government)" retrieved on its own says
    nothing about what it is a list of.
    """
    if len(text) <= limit:
        return [text]

    # Split before each "(a) ", "(b) ", "(z.001) " marker.
    pieces = re.split(r"(?=\([a-z]+(?:\.\d+)?\)\s)", text)
    stem = pieces[0].strip()
    items = [piece.strip() for piece in pieces[1:] if piece.strip()]
    if not items:
        return [text]

    chunks: list[str] = []
    current: list[str] = []
    current_len = len(stem)
    for item in items:
        if current and current_len + len(item) > limit:
            chunks.append(" ".join([stem, *current]))
            current = []
            current_len = len(stem)
        current.append(item)
        current_len += len(item)
    if current:
        chunks.append(" ".join([stem, *current]))
    return chunks


def chunk_paragraphs(paragraphs: list[Paragraph]) -> list[Chunk]:
    """Group cleaned paragraphs into citable chunks."""
    paragraphs = merge_page_breaks(paragraphs)

    chunks: list[Chunk] = []
    part: str | None = None
    section: str | None = None
    subsection: str | None = None
    pending_heading: str | None = None
    in_definitions = False

    # The provision currently being accumulated.
    body: list[str] = []
    meta: dict = {}

    def flush() -> None:
        nonlocal body, meta
        if not body:
            return
        text = " ".join(body).strip()
        heading = meta.get("heading")
        for piece in _split_oversized(text, MAX_CHUNK_CHARS):
            # A definition's heading is its defined term, which the text already
            # opens with; repeating it would waste tokens and read as a stutter.
            if heading and not piece.startswith(heading):
                full = f"{heading}\n{piece}"
            else:
                full = piece
            chunks.append(
                Chunk(
                    text=full,
                    section=meta.get("section"),
                    subsection=meta.get("subsection"),
                    heading=heading,
                    part=meta.get("part"),
                    pages=tuple(sorted(set(meta.get("pages", ())))),
                )
            )
        body = []
        meta = {}

    def start(text: str, page: int, heading: str | None) -> None:
        nonlocal body, meta
        flush()
        body = [text]
        meta = {
            "section": section,
            "subsection": subsection,
            "heading": heading,
            "part": part,
            "pages": [page],
        }

    excluded = False
    for paragraph in paragraphs:
        text = paragraph.text
        banner = text.strip().upper()

        # Everything from the first end-matter banner onward is dropped.
        if banner in EXCLUDED_SECTIONS:
            flush()
            excluded = True
            continue
        if excluded:
            continue

        if banner == SCHEDULE_HEADING:
            flush()
            part = SCHEDULE_HEADING
            section = subsection = None
            pending_heading = None
            continue

        # Structural furniture that should not be indexed.
        if _CITATION.match(text) or _REPEALED.match(text):
            continue

        part_match = _PART.match(text)
        if part_match:
            flush()
            part = part_match.group(1)
            pending_heading = None
            continue

        section_sub = _SECTION_WITH_SUB.match(text)
        subsection_only = _SUBSECTION.match(text)
        section_only = _SECTION.match(text)

        if section_sub:
            section, subsection = section_sub.group(1), section_sub.group(2)
            in_definitions = bool(_OPENS_DEFINITIONS.search(text))
            start(text, paragraph.page, pending_heading)
            pending_heading = None
            continue

        if section_only:
            section, subsection = section_only.group(1), None
            in_definitions = bool(_OPENS_DEFINITIONS.search(text))
            start(text, paragraph.page, pending_heading)
            pending_heading = None
            continue

        if subsection_only:
            subsection = subsection_only.group(1)
            in_definitions = bool(_OPENS_DEFINITIONS.search(text))
            start(text, paragraph.page, pending_heading)
            pending_heading = None
            continue

        # Inside a definitions subsection, every defined term is its own chunk.
        if in_definitions:
            definition = _DEFINITION.match(text)
            if definition:
                start(text, paragraph.page, definition.group(1).strip())
                continue

        if is_heading(text):
            # A marginal note introduces what follows, so hold it for the next
            # provision rather than attaching it to the one just ended.
            pending_heading = text
            continue

        if body:
            body.append(text)
            meta.setdefault("pages", []).append(paragraph.page)
        # Text before the first section (the long title) is not a provision and
        # is deliberately dropped.

    flush()
    return chunks
