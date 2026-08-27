"""
Checks that keep the system from answering when it should not.

For legal information a confident wrong answer is worse than no answer, so there
are two independent guards, aimed at two different failures.

**Retrieval failed.** Nothing relevant came back, and any answer would be the
model drawing on training data rather than the Act. Detected before generation
from a support score, and cheap.

**Generation failed.** The model cited a provision that was never retrieved.
This is the signature failure of legal AI -- fabricated citations are what got
lawyers sanctioned in Mata v. Avianca -- and it is fully deterministic to catch:
every citation in the answer either appears in the retrieved set or it does not.
No model call, no judgement, just set membership.

Note what is *not* claimed here. Neither check verifies that the answer
faithfully paraphrases the provisions it cites. A model can cite s. 122(1)
correctly and still misdescribe it, and nothing in this module would notice.
Catching that needs a verifier pass, which is a different mechanism with
different costs.

The support threshold is deliberately absent until it is calibrated against
`eval_negatives.json`; a constant with no measured headroom on either side is
guesswork, and this project has made that mistake once already.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

# Citations as they appear in generated answers: "s. 122", "s. 122(1)",
# "s. 2.1(3)". The model is instructed to use the labels it was given, which are
# produced by Chunk.citation, so the two formats match by construction.
_ANSWER_CITATION = re.compile(r"\bs\.\s*(\d+(?:\.\d+)?)(?:\s*\((\d+)\))?")


def support_score(query_embedding, chunk_embeddings) -> float:
    """How close is the question to the closest thing retrieved?

    Cosine similarity is used rather than the fused ranking score because
    reciprocal rank fusion produces values around 0.016 that are meaningless in
    isolation and not comparable between queries, and because provisions found
    by citation lookup carry no score at all. Cosine is bounded and behaves the
    same way for every retrieval path.

    Returns 0.0 when nothing was retrieved, which reads as "no support".
    """
    if len(chunk_embeddings) == 0:
        return 0.0

    query = np.asarray(query_embedding, dtype=np.float32)
    matrix = np.asarray(chunk_embeddings, dtype=np.float32)

    query_norm = max(float(np.linalg.norm(query)), 1e-12)
    matrix_norms = np.maximum(np.linalg.norm(matrix, axis=1), 1e-12)

    similarities = (matrix @ query) / (matrix_norms * query_norm)
    return float(similarities.max())


def cited_provisions(answer: str) -> set[tuple[str, str | None]]:
    """Extract every provision an answer claims to rely on."""
    return {
        (match.group(1), match.group(2)) for match in _ANSWER_CITATION.finditer(answer)
    }


def _format(section: str, subsection: str | None) -> str:
    return f"s. {section}({subsection})" if subsection else f"s. {section}"


@dataclass(frozen=True)
class CitationAudit:
    """How each provision an answer cites relates to what was retrieved."""

    grounded: list[str]
    cross_referenced: list[str]
    unsupported: list[str]


def audit_citations(answer: str, retrieved) -> CitationAudit:
    """Sort an answer's citations into three kinds.

    The middle category exists because of a case this check actually caught.
    Asked to explain s. 190(1), the model cited s. 173 -- which had not been
    retrieved. It was not inventing it: s. 190(1) itself reads "amend its
    articles under section 173 or 174", so the reference came straight out of
    the supplied text. But s. 173's own content was never retrieved, so anything
    the answer says *about* s. 173 is unverified.

    Calling that a fabrication would be wrong, and ignoring it would be worse.
    So:

    - **grounded**: the provision was retrieved, and claims about it can be
      checked against the excerpt.
    - **cross_referenced**: not retrieved, but named inside a provision that
      was. The model is repeating the Act rather than inventing, yet the
      system never saw what that provision says.
    - **unsupported**: neither retrieved nor mentioned anywhere in the retrieved
      text. This is the dangerous one -- the reference came from outside the
      excerpts entirely.

    A citation to a section whose *other* subsections were retrieved counts as
    unsupported rather than a cross-reference: inventing s. 122(9) while holding
    s. 122(1) is a fabricated subsection, not a reference to elsewhere.
    """
    corpus = " ".join(chunk.text for chunk in retrieved)
    grounded, cross_referenced, unsupported = [], [], []

    for section, subsection in cited_provisions(answer):
        label = _format(section, subsection)
        same_section = [c for c in retrieved if c.section == section]

        if any(subsection is None or c.subsection == subsection for c in same_section):
            grounded.append(label)
        elif same_section:
            # We hold this section but not that subsection: invented.
            unsupported.append(label)
        elif re.search(rf"\b{re.escape(section)}\b", corpus):
            cross_referenced.append(label)
        else:
            unsupported.append(label)

    return CitationAudit(
        grounded=sorted(grounded),
        cross_referenced=sorted(cross_referenced),
        unsupported=sorted(unsupported),
    )


def unsupported_citations(answer: str, retrieved) -> list[str]:
    """Citations that came from outside the excerpts entirely.

    The strict subset of `audit_citations`: a non-empty result means the model
    produced a provision reference that appears nowhere in what it was given,
    which for a legal tool is the error that matters most.
    """
    return audit_citations(answer, retrieved).unsupported
