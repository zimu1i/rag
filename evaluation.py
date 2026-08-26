"""
Measure retrieval quality against a labelled question set.

Without this, "the results seem worse" is the only available verdict on a change
to retrieval, and tuning becomes a matter of taste. With it, adding BM25 or
changing fusion weights produces a number that can be compared.

Two decisions here are deliberate:

1.  The harness takes a *retriever* -- any callable from question to ranked
    chunks -- rather than calling the embedder itself. Semantic, BM25 and the
    hybrid merge are then measured by identical code, so a comparison between
    them cannot be an artifact of how each was evaluated.

2.  Results are reported per category as well as overall. An average hides the
    failure that matters: retrieval that is strong on paraphrased questions and
    blind to section numbers scores respectably overall while being useless for
    the query a lawyer is most likely to type.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

DEFAULT_EVAL_SET = "eval_set.json"
DEFAULT_K_VALUES = (1, 3, 5)

# Query embeddings for the eval set, cached to disk.
#
# The eval questions are fixed, so re-embedding them on every run costs money,
# needs an API key, and makes retrieval experiments non-reproducible. With this
# cached, comparing fusion variants is free, offline and deterministic -- which
# matters because tuning a retriever means running the eval many times.
QUERY_CACHE_FILE = "query_embeddings.json"


@dataclass(frozen=True)
class Target:
    """One specific chunk, identified by citation and optionally by term."""

    citation: str
    term: str | None = None

    def matches(self, chunk) -> bool:
        if chunk.citation != self.citation:
            return False
        # ~39 definition chunks share the citation s. 2(1), so a definition
        # question is only satisfied by the chunk for that specific term.
        return self.term is None or chunk.heading == self.term

    def describe(self) -> str:
        return self.citation + (f" [{self.term}]" if self.term else "")


@dataclass(frozen=True)
class Expectation:
    """One requirement of a question, satisfied by any of its targets.

    Alternatives exist because a provision's procedural hook and its substance
    are often separate subsections. s. 241(1) says only that "a complainant may
    apply to a court for an order under this section"; the grounds a reader
    actually needs -- "oppressive", "unfairly prejudicial" -- are in s. 241(2).
    Retrieving either is a good outcome, so scoring one of them as a miss would
    understate retrieval quality.

    This is distinct from listing several expectations, which requires ALL of
    them: alternatives are "either will do", separate expectations are "both
    are needed".
    """

    targets: tuple[Target, ...]

    @classmethod
    def of(cls, citation: str, term: str | None = None) -> "Expectation":
        return cls((Target(citation, term),))

    @classmethod
    def either(cls, targets) -> "Expectation":
        return cls(tuple(targets))

    def matches(self, chunk) -> bool:
        return any(target.matches(chunk) for target in self.targets)

    def describe(self) -> str:
        return " or ".join(target.describe() for target in self.targets)


@dataclass(frozen=True)
class EvalCase:
    question: str
    category: str
    expected: tuple[Expectation, ...]


@dataclass(frozen=True)
class CaseResult:
    case: EvalCase
    ranks: tuple[int | None, ...]  # rank of each expectation, None if unretrieved
    # What actually came back. A miss says retrieval failed; this says how --
    # whether the right provision ranked just outside k, or whether retrieval
    # went somewhere else entirely. Those need different fixes.
    retrieved: tuple = ()

    @property
    def first_rank(self) -> int | None:
        found = [r for r in self.ranks if r is not None]
        return min(found) if found else None

    def hit_at(self, k: int) -> bool:
        """Did any expected provision appear in the top k?"""
        rank = self.first_rank
        return rank is not None and rank <= k

    def coverage_at(self, k: int) -> float:
        """What fraction of the expected provisions appeared in the top k?

        Distinct from hit_at for multi-provision questions, where retrieving one
        of two required provisions is a partial answer, not a success.
        """
        if not self.ranks:
            return 0.0
        found = sum(1 for r in self.ranks if r is not None and r <= k)
        return found / len(self.ranks)

    @property
    def reciprocal_rank(self) -> float:
        rank = self.first_rank
        return 1.0 / rank if rank else 0.0


def load_eval_set(path: str = DEFAULT_EVAL_SET) -> list[EvalCase]:
    with open(path) as handle:
        data = json.load(handle)
    def parse(entry: dict) -> Expectation:
        # {"citation": ...} requires that chunk; {"any_of": [...]} accepts any.
        if "any_of" in entry:
            return Expectation.either(
                Target(citation=alt["citation"], term=alt.get("term"))
                for alt in entry["any_of"]
            )
        return Expectation.of(entry["citation"], entry.get("term"))

    return [
        EvalCase(
            question=case["question"],
            category=case["category"],
            expected=tuple(parse(entry) for entry in case["expected"]),
        )
        for case in data["cases"]
    ]


DEFAULT_NEGATIVES = "eval_negatives.json"


def load_negatives(path: str = DEFAULT_NEGATIVES) -> list[dict]:
    """Load out-of-scope questions used to calibrate the groundedness check.

    Kept separate from the retrieval eval set: these have no correct provision,
    so scoring them as retrieval misses would corrupt recall and MRR.
    """
    with open(path) as handle:
        return json.load(handle)["cases"]


def evaluate_case(case: EvalCase, retrieved) -> CaseResult:
    """Locate each expected provision in a ranked result list."""
    ranks: list[int | None] = []
    for expectation in case.expected:
        rank = None
        for position, chunk in enumerate(retrieved, start=1):
            if expectation.matches(chunk):
                rank = position
                break
        ranks.append(rank)
    return CaseResult(case=case, ranks=tuple(ranks), retrieved=tuple(retrieved))


def evaluate(cases, retriever, k_values=DEFAULT_K_VALUES, depth=None):
    """Run every case through a retriever.

    `retriever` maps (question, k) to ranked chunks. Retrieval depth defaults to
    the largest k being measured.
    """
    depth = depth or max(k_values)
    return [evaluate_case(case, retriever(case.question, depth)) for case in cases]


def summarise(results, k_values=DEFAULT_K_VALUES) -> dict:
    """Aggregate results overall and by category."""

    def metrics(subset):
        if not subset:
            return {}
        summary = {"n": len(subset)}
        for k in k_values:
            summary[f"hit@{k}"] = sum(r.hit_at(k) for r in subset) / len(subset)
            summary[f"coverage@{k}"] = sum(r.coverage_at(k) for r in subset) / len(subset)
        summary["mrr"] = sum(r.reciprocal_rank for r in subset) / len(subset)
        return summary

    categories = sorted({r.case.category for r in results})
    return {
        "overall": metrics(results),
        "by_category": {
            category: metrics([r for r in results if r.case.category == category])
            for category in categories
        },
    }


def format_report(results, k_values=DEFAULT_K_VALUES) -> str:
    """A plain-text report, including every individual miss."""
    summary = summarise(results, k_values)
    lines = []

    header = f"{'category':<16}{'n':>4}" + "".join(f"{f'hit@{k}':>9}" for k in k_values)
    header += f"{'MRR':>8}"
    lines.append(header)
    lines.append("-" * len(header))

    for category, stats in summary["by_category"].items():
        row = f"{category:<16}{stats['n']:>4}"
        row += "".join(f"{stats[f'hit@{k}']:>9.2f}" for k in k_values)
        row += f"{stats['mrr']:>8.2f}"
        lines.append(row)

    overall = summary["overall"]
    lines.append("-" * len(header))
    row = f"{'OVERALL':<16}{overall['n']:>4}"
    row += "".join(f"{overall[f'hit@{k}']:>9.2f}" for k in k_values)
    row += f"{overall['mrr']:>8.2f}"
    lines.append(row)

    misses = [r for r in results if not r.hit_at(max(k_values))]
    if misses:
        lines.append("")
        lines.append(f"Misses (nothing expected in top {max(k_values)}):")
        for result in misses:
            wanted = ", ".join(e.describe() for e in result.case.expected)
            lines.append(f"  [{result.case.category}] {result.case.question}")
            lines.append(f"      expected: {wanted}")
            for position, chunk in enumerate(result.retrieved[:3], start=1):
                label = chunk.heading or chunk.text[:40]
                lines.append(f"      got {position}: {chunk.citation:<12} {label}")

    return "\n".join(lines)


def semantic_retriever(client, chunks, matrix):
    """Build a retriever backed by the current embedding index."""
    import rag

    def retrieve(question, k):
        query = rag.embed_query(client, question)
        return [chunk for _, chunk in rag.find_chunks(query, chunks, matrix, top_k=k)]

    return retrieve


def load_query_cache(path: str = QUERY_CACHE_FILE) -> dict:
    """Load cached query embeddings, or an empty cache.

    A cache built with a different embedding model is discarded rather than
    mixed, since vectors from two models are not comparable.
    """
    import rag

    if not os.path.exists(path):
        return {}
    with open(path) as handle:
        data = json.load(handle)
    if data.get("model") != rag.EMBEDDING_MODEL:
        return {}
    return data.get("embeddings", {})


def save_query_cache(embeddings: dict, path: str = QUERY_CACHE_FILE) -> None:
    import rag

    with open(path, "w") as handle:
        json.dump({"model": rag.EMBEDDING_MODEL, "embeddings": embeddings}, handle)


def warm_query_cache(client, questions, path: str = QUERY_CACHE_FILE) -> dict:
    """Embed any question not already cached. The only step that needs an API key."""
    import rag

    cache = load_query_cache(path)
    missing = [question for question in questions if question not in cache]
    if missing:
        vectors = rag.embed_texts(client, missing)
        cache.update(dict(zip(missing, vectors)))
        save_query_cache(cache, path)
    return cache


def cached_semantic_retriever(chunks, matrix, path: str = QUERY_CACHE_FILE):
    """Semantic retrieval using cached query embeddings, with no API calls.

    Raises on an uncached question rather than silently falling back to a live
    call, so a missing entry surfaces as an error instead of an unexpected bill.
    """
    import rag

    cache = load_query_cache(path)

    def retrieve(question, k):
        if question not in cache:
            raise KeyError(
                f"No cached embedding for {question!r}. "
                f"Run `./venv/bin/python evaluation.py warm` first."
            )
        return [chunk for _, chunk in rag.find_chunks(cache[question], chunks, matrix, top_k=k)]

    return retrieve


def bm25_retriever(chunks):
    """Build a keyword retriever. Needs no API calls, so it runs offline."""
    import bm25

    index = bm25.build_index(chunks)

    def retrieve(question, k):
        return [chunk for _, chunk in index.search(question, top_k=k)]

    return retrieve


def hybrid_retriever(chunks, matrix, path: str = QUERY_CACHE_FILE, use_structured=True):
    """Rank fusion over semantic and keyword retrieval, plus citation lookup."""
    import hybrid

    return hybrid.build_retriever(
        chunks,
        cached_semantic_retriever(chunks, matrix, path),
        bm25_retriever(chunks),
        use_structured=use_structured,
    )


def report_support(chunks, embeddings, matrix):
    """Print the support-score distribution for answerable vs out-of-scope questions.

    This is the measurement that decides whether a threshold can separate them
    at all. If the two distributions overlap, no single cutoff will work and the
    check needs a different signal.
    """
    import groundedness

    cache = load_query_cache()
    retriever = hybrid_retriever(chunks, matrix)
    position = {chunk: index for index, chunk in enumerate(chunks)}

    def score(question):
        retrieved = retriever(question, rag.DEFAULT_TOP_K)
        vectors = [embeddings[position[chunk]] for chunk in retrieved]
        return groundedness.support_score(cache[question], vectors), retrieved

    missing = [
        question
        for question in [c.question for c in load_eval_set()]
        + [n["question"] for n in load_negatives()]
        if question not in cache
    ]
    if missing:
        raise SystemExit(
            f"{len(missing)} question(s) not in the query cache. "
            f"Run `./venv/bin/python evaluation.py warm` first."
        )

    print("ANSWERABLE (should be answered)")
    answerable = []
    for case in load_eval_set():
        value, retrieved = score(case.question)
        answerable.append(value)
        print(f"  {value:.3f}  [{case.category:<15}] {case.question[:58]}")

    print("\nOUT OF SCOPE (should be declined)")
    negatives = []
    for negative in sorted(load_negatives(), key=lambda n: n["difficulty"]):
        value, retrieved = score(negative["question"])
        negatives.append(value)
        top = retrieved[0].citation if retrieved else "-"
        print(
            f"  {value:.3f}  [{negative['difficulty']:<10}] "
            f"{negative['question'][:48]:<48} top={top}"
        )

    print(
        f"\nanswerable : min {min(answerable):.3f}  "
        f"median {sorted(answerable)[len(answerable) // 2]:.3f}  max {max(answerable):.3f}"
    )
    print(
        f"out-of-scope: min {min(negatives):.3f}  "
        f"median {sorted(negatives)[len(negatives) // 2]:.3f}  max {max(negatives):.3f}"
    )
    gap = min(answerable) - max(negatives)
    print(
        f"\nseparation : {gap:+.3f} "
        f"({'clean, a threshold exists' if gap > 0 else 'OVERLAP - no single cutoff separates them'})"
    )


def report_refusals(chunks, matrix):
    """Measure how often the model declines a question the Act cannot answer.

    Whether an answer counts as a refusal is decided by whether it cites any
    provision. The system prompt requires a citation for every statement, so an
    answer with no citation is not asserting anything about the Act. That is a
    proxy rather than a definition -- it is printed alongside each answer so the
    classification can be checked by eye rather than trusted.
    """
    import groundedness
    import rag
    from openai import OpenAI

    client = OpenAI()
    retriever = hybrid_retriever(chunks, matrix)

    def ask(question):
        retrieved = retriever(question, rag.DEFAULT_TOP_K)
        result = rag.answer_question(client, question, lambda q, k: retrieved, verbose=False)
        return result.text, groundedness.cited_provisions(result.text), result.audit

    print("OUT OF SCOPE (a refusal is the correct outcome)\n")
    declined = 0
    for negative in sorted(load_negatives(), key=lambda n: n["difficulty"]):
        answer, cites, audit = ask(negative["question"])
        refused = not cites
        declined += refused
        print(f"  [{'DECLINED' if refused else 'ANSWERED'}] {negative['question']}")
        print(f"      {answer.strip()[:190]}")
        if audit.unsupported:
            print(f"      !! cited but never retrieved: {', '.join(audit.unsupported)}")
        print()

    total = len(load_negatives())
    print(f"declined {declined}/{total} out-of-scope questions\n")

    print("ANSWERABLE (a refusal here is a false decline)\n")
    control = load_eval_set()
    false_declines = []
    unsupported_cases = []
    cross_referenced = []
    for case in control:
        answer, cites, audit = ask(case.question)
        if not cites:
            false_declines.append((case, answer))
        if audit.unsupported:
            unsupported_cases.append((case, audit.unsupported))
        if audit.cross_referenced:
            cross_referenced.append((case, audit.cross_referenced))

    print(f"  answered {len(control) - len(false_declines)}/{len(control)}")
    if false_declines:
        print("\n  False declines, with what the model said:\n")
        for case, answer in false_declines:
            print(f"    [{case.category}] {case.question}")
            print(f"      {answer.strip()[:230]}\n")

    print(f"\nfalse declines      : {len(false_declines)}/{len(control)}")
    print(f"unsupported cites   : {len(unsupported_cases)}/{len(control)}"
          "   (cited, and absent from the excerpts entirely)")
    for case, unsupported in unsupported_cases:
        print(f"  !! {case.question} -> {', '.join(unsupported)}")
    print(f"cross-referenced    : {len(cross_referenced)}/{len(control)}"
          "   (named inside a retrieved provision, but its own text was not seen)")
    for case, refs in cross_referenced:
        print(f"   ~ {case.question} -> {', '.join(refs)}")


def main():
    import sys

    import rag

    modes = {
        "semantic",
        "bm25",
        "hybrid",
        "fusion",
        "warm",
        "compare",
        "support",
        "refusal",
    }
    which = sys.argv[1] if len(sys.argv) > 1 else "hybrid"
    if which not in modes:
        raise SystemExit(f"Unknown mode {which!r}. Use one of: {', '.join(sorted(modes))}.")

    cached = rag.load_cache()
    if cached is None:
        raise SystemExit(
            "No usable embedding cache. Run `./venv/bin/python rag.py` first to build it."
        )

    chunks, embeddings = cached
    cases = load_eval_set()

    if which == "warm":
        from openai import OpenAI

        before = len(load_query_cache())
        questions = [case.question for case in cases]
        questions += [negative["question"] for negative in load_negatives()]
        cache = warm_query_cache(OpenAI(), questions)
        print(
            f"Query cache: {len(cache)} questions "
            f"({len(cache) - before} newly embedded) -> {QUERY_CACHE_FILE}"
        )
        print("Evaluation can now run offline with no API key.")
        return

    matrix = rag.to_matrix(embeddings)

    if which == "support":
        report_support(chunks, embeddings, matrix)
        return

    if which == "refusal":
        report_refusals(chunks, matrix)
        return

    retrievers = {
        "semantic": lambda: cached_semantic_retriever(chunks, matrix),
        "bm25": lambda: bm25_retriever(chunks),
        # "fusion" is the ablation: rank fusion with the citation lookup off.
        "fusion": lambda: hybrid_retriever(chunks, matrix, use_structured=False),
        "hybrid": lambda: hybrid_retriever(chunks, matrix),
    }

    if which == "compare":
        print(f"{len(cases)} questions against {len(chunks)} chunks\n")
        header = f"{'retriever':<12}{'hit@1':>8}{'hit@3':>8}{'hit@5':>8}{'MRR':>8}"
        print(header)
        print("-" * len(header))
        for name, factory in retrievers.items():
            stats = summarise(evaluate(cases, factory()))["overall"]
            print(
                f"{name:<12}{stats['hit@1']:>8.2f}{stats['hit@3']:>8.2f}"
                f"{stats['hit@5']:>8.2f}{stats['mrr']:>8.2f}"
            )
        return

    print(f"Retriever: {which}")
    print(f"Evaluating {len(cases)} questions against {len(chunks)} chunks...\n")
    print(format_report(evaluate(cases, retrievers[which]())))


if __name__ == "__main__":
    main()
