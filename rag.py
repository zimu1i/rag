"""
Muzi Li
2026.06.25
Data Science Student at UWaterloo

Retrieval-augmented question answering over the Canada Business Corporations Act
(R.S.C., 1985, c. C-44).

The pipeline is:

    ingest.py    two-column bilingual PDF -> English lines with coordinates
    cleanup.py   lines -> clean prose paragraphs
    chunking.py  paragraphs -> citable chunks, one per provision
    rag.py       embed, retrieve, answer                     <- this file

Retrieval is currently semantic only. BM25 and the hybrid merge are not built
yet; see README.md for the current status of each component.
"""

import hashlib
import json
import os

import numpy as np
from openai import OpenAI

from chunking import Chunk, chunk_paragraphs
from cleanup import to_paragraphs
from ingest import extract_bilingual

PDF_PATH = "C-44.pdf"
CACHE_FILE = "embeddings.json"
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"

# The embeddings endpoint accepts a list of inputs per call. Embedding the Act
# one chunk at a time costs ~1,125 round trips; in batches of 100 it costs 12.
# The ceiling here is the request's total token budget, not the batch count, and
# 100 chunks of ~300 characters is a small fraction of it.
BATCH_SIZE = 100


# --------------------------------------------------------------------------
# Building chunks
# --------------------------------------------------------------------------


def build_chunks(pdf_path=PDF_PATH):
    """Run the ingestion pipeline and return citable chunks."""
    document = extract_bilingual(pdf_path)
    paragraphs = to_paragraphs(document.left, document.page_height)
    return chunk_paragraphs(paragraphs)


# --------------------------------------------------------------------------
# Embeddings
# --------------------------------------------------------------------------


def embed_texts(client, texts, batch_size=BATCH_SIZE, progress=False):
    """Embed many texts, in batches."""
    embeddings = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        response = client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
        # Sort by the index the API echoes back rather than trusting response
        # order, so a batch can never be silently misaligned with its texts.
        embeddings.extend(item.embedding for item in sorted(response.data, key=lambda d: d.index))
        if progress:
            done = min(start + batch_size, len(texts))
            print(f"  embedded {done}/{len(texts)}")
    return embeddings


def embed_query(client, text):
    """Embed a single query."""
    return embed_texts(client, [text])[0]


def fingerprint(chunks):
    """Identify the exact chunk set an embedding cache was built from.

    The cache stores vectors positionally, so if chunking changes and the cache
    does not, every chunk silently pairs with another chunk's vector and
    retrieval degrades without erroring. Hashing the chunk texts together with
    the model name turns that silent corruption into a cheap rebuild.
    """
    digest = hashlib.sha256()
    digest.update(EMBEDDING_MODEL.encode())
    for chunk in chunks:
        digest.update(chunk.text.encode())
        digest.update(b"\0")
    return digest.hexdigest()


def _chunk_to_dict(chunk):
    return {
        "text": chunk.text,
        "section": chunk.section,
        "subsection": chunk.subsection,
        "heading": chunk.heading,
        "part": chunk.part,
        "pages": list(chunk.pages),
    }


def _chunk_from_dict(data):
    return Chunk(
        text=data["text"],
        section=data["section"],
        subsection=data["subsection"],
        heading=data["heading"],
        part=data["part"],
        pages=tuple(data["pages"]),
    )


def save_cache(chunks, embeddings, path=CACHE_FILE):
    with open(path, "w") as handle:
        json.dump(
            {
                "fingerprint": fingerprint(chunks),
                "model": EMBEDDING_MODEL,
                "chunks": [_chunk_to_dict(chunk) for chunk in chunks],
                "embeddings": embeddings,
            },
            handle,
        )


def load_cache(path=CACHE_FILE):
    """Load cached chunks and embeddings, or None if there is no usable cache."""
    if not os.path.exists(path):
        return None
    with open(path) as handle:
        data = json.load(handle)
    chunks = [_chunk_from_dict(item) for item in data.get("chunks", [])]
    if data.get("fingerprint") != fingerprint(chunks):
        return None
    return chunks, data["embeddings"]


def get_embeddings(client, chunks, path=CACHE_FILE):
    """Return embeddings for these chunks, using the cache when it matches."""
    cached = load_cache(path)
    if cached is not None and fingerprint(cached[0]) == fingerprint(chunks):
        print(f"Loaded {len(cached[0])} cached embeddings.")
        return cached

    print(f"Embedding {len(chunks)} chunks (no usable cache)...")
    embeddings = embed_texts(client, [chunk.text for chunk in chunks], progress=True)
    save_cache(chunks, embeddings, path)
    print(f"Saved to {path}")
    return chunks, embeddings


# --------------------------------------------------------------------------
# Retrieval
# --------------------------------------------------------------------------


def to_matrix(embeddings):
    """Stack embeddings into an L2-normalised matrix.

    Normalising once, at load, means each query is a single matrix multiply:
    for unit vectors the cosine similarity is just the dot product. The original
    version recomputed both norms inside a Python loop for all ~1,100 chunks on
    every question.
    """
    matrix = np.asarray(embeddings, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)


def find_chunks(query_embedding, chunks, matrix, top_k=3):
    """Return the top_k (score, chunk) pairs by cosine similarity."""
    query = np.asarray(query_embedding, dtype=np.float32)
    query = query / max(float(np.linalg.norm(query)), 1e-12)

    scores = matrix @ query
    # argpartition finds the top k without sorting all ~1,100 scores.
    top = np.argpartition(-scores, min(top_k, len(scores) - 1))[:top_k]
    top = top[np.argsort(-scores[top])]
    return [(float(scores[i]), chunks[i]) for i in top]


# --------------------------------------------------------------------------
# Generation
# --------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a careful legal research assistant answering questions about the Canada Business Corporations Act.

Answer only from the provided excerpts. Each excerpt is labelled with the provision it comes from.

- Cite the provision for every statement you make, using the labels given, e.g. "s. 122(1)".
- If the excerpts do not answer the question, say so plainly instead of inferring.
- Do not rely on knowledge of the Act beyond the excerpts.
- This is information about the legislation, not legal advice."""


def build_context(results):
    """Format retrieved chunks for the model, labelled by citation."""
    parts = []
    for score, chunk in results:
        parts.append(f"[{chunk.citation}, relevance {score:.2f}]\n{chunk.text}")
    return "\n\n".join(parts)


def answer_question(client, question, chunks, matrix, top_k=3, verbose=True):
    results = find_chunks(embed_query(client, question), chunks, matrix, top_k)

    if verbose:
        print("--- Retrieved ---")
        for score, chunk in results:
            print(f"  {score:.3f}  [{chunk.citation}]  {chunk.text[:90]}...")
        print()

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Excerpts:\n{build_context(results)}\n\nQuestion: {question}",
            },
        ],
    )
    return response.choices[0].message.content


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main():
    client = OpenAI()

    print("Building chunks from the Act...")
    chunks = build_chunks()
    print(f"{len(chunks)} chunks.")

    chunks, embeddings = get_embeddings(client, chunks)
    matrix = to_matrix(embeddings)

    print("\n=== LegalMind ready ===")
    print("Ask a question about the Canada Business Corporations Act.")
    print("Type 'quit' to exit.\n")

    while True:
        question = input("Your question: ").strip()
        if question.lower() in {"quit", "exit"}:
            break
        if not question:
            continue
        print(f"\nAnswer: {answer_question(client, question, chunks, matrix)}\n")


# Nothing runs on import: the test suite imports this module, and building the
# index or opening a prompt as a side effect of an import would make it
# untestable.
if __name__ == "__main__":
    main()
