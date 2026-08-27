"""LegalMind: retrieval-augmented question answering over the Canada Business
Corporations Act.

Paths are resolved relative to this package rather than the working directory,
so the tools run correctly from anywhere in the filesystem.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

# The source statute.
ACT_PDF = DATA_DIR / "C-44.pdf"

# Generated at runtime and not checked in: it is derived from the PDF.
EMBEDDING_CACHE = ROOT / "embeddings.json"
