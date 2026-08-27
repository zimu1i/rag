"""Measure retrieval and answer quality.

Usage:
    ./venv/bin/python evaluate.py compare    # ablation across retrievers
    ./venv/bin/python evaluate.py hybrid     # full system, per category
    ./venv/bin/python evaluate.py support    # groundedness signal separation
    ./venv/bin/python evaluate.py refusal    # out-of-scope refusal rate
    ./venv/bin/python evaluate.py warm       # cache query embeddings (needs API key)
"""

from legalmind.evaluation import main

if __name__ == "__main__":
    main()
