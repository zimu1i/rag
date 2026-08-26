"""Tests for embedding, caching, retrieval and generation.

The OpenAI client is mocked throughout: no API key, no network, no cost, and
identical results on every run. The fake returns deterministic vectors derived
from the text, so similarity rankings are predictable and can be asserted
exactly rather than approximately.
"""

import json
from types import SimpleNamespace

import numpy as np
import pytest

import rag
from chunking import Chunk


def vector_for(text):
    """A deterministic stand-in for an embedding.

    Counts a few marker words, so texts sharing markers are close together and
    the expected ranking is obvious by inspection.
    """
    lowered = text.lower()
    return [
        float(lowered.count("director")),
        float(lowered.count("share")),
        float(lowered.count("meeting")),
        1.0,  # keeps every vector non-zero
    ]


class FakeEmbeddings:
    def __init__(self, recorder):
        self.recorder = recorder

    def create(self, model, input):
        self.recorder.append(list(input))
        return SimpleNamespace(
            data=[
                SimpleNamespace(index=i, embedding=vector_for(text))
                for i, text in enumerate(input)
            ]
        )


class FakeChat:
    def __init__(self, recorder):
        self.recorder = recorder

    def create(self, model, messages):
        self.recorder.append({"model": model, "messages": messages})
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="A test answer."))]
        )


class FakeClient:
    """Stands in for openai.OpenAI()."""

    def __init__(self):
        self.embedding_calls = []
        self.chat_calls = []
        self.embeddings = FakeEmbeddings(self.embedding_calls)
        self.chat = SimpleNamespace(completions=FakeChat(self.chat_calls))


@pytest.fixture
def client():
    return FakeClient()


@pytest.fixture
def chunks():
    return [
        Chunk(text="Every director shall act honestly.", section="122", subsection="1"),
        Chunk(text="Shares shall be in registered form.", section="24", subsection="1"),
        Chunk(text="A meeting of shareholders may be held.", section="132", subsection="1"),
    ]


class TestEmbedTexts:
    def test_returns_one_vector_per_text(self, client):
        result = rag.embed_texts(client, ["alpha", "beta", "gamma"])

        assert len(result) == 3

    def test_sends_texts_in_batches(self, client):
        rag.embed_texts(client, [f"text {i}" for i in range(250)], batch_size=100)

        assert [len(batch) for batch in client.embedding_calls] == [100, 100, 50]

    def test_preserves_order_even_if_the_api_returns_shuffled_results(self, client, monkeypatch):
        """Vectors are paired with chunks positionally, so order is load-bearing.

        A shuffled response that went unnoticed would attach every chunk to
        another chunk's vector.
        """
        def shuffled_create(model, input):
            data = [
                SimpleNamespace(index=i, embedding=vector_for(text))
                for i, text in enumerate(input)
            ]
            return SimpleNamespace(data=list(reversed(data)))

        monkeypatch.setattr(client.embeddings, "create", shuffled_create)

        result = rag.embed_texts(client, ["director", "share", "meeting"])

        assert result == [vector_for("director"), vector_for("share"), vector_for("meeting")]

    def test_empty_input_makes_no_calls(self, client):
        assert rag.embed_texts(client, []) == []
        assert client.embedding_calls == []


class TestFingerprint:
    def test_same_chunks_give_the_same_fingerprint(self, chunks):
        assert rag.fingerprint(chunks) == rag.fingerprint(list(chunks))

    def test_changing_chunk_text_changes_the_fingerprint(self, chunks):
        altered = [Chunk(text="Different text.", section="122", subsection="1")] + chunks[1:]

        assert rag.fingerprint(altered) != rag.fingerprint(chunks)

    def test_adding_a_chunk_changes_the_fingerprint(self, chunks):
        assert rag.fingerprint(chunks + [Chunk(text="Extra.")]) != rag.fingerprint(chunks)


class TestCache:
    def test_round_trip_preserves_chunks_and_metadata(self, client, chunks, tmp_path):
        path = tmp_path / "cache.json"
        embeddings = rag.embed_texts(client, [c.text for c in chunks])

        rag.save_cache(chunks, embeddings, path)
        loaded_chunks, loaded_embeddings = rag.load_cache(path)

        assert [c.text for c in loaded_chunks] == [c.text for c in chunks]
        assert [c.citation for c in loaded_chunks] == ["s. 122(1)", "s. 24(1)", "s. 132(1)"]
        assert loaded_embeddings == embeddings

    def test_missing_cache_returns_none(self, tmp_path):
        assert rag.load_cache(tmp_path / "absent.json") is None

    def test_stale_cache_is_rejected(self, client, chunks, tmp_path):
        """The bug this guards against.

        Embeddings are stored positionally. If chunking changes and the cache
        does not, every chunk pairs with the wrong vector and retrieval quietly
        degrades with no error anywhere.
        """
        path = tmp_path / "cache.json"
        rag.save_cache(chunks, rag.embed_texts(client, [c.text for c in chunks]), path)

        # Simulate a chunking change: same count, different text.
        data = json.loads(path.read_text())
        data["chunks"][0]["text"] = "Chunking changed underneath the cache."
        path.write_text(json.dumps(data))

        assert rag.load_cache(path) is None

    def test_rebuilds_when_the_cache_does_not_match(self, client, chunks, tmp_path):
        path = tmp_path / "cache.json"
        rag.save_cache(chunks[:1], rag.embed_texts(client, [chunks[0].text]), path)
        client.embedding_calls.clear()

        _, embeddings = rag.get_embeddings(client, chunks, path)

        assert len(embeddings) == 3
        assert client.embedding_calls, "expected a rebuild"

    def test_uses_the_cache_when_it_matches(self, client, chunks, tmp_path):
        path = tmp_path / "cache.json"
        rag.save_cache(chunks, rag.embed_texts(client, [c.text for c in chunks]), path)
        client.embedding_calls.clear()

        rag.get_embeddings(client, chunks, path)

        assert client.embedding_calls == [], "should not re-embed a matching cache"


class TestRetrieval:
    def test_normalises_to_unit_vectors(self):
        matrix = rag.to_matrix([[3.0, 4.0], [1.0, 0.0]])

        assert np.allclose(np.linalg.norm(matrix, axis=1), 1.0)

    def test_handles_a_zero_vector_without_dividing_by_zero(self):
        matrix = rag.to_matrix([[0.0, 0.0]])

        assert not np.isnan(matrix).any()

    def test_ranks_the_most_similar_chunk_first(self, client, chunks):
        matrix = rag.to_matrix(rag.embed_texts(client, [c.text for c in chunks]))

        results = rag.find_chunks(vector_for("director duties"), chunks, matrix, top_k=3)

        assert results[0][1].citation == "s. 122(1)"

    def test_returns_requested_number_of_results(self, client, chunks):
        matrix = rag.to_matrix(rag.embed_texts(client, [c.text for c in chunks]))

        assert len(rag.find_chunks(vector_for("share"), chunks, matrix, top_k=2)) == 2

    def test_scores_are_descending(self, client, chunks):
        matrix = rag.to_matrix(rag.embed_texts(client, [c.text for c in chunks]))

        scores = [s for s, _ in rag.find_chunks(vector_for("meeting"), chunks, matrix, top_k=3)]

        assert scores == sorted(scores, reverse=True)

    def test_cosine_of_identical_vectors_is_one(self, client, chunks):
        matrix = rag.to_matrix([vector_for(chunks[0].text)])

        score, _ = rag.find_chunks(vector_for(chunks[0].text), chunks, matrix, top_k=1)[0]

        assert score == pytest.approx(1.0)


class TestGeneration:
    def test_context_labels_every_excerpt_with_its_citation(self, chunks):
        context = rag.build_context([(0.87, chunks[0]), (0.61, chunks[1])])

        assert "[s. 122(1), relevance 0.87]" in context
        assert "[s. 24(1), relevance 0.61]" in context

    def test_answer_question_passes_context_and_returns_the_reply(self, client, chunks):
        matrix = rag.to_matrix(rag.embed_texts(client, [c.text for c in chunks]))

        answer = rag.answer_question(client, "director duties?", chunks, matrix, verbose=False)

        assert answer == "A test answer."
        sent = client.chat_calls[0]["messages"][1]["content"]
        assert "s. 122(1)" in sent
        assert "director duties?" in sent

    def test_system_prompt_requires_citations_and_refusal(self, client, chunks):
        matrix = rag.to_matrix(rag.embed_texts(client, [c.text for c in chunks]))

        rag.answer_question(client, "anything", chunks, matrix, verbose=False)

        system = client.chat_calls[0]["messages"][0]["content"]
        assert "Cite the provision" in system
        assert "say so plainly" in system

    def test_uses_the_configured_model(self, client, chunks):
        matrix = rag.to_matrix(rag.embed_texts(client, [c.text for c in chunks]))

        rag.answer_question(client, "anything", chunks, matrix, verbose=False)

        assert client.chat_calls[0]["model"] == rag.CHAT_MODEL


@pytest.mark.slow
def test_build_chunks_runs_the_real_pipeline():
    chunks = rag.build_chunks()

    assert len(chunks) > 900
    assert all(c.citation != "(uncited)" for c in chunks)
