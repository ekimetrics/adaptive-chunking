"""Coverage for JinaEmbedder happy path with a mocked httpx.AsyncClient."""

import numpy as np
import pytest

from adaptive_chunking import jina_embedder
from adaptive_chunking.jina_embedder import JinaEmbedder

EMBED_DIM = 4


class _FakeResponse:
    def __init__(self, texts):
        # One item per input text; embeddings are deterministic per index.
        self._texts = texts

    def raise_for_status(self):
        return None

    def json(self):
        data = []
        for i, _ in enumerate(self._texts):
            # Give each vector a distinct non-zero pattern.
            embedding = [float(i + 1)] * EMBED_DIM
            data.append({"index": i, "embedding": embedding})
        return {"data": data}


class _ZeroVectorResponse(_FakeResponse):
    def json(self):
        data = []
        for i, _ in enumerate(self._texts):
            data.append({"index": i, "embedding": [0.0] * EMBED_DIM})
        return {"data": data}


def _make_fake_client(response_cls=_FakeResponse):
    class _FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, url, json=None, headers=None, timeout=None):
            # `json` is the payload dict; "input" holds the batch of texts.
            return response_cls(json["input"])

    return _FakeAsyncClient


@pytest.fixture
def patch_client(monkeypatch):
    def _patch(response_cls=_FakeResponse):
        monkeypatch.setattr(
            jina_embedder.httpx,
            "AsyncClient",
            _make_fake_client(response_cls),
        )

    return _patch


def test_encode_empty_returns_zero_shape(patch_client):
    patch_client()
    emb = JinaEmbedder(api_key="test-key").encode([])
    assert isinstance(emb, np.ndarray)
    assert emb.shape == (0, 1024)


def test_encode_returns_ndarray_shape(patch_client):
    patch_client()
    emb = JinaEmbedder(api_key="test-key").encode(
        ["a", "b"], normalize_embeddings=False
    )
    assert isinstance(emb, np.ndarray)
    assert emb.shape == (2, EMBED_DIM)
    # index 0 -> all 1.0, index 1 -> all 2.0
    assert np.allclose(emb[0], 1.0)
    assert np.allclose(emb[1], 2.0)


def test_encode_normalizes(patch_client):
    patch_client()
    emb = JinaEmbedder(api_key="test-key").encode(
        ["a", "b"], normalize_embeddings=True
    )
    norms = np.linalg.norm(emb, axis=1)
    assert np.allclose(norms, 1.0)


def test_encode_normalize_zero_vector_guarded(patch_client):
    # Zero vectors must not produce NaNs (norm==0 replaced with 1.0).
    patch_client(_ZeroVectorResponse)
    emb = JinaEmbedder(api_key="test-key").encode(
        ["a", "b"], normalize_embeddings=True
    )
    assert not np.isnan(emb).any()
    assert np.allclose(emb, 0.0)


def test_encode_show_progress_bar_prints(patch_client, capsys):
    patch_client()
    JinaEmbedder(api_key="test-key").encode(
        ["a", "b"], show_progress_bar=True, normalize_embeddings=False
    )
    out = capsys.readouterr().out
    assert "Jina API" in out
    assert "Done in" in out


def test_encode_batches_when_over_batch_size(patch_client):
    patch_client()
    texts = [f"t{i}" for i in range(5)]
    emb = JinaEmbedder(api_key="test-key").encode(
        texts, batch_size=2, normalize_embeddings=False
    )
    assert emb.shape == (5, EMBED_DIM)


def test_constructor_uses_env_key(monkeypatch):
    monkeypatch.setenv("JINA_API_KEY", "from-env")
    assert JinaEmbedder().api_key == "from-env"
