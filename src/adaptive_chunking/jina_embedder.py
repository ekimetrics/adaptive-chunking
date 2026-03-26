"""Async-batched Jina Embeddings API wrapper with the same interface as SentenceTransformer.

Drops in as `sentence_embedder` in compute_metrics_per_origin — implements the
`encode(texts, batch_size, normalize_embeddings, convert_to_numpy, show_progress_bar)`
signature expected by metrics.py.
"""

import asyncio
import os
import random
import time
from typing import Sequence

import httpx
import numpy as np

JINA_API_URL = "https://api.jina.ai/v1/embeddings"
JINA_MODEL   = "jina-embeddings-v3"
MAX_BATCH    = 256   # Jina allows up to 2048
MAX_RETRIES  = 6
RETRY_DELAY  = 2.0   # seconds between retries on 429
MAX_CHARS    = 20000  # jina-embeddings-v3 max ~8192 tokens; conservative truncation at ~5k tokens


class JinaEmbedder:
    """Drop-in replacement for SentenceTransformer using the Jina Embeddings API.

    Usage:
        embedder = JinaEmbedder()  # reads JINA_API_KEY from env
        emb = embedder.encode(texts, normalize_embeddings=True)  # → np.ndarray (N, D)
    """

    def __init__(self, api_key: str | None = None, max_concurrent: int = 3):
        self.api_key = api_key or os.environ["JINA_API_KEY"]
        self.max_concurrent = max_concurrent

    # ------------------------------------------------------------------
    # Internal async helpers
    # ------------------------------------------------------------------

    async def _embed_batch_async(
        self,
        client: httpx.AsyncClient,
        texts: list[str],
        semaphore: asyncio.Semaphore,
        normalize: bool,
    ) -> np.ndarray:
        # Truncate any text exceeding the model's effective token limit
        texts = [t[:MAX_CHARS] if len(t) > MAX_CHARS else t for t in texts]
        payload = {
            "model": JINA_MODEL,
            "input": texts,
            "normalized": normalize,
            "embedding_type": "float",
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        for attempt in range(MAX_RETRIES):
            async with semaphore:
                try:
                    resp = await client.post(
                        JINA_API_URL, json=payload, headers=headers, timeout=120.0
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    vecs = [item["embedding"] for item in sorted(data["data"], key=lambda x: x["index"])]
                    await asyncio.sleep(0.5)  # gentle throttle after each success
                    return np.array(vecs, dtype=np.float32)
                except httpx.HTTPStatusError as e:
                    if e.response.status_code < 500 and e.response.status_code != 429:
                        raise  # 4xx (not rate-limit) are permanent, don't retry
                    if attempt == MAX_RETRIES - 1:
                        raise
                    wait = RETRY_DELAY * (2 ** attempt) + random.uniform(0, 2.0)
                    print(f"  Jina {e.response.status_code}, waiting {wait:.1f}s (attempt {attempt+1}/{MAX_RETRIES}) ...")
                    await asyncio.sleep(wait)
                except httpx.RequestError:
                    if attempt == MAX_RETRIES - 1:
                        raise
                    wait = RETRY_DELAY * (2 ** attempt) + random.uniform(0, 2.0)
                    print(f"  Jina request error, waiting {wait:.1f}s (attempt {attempt+1}/{MAX_RETRIES}) ...")
                    await asyncio.sleep(wait)

    async def _embed_all_async(
        self, texts: list[str], batch_size: int, normalize: bool
    ) -> np.ndarray:
        batches = [texts[i : i + batch_size] for i in range(0, len(texts), batch_size)]
        semaphore = asyncio.Semaphore(self.max_concurrent)
        limits = httpx.Limits(max_connections=self.max_concurrent, max_keepalive_connections=self.max_concurrent)
        async with httpx.AsyncClient(limits=limits) as client:
            tasks = [
                self._embed_batch_async(client, batch, semaphore, normalize)
                for batch in batches
            ]
            results = await asyncio.gather(*tasks)
        return np.concatenate(results, axis=0)

    # ------------------------------------------------------------------
    # Public encode() — same signature as SentenceTransformer.encode()
    # ------------------------------------------------------------------

    def encode(
        self,
        sentences: Sequence[str],
        batch_size: int = MAX_BATCH,
        show_progress_bar: bool = False,
        convert_to_numpy: bool = True,
        normalize_embeddings: bool = True,
        **kwargs,
    ) -> np.ndarray:
        texts = list(sentences)
        if not texts:
            return np.zeros((0, 1024), dtype=np.float32)

        effective_batch = min(batch_size, MAX_BATCH)
        n_batches = (len(texts) + effective_batch - 1) // effective_batch

        if show_progress_bar:
            print(f"  Jina API: {len(texts)} texts → {n_batches} batches of ≤{effective_batch}")

        t0 = time.time()
        embeddings = asyncio.run(
            self._embed_all_async(texts, effective_batch, normalize_embeddings)
        )
        elapsed = time.time() - t0

        if show_progress_bar:
            print(f"  Done in {elapsed:.1f}s ({len(texts)/elapsed:.0f} texts/s)")

        if normalize_embeddings:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1.0, norms)
            embeddings = embeddings / norms

        return embeddings
