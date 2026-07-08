"""Coverage for compute_metrics.compute_metrics_per_origin using a fake embedder.

A fake sentence embedder returns deterministic normalized vectors so no ML
dependency is needed.  Chunk texts are constructed to concatenate back to the
document full_text (required by the metric helpers that locate chunks).
"""

import json

import numpy as np
import pandas as pd
import pytest

from adaptive_chunking.compute_metrics import compute_metrics_per_origin

CHUNK1 = "Alice went to the market early in the morning. "
CHUNK2 = "She bought fresh apples and then she walked back home."
FULL_TEXT = CHUNK1 + CHUNK2


class FakeModel:
    """Deterministic stand-in for a SentenceTransformer."""

    def __init__(self):
        self.encode_calls = 0

    def encode(self, texts, batch_size=32, show_progress_bar=False,
               convert_to_numpy=True, normalize_embeddings=False, **kwargs):
        self.encode_calls += 1
        rows = []
        for t in texts:
            v = np.array(
                [
                    (len(t) % 7) + 1.0,
                    t.count("e") + 1.0,
                    t.count(" ") + 1.0,
                    (len(t) % 3) + 1.0,
                ],
                dtype=np.float64,
            )
            if normalize_embeddings:
                norm = np.linalg.norm(v)
                if norm > 0:
                    v = v / norm
            rows.append(v)
        return np.array(rows, dtype=np.float64)


def _write_chunks(chunks_dir, rows):
    """rows: list of dicts with doc_name, method, chunk_text, chunk_len."""
    chunks_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_parquet(chunks_dir / "chunks.parquet")


def _chunk_rows(doc_name, method, chunks):
    return [
        {
            "doc_name": doc_name,
            "method": method,
            "chunk_text": c,
            "chunk_len": len(c.split()),
        }
        for c in chunks
    ]


def _write_parsed(parsed_dir, name="doc1", full_text=FULL_TEXT, split_points=None):
    parsed_dir.mkdir(parents=True, exist_ok=True)
    doc = {
        "document_name": name,
        "full_text": full_text,
        "pages": {"1": full_text},
        "split_points": split_points or [],
        "titles": [],
    }
    (parsed_dir / f"{name}.json").write_text(json.dumps(doc))


def _write_mentions(mentions_dir, doc_name, pairs):
    mentions_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({"doc_name": [doc_name], "entity_pron_mentions": [pairs]})
    df.to_parquet(mentions_dir / f"{doc_name}.parquet")


def _dirs(tmp_path):
    return (
        tmp_path / "chunks",
        tmp_path / "mentions",
        tmp_path / "parsed",
        tmp_path / "out",
    )


def test_basic_metrics_written(tmp_path):
    chunks_dir, mentions_dir, parsed_dir, out_dir = _dirs(tmp_path)
    _write_chunks(chunks_dir, _chunk_rows("doc1", "recursive", [CHUNK1, CHUNK2]))
    _write_parsed(parsed_dir)
    # a pair crossing the chunk boundary at len(CHUNK1)
    b = len(CHUNK1)
    _write_mentions(mentions_dir, "doc1", [[[0, 4], [b + 2, b + 4]]])

    model = FakeModel()
    compute_metrics_per_origin(
        chunks_dir=chunks_dir,
        mentions_dir=mentions_dir,
        parsed_docs_dir=parsed_dir,
        models={"sentence_embedder": model},
        output_dir=out_dir,
        batch_size=8,
    )

    metrics_df = pd.read_parquet(out_dir / "chunking_metrics.parquet")
    perf_df = pd.read_parquet(out_dir / "metrics_performance.parquet")

    assert set(metrics_df.columns) == {
        "doc_name",
        "chunking_method",
        "metric_name",
        "score",
    }
    assert (metrics_df["doc_name"] == "doc1").all()
    metric_names = set(metrics_df["metric_name"].unique())
    # all metric families are present
    assert {
        "size_compliance",
        "block_integrity",
        "intrachunk_cohesion",
        "document_contextual_coherence",
        "references_completeness",
        "avg_chunk_tokens",
        "num_chunks",
    }.issubset(metric_names)

    # references_completeness should be a real score (pair crosses a boundary -> 0.0)
    ref_score = metrics_df[
        metrics_df["metric_name"] == "references_completeness"
    ]["score"].iloc[0]
    assert ref_score == pytest.approx(0.0)

    assert set(perf_df.columns) == {"doc_name", "metric", "time"}
    assert model.encode_calls > 0


def test_references_completeness_none_branch(tmp_path):
    """Empty mentions list -> compute_filtered_missing_ref_error returns None."""
    chunks_dir, mentions_dir, parsed_dir, out_dir = _dirs(tmp_path)
    _write_chunks(chunks_dir, _chunk_rows("doc1", "recursive", [CHUNK1, CHUNK2]))
    _write_parsed(parsed_dir)
    _write_mentions(mentions_dir, "doc1", [])  # empty -> None score

    compute_metrics_per_origin(
        chunks_dir=chunks_dir,
        mentions_dir=mentions_dir,
        parsed_docs_dir=parsed_dir,
        models={"sentence_embedder": FakeModel()},
        output_dir=out_dir,
    )

    metrics_df = pd.read_parquet(out_dir / "chunking_metrics.parquet")
    ref_rows = metrics_df[metrics_df["metric_name"] == "references_completeness"]
    assert ref_rows["score"].isna().all()


def test_multiple_docs_incremental_concat(tmp_path):
    """Two docs exercise the 'existing file -> concat' incremental save path."""
    chunks_dir, mentions_dir, parsed_dir, out_dir = _dirs(tmp_path)
    rows = _chunk_rows("doc1", "recursive", [CHUNK1, CHUNK2]) + _chunk_rows(
        "doc2", "recursive", [CHUNK1, CHUNK2]
    )
    _write_chunks(chunks_dir, rows)
    _write_parsed(parsed_dir, name="doc1")
    _write_parsed(parsed_dir, name="doc2")
    _write_mentions(mentions_dir, "doc1", [])
    _write_mentions(mentions_dir, "doc2", [])

    compute_metrics_per_origin(
        chunks_dir=chunks_dir,
        mentions_dir=mentions_dir,
        parsed_docs_dir=parsed_dir,
        models={"sentence_embedder": FakeModel()},
        output_dir=out_dir,
    )

    metrics_df = pd.read_parquet(out_dir / "chunking_metrics.parquet")
    assert set(metrics_df["doc_name"].unique()) == {"doc1", "doc2"}
    perf_df = pd.read_parquet(out_dir / "metrics_performance.parquet")
    assert set(perf_df["doc_name"].unique()) == {"doc1", "doc2"}


def test_resumability_skips_existing_doc(tmp_path):
    """Pre-seed chunking_metrics.parquet with doc1 -> it is skipped."""
    chunks_dir, mentions_dir, parsed_dir, out_dir = _dirs(tmp_path)
    _write_chunks(chunks_dir, _chunk_rows("doc1", "recursive", [CHUNK1, CHUNK2]))
    _write_parsed(parsed_dir)
    _write_mentions(mentions_dir, "doc1", [])

    # pre-create output with doc1 already present
    out_dir.mkdir(parents=True, exist_ok=True)
    seed = pd.DataFrame(
        [{
            "doc_name": "doc1",
            "chunking_method": "recursive",
            "metric_name": "size_compliance",
            "score": 0.5,
        }]
    )
    seed.to_parquet(out_dir / "chunking_metrics.parquet")

    model = FakeModel()
    compute_metrics_per_origin(
        chunks_dir=chunks_dir,
        mentions_dir=mentions_dir,
        parsed_docs_dir=parsed_dir,
        models={"sentence_embedder": model},
        output_dir=out_dir,
    )

    # doc1 was skipped -> embedder never invoked, file unchanged
    assert model.encode_calls == 0
    metrics_df = pd.read_parquet(out_dir / "chunking_metrics.parquet")
    assert len(metrics_df) == 1
    assert metrics_df["score"].iloc[0] == 0.5
    # no perf file written because nothing was processed
    assert not (out_dir / "metrics_performance.parquet").exists()


def test_multiple_methods_per_doc(tmp_path):
    chunks_dir, mentions_dir, parsed_dir, out_dir = _dirs(tmp_path)
    rows = _chunk_rows("doc1", "recursive", [CHUNK1, CHUNK2]) + _chunk_rows(
        "doc1", "semantic", [FULL_TEXT]
    )
    _write_chunks(chunks_dir, rows)
    _write_parsed(parsed_dir)
    _write_mentions(mentions_dir, "doc1", [])

    compute_metrics_per_origin(
        chunks_dir=chunks_dir,
        mentions_dir=mentions_dir,
        parsed_docs_dir=parsed_dir,
        models={"sentence_embedder": FakeModel()},
        output_dir=out_dir,
    )

    metrics_df = pd.read_parquet(out_dir / "chunking_metrics.parquet")
    methods = set(metrics_df["chunking_method"].unique())
    assert methods == {"recursive", "semantic"}
