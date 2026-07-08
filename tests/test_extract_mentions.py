"""Coverage for extract_mentions.find_mentions_per_origin.

The real ``extract_entity_pronoun_pairs`` needs a spaCy model, so it is
monkeypatched on the ``adaptive_chunking.metrics`` module (the driver imports it
lazily via ``from .metrics import extract_entity_pronoun_pairs`` at call time).
The coref solver is a simple fake.
"""

import json

import pandas as pd

from adaptive_chunking import metrics
from adaptive_chunking.extract_mentions import find_mentions_per_origin

ENGLISH = "Alice went to the market. She bought apples and then she went home. " * 3
GERMAN = "Der Hund laeuft sehr schnell durch den gruenen Wald. " * 12


class FakeCoref:
    def __init__(self):
        self.calls = []

    def find_mentions(self, text):
        self.calls.append(text)
        # one cluster with two mentions (entity span + pronoun span)
        return [[(0, 4), (26, 28)]]


def _fake_pairs(text, clusters, spacy_model):
    # return one entity-pronoun pair
    return [[(0, 4), (26, 28)]]


def _write_parsed(parsed_dir, name, full_text):
    parsed_dir.mkdir(parents=True, exist_ok=True)
    doc = {
        "document_name": name,
        "full_text": full_text,
        "pages": {"1": full_text},
        "split_points": [],
        "titles": [],
    }
    (parsed_dir / f"{name}.json").write_text(json.dumps(doc))


def _models():
    return {"coref_solver": FakeCoref(), "spacy_model": object()}


def test_normal_english_doc(tmp_path, monkeypatch):
    monkeypatch.setattr(metrics, "extract_entity_pronoun_pairs", _fake_pairs)
    parsed_dir = tmp_path / "parsed"
    out_dir = tmp_path / "out"
    _write_parsed(parsed_dir, "doc1", ENGLISH)

    models = _models()
    find_mentions_per_origin(
        parsed_docs_dir=parsed_dir,
        models=models,
        output_dir=out_dir,
        skip_non_english=True,
    )

    doc_pq = out_dir / "doc1.parquet"
    assert doc_pq.exists()
    df = pd.read_parquet(doc_pq)
    assert set(df.columns) == {"doc_name", "mentions", "entity_pron_mentions"}
    assert df["doc_name"].iloc[0] == "doc1"
    assert len(df["entity_pron_mentions"].iloc[0]) == 1

    perf_pq = out_dir / "performances" / "mentions_performance.parquet"
    assert perf_pq.exists()
    perf_df = pd.read_parquet(perf_pq)
    assert set(perf_df.columns) == {"doc_name", "time"}
    assert perf_df["doc_name"].tolist() == ["doc1"]
    # the coref solver was actually invoked
    assert models["coref_solver"].calls == [ENGLISH]


def test_skip_non_english(tmp_path, monkeypatch):
    monkeypatch.setattr(metrics, "extract_entity_pronoun_pairs", _fake_pairs)
    parsed_dir = tmp_path / "parsed"
    out_dir = tmp_path / "out"
    _write_parsed(parsed_dir, "de", GERMAN)

    models = _models()
    find_mentions_per_origin(
        parsed_docs_dir=parsed_dir,
        models=models,
        output_dir=out_dir,
        skip_non_english=True,
    )

    # non-English doc skipped: no per-doc parquet, coref never called
    assert not (out_dir / "de.parquet").exists()
    assert models["coref_solver"].calls == []
    # empty performances file still written
    perf_pq = out_dir / "performances" / "mentions_performance.parquet"
    assert perf_pq.exists()
    perf_df = pd.read_parquet(perf_pq)
    assert len(perf_df) == 0


def test_non_english_processed_when_filter_disabled(tmp_path, monkeypatch):
    monkeypatch.setattr(metrics, "extract_entity_pronoun_pairs", _fake_pairs)
    parsed_dir = tmp_path / "parsed"
    out_dir = tmp_path / "out"
    _write_parsed(parsed_dir, "de", GERMAN)

    models = _models()
    find_mentions_per_origin(
        parsed_docs_dir=parsed_dir,
        models=models,
        output_dir=out_dir,
        skip_non_english=False,  # filter disabled -> processed anyway
    )

    assert (out_dir / "de.parquet").exists()
    assert models["coref_solver"].calls == [GERMAN]


def test_mixed_docs(tmp_path, monkeypatch):
    monkeypatch.setattr(metrics, "extract_entity_pronoun_pairs", _fake_pairs)
    parsed_dir = tmp_path / "parsed"
    out_dir = tmp_path / "out"
    _write_parsed(parsed_dir, "eng", ENGLISH)
    _write_parsed(parsed_dir, "de", GERMAN)

    find_mentions_per_origin(
        parsed_docs_dir=parsed_dir,
        models=_models(),
        output_dir=out_dir,
        skip_non_english=True,
    )

    assert (out_dir / "eng.parquet").exists()
    assert not (out_dir / "de.parquet").exists()
    perf_df = pd.read_parquet(out_dir / "performances" / "mentions_performance.parquet")
    assert perf_df["doc_name"].tolist() == ["eng"]
