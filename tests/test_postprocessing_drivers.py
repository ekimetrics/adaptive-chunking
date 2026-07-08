"""Coverage for the *_from_df driver functions in adaptive_chunking.postprocessing.

These build a tiny parsed-doc JSON + chunks parquet on disk and run the drivers,
asserting the output parquets are written and readable. ``full_text`` is the exact
concatenation of the chunk rows so the gap-recovery assert inside the drivers passes.
"""

import json

import pandas as pd
import pytest

from adaptive_chunking.postprocessing import (
    split_oversized_chunks_from_df,
    merge_small_chunks_from_df,
)

# full_text == concatenation of the two chunk rows below
FULL_TEXT = "Hello world."
CHUNK_ROWS = ["Hello ", "world."]


def _identity(chunks):
    """No-op regularizer: keeps chunks unchanged (deterministic)."""
    return chunks


def _write_parsed_doc(parsed_dir, doc_name="doc1"):
    parsed_dir.mkdir(parents=True, exist_ok=True)
    doc = {
        "document_name": doc_name,
        "full_text": FULL_TEXT,
        # JSON keys are strings; get_page_info int()-casts them
        "pages": {"1": FULL_TEXT},
        "split_points": [],
        "titles": [],
    }
    (parsed_dir / f"{doc_name}.json").write_text(json.dumps(doc))


def _write_chunks_parquet(path, methods=("mymethod", "other"), doc_name="doc1"):
    rows = []
    for method in methods:
        for chunk_text in CHUNK_ROWS:
            rows.append({"doc_name": doc_name, "method": method, "chunk_text": chunk_text})
    pd.DataFrame(rows).to_parquet(path)


@pytest.fixture
def parsed_dir(tmp_path):
    d = tmp_path / "parsed"
    _write_parsed_doc(d)
    return d


@pytest.fixture
def chunks_path(tmp_path):
    p = tmp_path / "chunks_in.parquet"
    _write_chunks_parquet(p)
    return p


# --------------------------------------------------------------------------
# split_oversized_chunks_from_df
# --------------------------------------------------------------------------
class TestSplitOversizedFromDf:
    def test_writes_readable_parquets(self, parsed_dir, chunks_path, tmp_path):
        out = tmp_path / "out"
        split_oversized_chunks_from_df(
            parsed_dir,
            chunks_path,
            out,
            methods_to_be_regularized={"mymethod"},
            split_oversized_func=_identity,
            count_tokens_func=len,
        )

        chunks_df = pd.read_parquet(out / "chunks.parquet")
        perf_df = pd.read_parquet(out / "performances.parquet")

        assert (out / "chunks.parquet").exists()
        assert (out / "performances.parquet").exists()

        # both the regularized and the passthrough ("other") methods are present
        assert set(chunks_df["method"]) == {"mymethod", "other"}
        assert set(chunks_df["type"]) == {"no_oversizing"}
        # page info computed from string-keyed pages
        assert [list(p) for p in chunks_df["chunk_pages"]] == [[1]] * len(chunks_df)
        # chunk_len uses count_tokens_func=len
        assert chunks_df["chunk_len"].tolist() == [len(t) for t in chunks_df["chunk_text"]]
        assert set(perf_df["method"]) == {"mymethod", "other"}

    def test_no_parsed_docs_returns_none(self, tmp_path, chunks_path):
        empty = tmp_path / "empty_parsed"
        empty.mkdir()
        out = tmp_path / "out"
        result = split_oversized_chunks_from_df(
            empty,
            chunks_path,
            out,
            methods_to_be_regularized={"mymethod"},
            split_oversized_func=_identity,
            count_tokens_func=len,
        )
        assert result is None
        # nothing written when there are no parsed docs
        assert not (out / "chunks.parquet").exists()

    def test_merge_with_existing_parquet(self, parsed_dir, chunks_path, tmp_path):
        out = tmp_path / "out"
        # first run writes the output parquets
        split_oversized_chunks_from_df(
            parsed_dir, chunks_path, out,
            methods_to_be_regularized={"mymethod"},
            split_oversized_func=_identity, count_tokens_func=len,
        )
        # second run (default replace_all_results=False) exercises the
        # merge-with-existing branch for both chunks and performances parquets
        split_oversized_chunks_from_df(
            parsed_dir, chunks_path, out,
            methods_to_be_regularized={"mymethod"},
            split_oversized_func=_identity, count_tokens_func=len,
        )
        merged = pd.read_parquet(out / "chunks.parquet")
        assert set(merged["method"]) == {"mymethod", "other"}
        assert len(merged) > 4  # existing "other" rows preserved + new rows added

    def test_corrupt_existing_parquet_falls_back_to_overwrite(
        self, parsed_dir, chunks_path, tmp_path
    ):
        out = tmp_path / "out"
        out.mkdir()
        # pre-create unreadable parquet files so the merge-with-existing read
        # raises and the except-handler overwrite path is taken
        (out / "chunks.parquet").write_text("not a parquet file")
        (out / "performances.parquet").write_text("not a parquet file")
        split_oversized_chunks_from_df(
            parsed_dir, chunks_path, out,
            methods_to_be_regularized={"mymethod"},
            split_oversized_func=_identity, count_tokens_func=len,
        )
        # files were overwritten with valid parquet output
        df = pd.read_parquet(out / "chunks.parquet")
        assert set(df["method"]) == {"mymethod", "other"}

    def test_replace_all_results_overwrites(self, parsed_dir, chunks_path, tmp_path):
        out = tmp_path / "out"
        # first run creates output
        split_oversized_chunks_from_df(
            parsed_dir, chunks_path, out,
            methods_to_be_regularized={"mymethod"},
            split_oversized_func=_identity, count_tokens_func=len,
        )
        first_len = len(pd.read_parquet(out / "chunks.parquet"))

        # replace_all_results=True overwrites rather than merging
        split_oversized_chunks_from_df(
            parsed_dir, chunks_path, out,
            methods_to_be_regularized={"mymethod"},
            split_oversized_func=_identity, count_tokens_func=len,
            replace_all_results=True,
        )
        second = pd.read_parquet(out / "chunks.parquet")
        assert len(second) == first_len  # not appended/grown
        assert set(second["method"]) == {"mymethod", "other"}


# --------------------------------------------------------------------------
# merge_small_chunks_from_df
# --------------------------------------------------------------------------
class TestMergeSmallFromDf:
    def test_writes_readable_parquets(self, parsed_dir, chunks_path, tmp_path):
        out = tmp_path / "out"
        merge_small_chunks_from_df(
            parsed_dir,
            chunks_path,
            out,
            methods_to_be_regularized={"mymethod"},
            merge_small_chunks_func=_identity,
            count_tokens_func=len,
        )
        chunks_df = pd.read_parquet(out / "chunks.parquet")
        perf_df = pd.read_parquet(out / "performances.parquet")

        assert set(chunks_df["method"]) == {"mymethod", "other"}
        assert set(chunks_df["type"]) == {"no_small_chunks"}
        assert set(perf_df["method"]) == {"mymethod", "other"}

    def test_no_parsed_docs_returns_none(self, tmp_path, chunks_path):
        empty = tmp_path / "empty_parsed"
        empty.mkdir()
        out = tmp_path / "out"
        result = merge_small_chunks_from_df(
            empty, chunks_path, out,
            methods_to_be_regularized={"mymethod"},
            merge_small_chunks_func=_identity, count_tokens_func=len,
        )
        assert result is None

    def test_merge_with_existing_parquet(self, parsed_dir, chunks_path, tmp_path):
        out = tmp_path / "out"
        # first run writes existing chunks.parquet (mymethod + other)
        merge_small_chunks_from_df(
            parsed_dir, chunks_path, out,
            methods_to_be_regularized={"mymethod"},
            merge_small_chunks_func=_identity, count_tokens_func=len,
        )
        # second run (replace_all_results=False, default) hits the merge-with-existing
        # branch: rows for "mymethod" are replaced, "other" rows preserved + re-added
        merge_small_chunks_from_df(
            parsed_dir, chunks_path, out,
            methods_to_be_regularized={"mymethod"},
            merge_small_chunks_func=_identity, count_tokens_func=len,
        )
        merged = pd.read_parquet(out / "chunks.parquet")
        # "other" survives from the existing parquet; "mymethod" re-added
        assert set(merged["method"]) == {"mymethod", "other"}
        # existing "other" rows preserved AND new rows concatenated -> grew
        assert len(merged) > 4

    def test_corrupt_existing_parquet_falls_back_to_overwrite(
        self, parsed_dir, chunks_path, tmp_path
    ):
        out = tmp_path / "out"
        out.mkdir()
        (out / "chunks.parquet").write_text("not a parquet file")
        (out / "performances.parquet").write_text("not a parquet file")
        merge_small_chunks_from_df(
            parsed_dir, chunks_path, out,
            methods_to_be_regularized={"mymethod"},
            merge_small_chunks_func=_identity, count_tokens_func=len,
        )
        df = pd.read_parquet(out / "chunks.parquet")
        assert set(df["method"]) == {"mymethod", "other"}

    def test_replace_all_results_overwrites(self, parsed_dir, chunks_path, tmp_path):
        out = tmp_path / "out"
        merge_small_chunks_from_df(
            parsed_dir, chunks_path, out,
            methods_to_be_regularized={"mymethod"},
            merge_small_chunks_func=_identity, count_tokens_func=len,
        )
        merge_small_chunks_from_df(
            parsed_dir, chunks_path, out,
            methods_to_be_regularized={"mymethod"},
            merge_small_chunks_func=_identity, count_tokens_func=len,
            replace_all_results=True,
        )
        df = pd.read_parquet(out / "chunks.parquet")
        assert len(df) == 4  # 2 methods x 2 chunks, no merge with existing
