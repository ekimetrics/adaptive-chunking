"""Coverage for split_documents.split_documents_from_dir using fake splitters.

The driver is async and is exercised via ``asyncio.run``.  All chunks produced
by the fakes concatenate back to the document full_text so that the internal
``check_chunk_gaps`` assertions pass.
"""

import asyncio
import json

import pandas as pd

from adaptive_chunking.split_documents import split_documents_from_dir

# Two pages whose concatenation IS the full text (required by the 'page' method
# which asserts the page list covers the whole document).
PAGE1 = "The cat sat on the mat. The dog ran fast. " * 4
PAGE2 = "Birds fly in the sky. Fish swim in the sea. " * 4
FULL_TEXT = PAGE1 + PAGE2

# Clearly non-English (German) text for the language-filter skip path.
GERMAN = "Der Hund laeuft sehr schnell durch den gruenen Wald. " * 12


def word_count(text):
    return len(text.split())


def _split_in_half(text):
    """Split text into two parts on a space boundary; parts concatenate to text."""
    if not text:
        return []
    mid = len(text) // 2
    idx = text.find(" ", mid)
    if idx == -1:
        idx = text.rfind(" ", 0, mid)
    if idx == -1:
        return [text]
    return [text[:idx], text[idx:]]


class FakeSyncSplitter:
    def split_text(self, text):
        return _split_in_half(text)


class FakeAsyncSplitter:
    async def split_text(self, text):
        await asyncio.sleep(0)
        return _split_in_half(text)


def _write_doc(parsed_dir, name="doc1", full_text=FULL_TEXT, pages=None, titles=None):
    parsed_dir.mkdir(parents=True, exist_ok=True)
    if pages is None:
        # default single page equal to full_text
        pages = {"1": full_text}
    doc = {
        "document_name": name,
        "full_text": full_text,
        "pages": pages,
        "split_points": [],
        "titles": titles or [],
    }
    (parsed_dir / f"{name}.json").write_text(json.dumps(doc))
    return doc


def _run(**kwargs):
    kwargs.setdefault("count_tokens_func", word_count)
    return asyncio.run(split_documents_from_dir(**kwargs))


EXPECTED_CHUNK_COLS = {
    "doc_name",
    "method",
    "type",
    "chunk_index",
    "chunk_text",
    "chunk_pages",
    "titles_context",
    "chunk_len",
}


def test_empty_dir_returns_none(tmp_path):
    parsed = tmp_path / "parsed"
    parsed.mkdir()
    out = tmp_path / "out"
    result = _run(
        parsed_docs_dir=parsed,
        sync_splitters={"recursive": FakeSyncSplitter()},
        async_splitters={},
        output_dir=out,
        skip_non_english=False,
    )
    assert result is None
    assert not (out / "chunks.parquet").exists()


def test_standard_sync_splitter_writes_parquets(tmp_path):
    parsed = tmp_path / "parsed"
    _write_doc(parsed, pages={"1": PAGE1, "2": PAGE2})
    out = tmp_path / "out"

    _run(
        parsed_docs_dir=parsed,
        sync_splitters={"recursive": FakeSyncSplitter()},
        async_splitters={},
        output_dir=out,
        skip_non_english=False,
    )

    chunks_df = pd.read_parquet(out / "chunks.parquet")
    perf_df = pd.read_parquet(out / "performances.parquet")

    assert EXPECTED_CHUNK_COLS.issubset(set(chunks_df.columns))
    assert set(chunks_df["method"].unique()) == {"recursive"}
    assert (chunks_df["doc_name"] == "doc1").all()
    # chunks concatenate back to full_text
    recombined = "".join(
        chunks_df.sort_values("chunk_index")["chunk_text"].tolist()
    )
    assert recombined == FULL_TEXT
    assert set(perf_df.columns) == {"doc_name", "method", "method_type", "time"}
    assert (perf_df["method_type"] == "sync").all()


def test_page_and_reckitt_and_standard_methods(tmp_path):
    parsed = tmp_path / "parsed"
    _write_doc(parsed, pages={"1": PAGE1, "2": PAGE2})
    out = tmp_path / "out"

    _run(
        parsed_docs_dir=parsed,
        sync_splitters={
            "page": object(),  # presence triggers page method (value unused)
            "reckitt": FakeSyncSplitter(),
            "recursive": FakeSyncSplitter(),
        },
        async_splitters={},
        output_dir=out,
        skip_non_english=False,
    )

    chunks_df = pd.read_parquet(out / "chunks.parquet")
    methods = set(chunks_df["method"].unique())
    assert methods == {"page", "reckitt", "recursive"}

    # page method => exactly the two page texts
    page_rows = chunks_df[chunks_df["method"] == "page"].sort_values("chunk_index")
    assert page_rows["chunk_text"].tolist() == [PAGE1, PAGE2]

    # each method's chunks concatenate to the full text
    for method in methods:
        rows = chunks_df[chunks_df["method"] == method].sort_values("chunk_index")
        assert "".join(rows["chunk_text"].tolist()) == FULL_TEXT


def test_async_splitter(tmp_path):
    parsed = tmp_path / "parsed"
    _write_doc(parsed, pages={"1": PAGE1, "2": PAGE2})
    out = tmp_path / "out"

    _run(
        parsed_docs_dir=parsed,
        sync_splitters={},
        async_splitters={"semantic": FakeAsyncSplitter()},
        output_dir=out,
        skip_non_english=False,
    )

    chunks_df = pd.read_parquet(out / "chunks.parquet")
    perf_df = pd.read_parquet(out / "performances.parquet")
    assert set(chunks_df["method"].unique()) == {"semantic"}
    assert (perf_df["method_type"] == "async").all()


def test_sync_and_async_combined(tmp_path):
    parsed = tmp_path / "parsed"
    _write_doc(parsed, pages={"1": PAGE1, "2": PAGE2})
    out = tmp_path / "out"

    _run(
        parsed_docs_dir=parsed,
        sync_splitters={"recursive": FakeSyncSplitter()},
        async_splitters={"semantic": FakeAsyncSplitter()},
        output_dir=out,
        skip_non_english=False,
    )

    perf_df = pd.read_parquet(out / "performances.parquet")
    types_by_method = dict(zip(perf_df["method"], perf_df["method_type"]))
    assert types_by_method["recursive"] == "sync"
    assert types_by_method["semantic"] == "async"


def test_skip_empty_full_text(tmp_path):
    parsed = tmp_path / "parsed"
    _write_doc(parsed, name="empty", full_text="", pages={})
    _write_doc(parsed, name="good", full_text=FULL_TEXT, pages={"1": PAGE1, "2": PAGE2})
    out = tmp_path / "out"

    _run(
        parsed_docs_dir=parsed,
        sync_splitters={"recursive": FakeSyncSplitter()},
        async_splitters={},
        output_dir=out,
        skip_non_english=False,
    )

    chunks_df = pd.read_parquet(out / "chunks.parquet")
    assert set(chunks_df["doc_name"].unique()) == {"good"}


def test_language_filter_keeps_english(tmp_path):
    parsed = tmp_path / "parsed"
    _write_doc(parsed, name="eng", full_text=FULL_TEXT, pages={"1": PAGE1, "2": PAGE2})
    out = tmp_path / "out"

    _run(
        parsed_docs_dir=parsed,
        sync_splitters={"recursive": FakeSyncSplitter()},
        async_splitters={},
        output_dir=out,
        skip_non_english=True,  # exercises langdetect on English text -> kept
    )

    chunks_df = pd.read_parquet(out / "chunks.parquet")
    assert set(chunks_df["doc_name"].unique()) == {"eng"}


def test_language_filter_skips_non_english(tmp_path):
    parsed = tmp_path / "parsed"
    _write_doc(parsed, name="de", full_text=GERMAN, pages={"1": GERMAN})
    _write_doc(parsed, name="eng", full_text=FULL_TEXT, pages={"1": PAGE1, "2": PAGE2})
    out = tmp_path / "out"

    _run(
        parsed_docs_dir=parsed,
        sync_splitters={"recursive": FakeSyncSplitter()},
        async_splitters={},
        output_dir=out,
        skip_non_english=True,
    )

    chunks_df = pd.read_parquet(out / "chunks.parquet")
    names = set(chunks_df["doc_name"].unique())
    assert "eng" in names
    assert "de" not in names


def test_replace_all_results_true_overwrites(tmp_path):
    parsed = tmp_path / "parsed"
    _write_doc(parsed, pages={"1": PAGE1, "2": PAGE2})
    out = tmp_path / "out"

    # first run with method A
    _run(
        parsed_docs_dir=parsed,
        sync_splitters={"methodA": FakeSyncSplitter()},
        async_splitters={},
        output_dir=out,
        skip_non_english=False,
    )
    # second run with method B and replace_all_results=True -> only B remains
    _run(
        parsed_docs_dir=parsed,
        sync_splitters={"methodB": FakeSyncSplitter()},
        async_splitters={},
        output_dir=out,
        skip_non_english=False,
        replace_all_results=True,
    )

    chunks_df = pd.read_parquet(out / "chunks.parquet")
    assert set(chunks_df["method"].unique()) == {"methodB"}


def test_merge_with_existing_parquet(tmp_path):
    parsed = tmp_path / "parsed"
    _write_doc(parsed, pages={"1": PAGE1, "2": PAGE2})
    out = tmp_path / "out"

    # first run with method A
    _run(
        parsed_docs_dir=parsed,
        sync_splitters={"methodA": FakeSyncSplitter()},
        async_splitters={},
        output_dir=out,
        skip_non_english=False,
    )
    # second run with method B, replace_all_results=False -> merge keeps both
    _run(
        parsed_docs_dir=parsed,
        sync_splitters={"methodB": FakeSyncSplitter()},
        async_splitters={},
        output_dir=out,
        skip_non_english=False,
        replace_all_results=False,
    )

    chunks_df = pd.read_parquet(out / "chunks.parquet")
    assert set(chunks_df["method"].unique()) == {"methodA", "methodB"}

    perf_df = pd.read_parquet(out / "performances.parquet")
    assert set(perf_df["method"].unique()) == {"methodA", "methodB"}


def test_merge_with_existing_async_method(tmp_path):
    """Merge path while supplying an async splitter (covers async-method update)."""
    parsed = tmp_path / "parsed"
    _write_doc(parsed, pages={"1": PAGE1, "2": PAGE2})
    out = tmp_path / "out"

    # seed with a sync method
    _run(
        parsed_docs_dir=parsed,
        sync_splitters={"recursive": FakeSyncSplitter()},
        async_splitters={},
        output_dir=out,
        skip_non_english=False,
    )
    # add an async method, merge (replace_all_results=False)
    _run(
        parsed_docs_dir=parsed,
        sync_splitters={},
        async_splitters={"semantic": FakeAsyncSplitter()},
        output_dir=out,
        skip_non_english=False,
        replace_all_results=False,
    )

    chunks_df = pd.read_parquet(out / "chunks.parquet")
    assert set(chunks_df["method"].unique()) == {"recursive", "semantic"}
    perf_df = pd.read_parquet(out / "performances.parquet")
    assert set(perf_df["method"].unique()) == {"recursive", "semantic"}


def test_corrupt_existing_parquet_is_overwritten(tmp_path):
    """A non-parquet file at the output paths triggers the except/overwrite path."""
    parsed = tmp_path / "parsed"
    _write_doc(parsed, pages={"1": PAGE1, "2": PAGE2})
    out = tmp_path / "out"
    out.mkdir(parents=True, exist_ok=True)
    (out / "chunks.parquet").write_text("not a parquet file")
    (out / "performances.parquet").write_text("not a parquet file")

    _run(
        parsed_docs_dir=parsed,
        sync_splitters={"recursive": FakeSyncSplitter()},
        async_splitters={},
        output_dir=out,
        skip_non_english=False,
        replace_all_results=False,  # attempts to read existing -> raises -> overwrite
    )

    chunks_df = pd.read_parquet(out / "chunks.parquet")
    assert set(chunks_df["method"].unique()) == {"recursive"}


def test_rerun_same_method_replaces_only_that_method(tmp_path):
    """Merge path where the re-run method already exists -> old rows dropped."""
    parsed = tmp_path / "parsed"
    _write_doc(parsed, pages={"1": PAGE1, "2": PAGE2})
    out = tmp_path / "out"

    _run(
        parsed_docs_dir=parsed,
        sync_splitters={"recursive": FakeSyncSplitter()},
        async_splitters={},
        output_dir=out,
        skip_non_english=False,
    )
    first = pd.read_parquet(out / "chunks.parquet")
    n_first = len(first)

    _run(
        parsed_docs_dir=parsed,
        sync_splitters={"recursive": FakeSyncSplitter()},
        async_splitters={},
        output_dir=out,
        skip_non_english=False,
        replace_all_results=False,
    )
    second = pd.read_parquet(out / "chunks.parquet")
    # not duplicated; same method count
    assert len(second) == n_first
    assert set(second["method"].unique()) == {"recursive"}
