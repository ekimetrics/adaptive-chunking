"""Coverage for pipeline.chunk_files using a fake parser (no docling needed)."""

import pytest

from adaptive_chunking.pipeline import chunk_files

PARA = (
    "The quick brown fox jumps over the lazy dog. "
    "Pack my box with five dozen liquor jugs. "
    "How vexingly quick daft zebras jump."
)


class FakeParser:
    """Stands in for a BaseParser. Returns one doc with given full_text."""

    def __init__(self, docs):
        self._docs = docs

    def parse_docs_in_dir(self, pdf_dir, raw_dir):
        return None

    def convert_raw_results_to_markdown(self, raw_dir, parsed_dir):
        return self._docs


def _doc(name="doc1", full_text=PARA):
    return {
        "document_name": name,
        "full_text": full_text,
        "pages": {1: full_text},
        "titles": [],
    }


def _expected_keys():
    return {
        "doc_name",
        "chunk_index",
        "chunk_text",
        "chunk_pages",
        "titles_context",
        "chunk_len",
    }


def test_chunk_files_dir_input(tmp_path):
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    (pdf_dir / "doc1.pdf").touch()

    parser = FakeParser([_doc()])
    results = chunk_files(pdf_dir, parser=parser)

    assert isinstance(results, list)
    assert len(results) >= 1
    for r in results:
        assert set(r.keys()) == _expected_keys()
    assert results[0]["doc_name"] == "doc1"
    assert results[0]["chunk_index"] == 0
    assert isinstance(results[0]["chunk_len"], int)


def test_chunk_files_single_file_input(tmp_path):
    pdf_file = tmp_path / "single.pdf"
    pdf_file.touch()

    parser = FakeParser([_doc(name="single")])
    results = chunk_files(pdf_file, parser=parser)

    assert len(results) >= 1
    assert results[0]["doc_name"] == "single"


def test_chunk_files_explicit_output_dir(tmp_path):
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    (pdf_dir / "doc1.pdf").touch()
    out_dir = tmp_path / "out"

    parser = FakeParser([_doc()])
    results = chunk_files(pdf_dir, parser=parser, output_dir=out_dir)

    # The explicit output_dir branch is exercised (raw_dir/parsed_dir derived
    # from out_dir rather than a temp dir); a real parser would populate it.
    assert len(results) >= 1
    assert results[0]["doc_name"] == "doc1"


def test_chunk_files_skips_empty_full_text(tmp_path):
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    (pdf_dir / "doc1.pdf").touch()

    parser = FakeParser([_doc(name="empty", full_text=""), _doc(name="ok")])
    results = chunk_files(pdf_dir, parser=parser)

    names = {r["doc_name"] for r in results}
    assert "empty" not in names
    assert "ok" in names


def test_chunk_files_missing_path_raises(tmp_path):
    missing = tmp_path / "does_not_exist"
    parser = FakeParser([_doc()])
    with pytest.raises(FileNotFoundError):
        chunk_files(missing, parser=parser)
