"""End-to-end PDF-to-chunks pipeline."""

import json
import shutil
import tempfile
from pathlib import Path
from typing import Literal

from .chunking_utils import count_tokens
from .postprocessing import (
    check_chunk_gaps,
    get_page_info,
    get_title_info,
    repair_gaps_between_chunks,
)
from .splitters import RecursiveSplitter


def chunk_files(
    input_path: str | Path,
    parser: "BaseParser | None" = None,
    chunk_size: int = 600,
    chunk_overlap: int = 50,
    separators: list[str] | None = None,
    merging: Literal["to_chunk_size", "small_only"] = "small_only",
    min_chunk_tokens: int = 100,
    output_dir: str | Path | None = None,
) -> list[dict]:
    """Parse PDF(s) and split into chunks in one step.

    Args:
        input_path: Path to a single PDF file or directory of PDFs.
        parser: A parser instance (DoclingParser, PyMuPDFParser, AzureDIParser).
            If None, uses DoclingParser (requires ``pip install adaptive-chunking[parsing]``).
        chunk_size: Maximum chunk size in tokens.
        chunk_overlap: Overlap between chunks in tokens.
        separators: Separators for recursive splitting.
            Defaults to ``["\\n\\n", "\\n", " ", ""]``.
        merging: Merge strategy (``"to_chunk_size"`` or ``"small_only"``).
        min_chunk_tokens: Minimum chunk size in tokens (for ``"small_only"`` merging).
        output_dir: Directory to save parsed JSON files. If *None*, uses a
            temporary directory that is cleaned up automatically.

    Returns:
        List of dicts, each with keys:

        - ``doc_name`` – source document name
        - ``chunk_index`` – position of the chunk in the document
        - ``chunk_text`` – the chunk content
        - ``chunk_pages`` – list of page numbers the chunk spans
        - ``titles_context`` – enclosing titles not present in the chunk text
        - ``chunk_len`` – chunk length in tokens
    """
    input_path = Path(input_path)

    if separators is None:
        separators = ["\n\n", "\n", " ", ""]

    # Create parser if not provided
    if parser is None:
        from .parsing import DoclingParser

        parser = DoclingParser()

    # Handle single file vs directory
    tmp_input_dir = None
    if input_path.is_file():
        tmp_input_dir = tempfile.mkdtemp()
        shutil.copy2(input_path, Path(tmp_input_dir) / input_path.name)
        pdf_dir = Path(tmp_input_dir)
    elif input_path.is_dir():
        pdf_dir = input_path
    else:
        raise FileNotFoundError(f"Path does not exist: {input_path}")

    # Setup output directories
    tmp_output_dir = None
    if output_dir is None:
        tmp_output_dir = tempfile.mkdtemp()
        raw_dir = Path(tmp_output_dir) / "raw"
        parsed_dir = Path(tmp_output_dir) / "parsed"
    else:
        output_dir = Path(output_dir)
        raw_dir = output_dir / "raw"
        parsed_dir = output_dir / "parsed"

    try:
        # Step 1: Parse PDFs → raw parser output
        parser.parse_docs_in_dir(pdf_dir, raw_dir)

        # Step 2: Convert raw output → standard JSON
        parsed_docs = parser.convert_raw_results_to_markdown(raw_dir, parsed_dir)

        # Step 3: Split into chunks
        splitter = RecursiveSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators,
            merging=merging,
            min_chunk_tokens=min_chunk_tokens,
        )

        results: list[dict] = []
        for doc in parsed_docs:
            doc_name = doc["document_name"]
            full_text = doc["full_text"]

            if not full_text:
                continue

            chunks = splitter.split_text(full_text)
            chunks = repair_gaps_between_chunks(chunks=chunks, text=full_text)

            if not check_chunk_gaps(chunks, full_text):
                raise RuntimeError(
                    f"Chunk gap recovery failed for '{doc_name}'. "
                    "This is a bug — please report it."
                )

            page_info = get_page_info(
                pages=doc["pages"], chunks=chunks, text=full_text
            )
            title_info = get_title_info(
                titles=doc["titles"], chunks=chunks, text=full_text
            )

            for i, chunk_text in enumerate(chunks):
                results.append(
                    {
                        "doc_name": doc_name,
                        "chunk_index": i,
                        "chunk_text": chunk_text,
                        "chunk_pages": page_info[i],
                        "titles_context": title_info[i],
                        "chunk_len": count_tokens(chunk_text),
                    }
                )

        return results

    finally:
        if tmp_input_dir:
            shutil.rmtree(tmp_input_dir, ignore_errors=True)
        if tmp_output_dir:
            shutil.rmtree(tmp_output_dir, ignore_errors=True)
