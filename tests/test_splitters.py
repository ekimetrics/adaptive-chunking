import pytest
from adaptive_chunking.splitters import RecursiveSplitter


def make_splitter(token_counter, **kwargs):
    defaults = dict(
        chunk_size=50,
        chunk_overlap=0,
        length_function=token_counter,
        separators=["\n\n", "\n", " ", ""],
    )
    defaults.update(kwargs)
    return RecursiveSplitter(**defaults)


def make_paragraphs(n=10, words_per_para=20):
    """Generate n paragraphs of roughly words_per_para words each."""
    return "\n\n".join(
        " ".join(f"word{i}_{j}" for j in range(words_per_para))
        for i in range(n)
    )


class TestBasicSplit:
    def test_produces_nonempty_chunks(self, token_counter):
        text = make_paragraphs(10, 20)
        splitter = make_splitter(token_counter, chunk_size=50)
        chunks = splitter.split_text(text)

        assert len(chunks) > 0
        assert all(len(c) > 0 for c in chunks)

    def test_covers_original_text(self, token_counter):
        text = make_paragraphs(5, 15)
        splitter = make_splitter(token_counter, chunk_size=40)
        chunks = splitter.split_text(text)

        joined = "".join(chunks)
        # Without overlap, joined text should equal original
        assert joined == text

    def test_empty_text(self, token_counter):
        splitter = make_splitter(token_counter)
        assert splitter.split_text("") == []

    def test_short_text_single_chunk(self, token_counter):
        text = "Hello world this is short."
        splitter = make_splitter(token_counter, chunk_size=100)
        chunks = splitter.split_text(text)

        assert len(chunks) == 1
        assert chunks[0] == text


class TestChunkSize:
    def test_chunk_size_respected(self, token_counter):
        text = make_paragraphs(20, 15)
        splitter = make_splitter(token_counter, chunk_size=50)
        chunks = splitter.split_text(text)

        for chunk in chunks:
            assert token_counter(chunk) <= 50


class TestOverlap:
    def test_overlap_present(self, token_counter):
        text = make_paragraphs(10, 20)
        splitter = make_splitter(token_counter, chunk_size=50, chunk_overlap=10)
        chunks = splitter.split_text(text)

        assert len(chunks) >= 2
        # Check that consecutive chunks share some text
        found_overlap = False
        for i in range(len(chunks) - 1):
            words_a = set(chunks[i].split())
            words_b = set(chunks[i + 1].split())
            if words_a & words_b:
                found_overlap = True
                break
        assert found_overlap

    def test_overlap_exceeds_chunk_size_raises(self, token_counter):
        with pytest.raises(ValueError, match="cannot be greater than"):
            make_splitter(token_counter, chunk_size=100, chunk_overlap=200)


class TestMerging:
    def test_small_only_merging(self, token_counter):
        text = make_paragraphs(10, 5)  # small paragraphs
        splitter = make_splitter(
            token_counter,
            chunk_size=50,
            merging="small_only",
            min_chunk_tokens=10,
        )
        chunks = splitter.split_text(text)

        # Most chunks should be >= min_chunk_tokens (except possibly the last)
        for chunk in chunks[:-1]:
            assert token_counter(chunk) >= 10

    def test_backward_merging_covers_text(self, token_counter):
        text = make_paragraphs(8, 15)
        splitter = make_splitter(
            token_counter,
            chunk_size=50,
            merging_order="backward",
        )
        chunks = splitter.split_text(text)

        assert len(chunks) > 0
        assert "".join(chunks) == text


class TestRegexSeparators:
    def test_regex_separator(self, token_counter):
        # Markdown-like text with headings
        text = "# Heading 1\nSome content here.\n## Heading 2\nMore content.\n### Heading 3\nFinal content."
        splitter = make_splitter(
            token_counter,
            chunk_size=100,
            separators=[r"\n#{1,3}\s"],
            is_separator_regex=True,
        )
        chunks = splitter.split_text(text)

        assert len(chunks) >= 1
        # Joined chunks should reconstruct the original
        assert "".join(chunks) == text
