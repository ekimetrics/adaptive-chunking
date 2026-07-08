"""Extra coverage tests for adaptive_chunking.splitters."""
import pytest

from adaptive_chunking.splitters import (
    RecursiveSplitter,
    group_chunks,
    group_pages,
    combine_blocks,
    regex_splitter,
)
from adaptive_chunking.postprocessing import (
    check_chunk_gaps,
    repair_gaps_between_chunks,
)


# ---------------------------------------------------------------------------
# __init__ validation
# ---------------------------------------------------------------------------

def test_init_overlap_greater_than_size_raises():
    with pytest.raises(ValueError):
        RecursiveSplitter(chunk_size=10, chunk_overlap=20, length_function=len)


def test_init_negative_min_chunk_tokens_raises():
    with pytest.raises(ValueError):
        RecursiveSplitter(min_chunk_tokens=-1, length_function=len)


def test_init_invalid_merging_order_raises():
    with pytest.raises(ValueError):
        RecursiveSplitter(merging_order="sideways", length_function=len)


def test_init_invalid_max_tokens_strategy_raises():
    with pytest.raises(ValueError):
        RecursiveSplitter(max_tokens_strategy="bogus", length_function=len)


def test_init_max_tokens_strategy_chunk_size():
    s = RecursiveSplitter(
        chunk_size=100,
        chunk_overlap=20,
        max_tokens_strategy="chunk_size",
        length_function=len,
    )
    assert s.chunk_size == 80  # chunk_size - chunk_overlap


def test_init_max_tokens_strategy_chunk_size_plus_overlap():
    s = RecursiveSplitter(
        chunk_size=100,
        chunk_overlap=20,
        max_tokens_strategy="chunk_size_plus_overlap",
        length_function=len,
    )
    assert s.chunk_size == 100


# ---------------------------------------------------------------------------
# split_text - empty
# ---------------------------------------------------------------------------

def test_split_text_empty():
    s = RecursiveSplitter(chunk_size=50, length_function=len)
    assert s.split_text("") == []


# ---------------------------------------------------------------------------
# split_text - merging="to_chunk_size"
# ---------------------------------------------------------------------------

def test_to_chunk_size_basic_covers_text():
    text = "Para one.\n\nPara two is here.\n\nPara three goes on.\n\nPara four ends it."
    s = RecursiveSplitter(
        chunk_size=25,
        chunk_overlap=0,
        merging="to_chunk_size",
        length_function=len,
    )
    chunks = s.split_text(text)
    assert chunks
    assert all(isinstance(c, str) for c in chunks)
    # With zero overlap the concatenation should reproduce the original text.
    assert "".join(chunks) == text


def test_to_chunk_size_with_overlap():
    text = (
        "Alpha bravo charlie.\n\nDelta echo foxtrot golf.\n\n"
        "Hotel india juliet kilo.\n\nLima mike november oscar papa."
    )
    s = RecursiveSplitter(
        chunk_size=30,
        chunk_overlap=10,
        merging="to_chunk_size",
        length_function=len,
    )
    chunks = s.split_text(text)
    assert len(chunks) >= 2
    # repair gaps then verify full coverage
    repaired = repair_gaps_between_chunks(chunks, text)
    assert check_chunk_gaps(repaired, text)


def test_to_chunk_size_backward_order():
    text = "One two three.\n\nFour five six seven.\n\nEight nine ten eleven twelve."
    s = RecursiveSplitter(
        chunk_size=25,
        chunk_overlap=0,
        merging="to_chunk_size",
        merging_order="backward",
        length_function=len,
    )
    chunks = s.split_text(text)
    assert chunks
    assert "".join(chunks) == text


def test_to_chunk_size_backward_with_overlap():
    text = (
        "Alpha bravo charlie delta.\n\nEcho foxtrot golf hotel.\n\n"
        "India juliet kilo lima.\n\nMike november oscar papa."
    )
    s = RecursiveSplitter(
        chunk_size=30,
        chunk_overlap=10,
        merging="to_chunk_size",
        merging_order="backward",
        length_function=len,
    )
    chunks = s.split_text(text)
    assert len(chunks) >= 2


# ---------------------------------------------------------------------------
# split_text - merging="small_only"
# ---------------------------------------------------------------------------

def test_small_only_single_split():
    text = "Just one short piece."
    s = RecursiveSplitter(
        chunk_size=1000,
        merging="small_only",
        min_chunk_tokens=5,
        length_function=len,
    )
    chunks = s.split_text(text)
    assert chunks == [text]


def test_small_only_multiple_splits_no_overlap():
    text = (
        "First paragraph has enough words here.\n\n"
        "Second paragraph also has plenty of content.\n\n"
        "Third paragraph wraps things up nicely now."
    )
    s = RecursiveSplitter(
        chunk_size=60,
        chunk_overlap=0,
        merging="small_only",
        min_chunk_tokens=5,
        length_function=len,
    )
    chunks = s.split_text(text)
    assert len(chunks) >= 2
    assert "".join(chunks) == text


def test_small_only_with_overlap():
    text = (
        "Alpha bravo charlie delta echo.\n\n"
        "Foxtrot golf hotel india juliet.\n\n"
        "Kilo lima mike november oscar.\n\n"
        "Papa quebec romeo sierra tango."
    )
    s = RecursiveSplitter(
        chunk_size=40,
        chunk_overlap=12,
        merging="small_only",
        min_chunk_tokens=5,
        length_function=len,
    )
    chunks = s.split_text(text)
    assert len(chunks) >= 2


def test_small_only_backward():
    text = (
        "Section one is reasonably sized text.\n\n"
        "Section two carries on with more text.\n\n"
        "Section three completes the document here."
    )
    s = RecursiveSplitter(
        chunk_size=60,
        chunk_overlap=0,
        merging="small_only",
        merging_order="backward",
        min_chunk_tokens=5,
        length_function=len,
    )
    chunks = s.split_text(text)
    assert chunks
    # backward processing preserves the constituent text segments
    assert "Section one" in "".join(chunks)
    assert "Section three" in "".join(chunks)


_MULTI = "aaaaaaaa\n\nbbbbbbbb\n\ncccccccc\n\ndddddddd\n\neeeeeeee\n\nffffffff\n\ngg"


def test_small_only_multi_split_merge_next_and_prev():
    # multiple initial splits each below min_chunk_tokens with room to merge
    # (exercises the merge-next and merge-prev grouping loop).
    s = RecursiveSplitter(
        chunk_size=60,
        chunk_overlap=0,
        merging="small_only",
        min_chunk_tokens=30,
        length_function=len,
    )
    chunks = s.split_text(_MULTI)
    assert chunks
    assert all(len(c) <= 60 for c in chunks)


def test_small_only_backward_with_overlap():
    # backward + overlap + multiple groups exercises the reversed-join and the
    # overlap-backtracking tail in _merge_small_splits.
    s = RecursiveSplitter(
        chunk_size=60,
        chunk_overlap=12,
        merging="small_only",
        merging_order="backward",
        min_chunk_tokens=30,
        length_function=len,
    )
    chunks = s.split_text(_MULTI)
    assert len(chunks) >= 2


def test_small_only_forward_overlap_accumulate():
    s = RecursiveSplitter(
        chunk_size=60,
        chunk_overlap=12,
        merging="small_only",
        merging_order="forward",
        min_chunk_tokens=30,
        length_function=len,
    )
    chunks = s.split_text(_MULTI)
    assert len(chunks) >= 2


def test_merge_small_splits_empty_direct():
    s = RecursiveSplitter(length_function=len)
    assert s._merge_small_splits([]) == []


def test_to_chunk_size_overlap_accumulate_small_parts():
    # parts (~18 chars) <= chunk_overlap (20) so the overlap loop appends parts
    # and then breaks on the size limit (lines 253-257) rather than resplitting.
    text = (
        "aaaaaaaa\nbbbbbbbb\ncccccccc\ndddddddd\n"
        "eeeeeeee\nffffffff\ngggggggg\nhhhhhhhh"
    )
    s = RecursiveSplitter(
        chunk_size=40,
        chunk_overlap=20,
        separators=["\n", ""],
        merging="to_chunk_size",
        attach_separator_to="start",
        length_function=len,
    )
    chunks = s.split_text(text)
    assert len(chunks) >= 2


def test_hard_split_fallback_no_empty_separator():
    # separators with NO "" -> a too-long token forces the [""] hard-split
    # fallback branch (line 201).
    text = "short " + "z" * 40
    s = RecursiveSplitter(
        chunk_size=8,
        chunk_overlap=0,
        separators=[" "],
        merging="small_only",
        min_chunk_tokens=0,
        attach_separator_to="end",
        length_function=len,
    )
    chunks = s.split_text(text)
    assert "".join(chunks) == text
    assert all(len(c) <= 8 for c in chunks)


def test_split_with_separator_invalid_attach_raises():
    s = RecursiveSplitter(length_function=len)
    s.attach_separator_to = "middle"
    with pytest.raises(ValueError):
        s._split_with_separator("one\n\ntwo", regex_pattern=r"\n\n", min_len=1)


def test_small_only_min_chunk_tokens_merging():
    # Tiny separators create small pieces that must be merged up to min_chunk_tokens.
    text = "a\n\nbb\n\nccc\n\ndddd\n\neeeee\n\nffffff\n\nggggggg"
    s = RecursiveSplitter(
        chunk_size=1000,
        chunk_overlap=0,
        merging="small_only",
        min_chunk_tokens=15,
        length_function=len,
    )
    chunks = s.split_text(text)
    # everything merges into a single chunk because min_chunk_tokens is large
    assert "".join(chunks) == text
    assert len(chunks) == 1


def test_small_only_min_chunk_merge_with_previous():
    # Small trailing group: merge-next is impossible (it is last), so it falls
    # back on merging with the previous group (lines 314-317).
    text = "aaaaaaaaaaaaaaaaaaaa\n\nbbbbbbbbbbbbbbbbbbbb\n\ncc"
    s = RecursiveSplitter(
        chunk_size=45,
        chunk_overlap=0,
        merging="small_only",
        min_chunk_tokens=10,
        length_function=len,
    )
    chunks = s.split_text(text)
    # the tiny "cc" piece is absorbed into the previous group
    assert len(chunks) == 2
    assert chunks[-1].endswith("cc")


# ---------------------------------------------------------------------------
# _split_with_separator attach end / start
# ---------------------------------------------------------------------------

def test_attach_separator_to_end():
    text = "one\n\ntwo\n\nthree"
    s = RecursiveSplitter(
        chunk_size=4,
        chunk_overlap=0,
        attach_separator_to="end",
        merging="small_only",
        min_chunk_tokens=0,
        length_function=len,
    )
    # call the internal splitter directly to inspect separator placement
    out = s._split_with_separator(text, regex_pattern=r"\n\n", min_len=1)
    assert out == ["one\n\n", "two\n\n", "three"]


def test_attach_separator_to_start():
    text = "one\n\ntwo\n\nthree"
    s = RecursiveSplitter(
        chunk_size=4,
        chunk_overlap=0,
        attach_separator_to="start",
        merging="small_only",
        min_chunk_tokens=0,
        length_function=len,
    )
    out = s._split_with_separator(text, regex_pattern=r"\n\n", min_len=1)
    assert out == ["one", "\n\ntwo", "\n\nthree"]


def test_split_with_separator_no_match_returns_pieces():
    s = RecursiveSplitter(length_function=len)
    out = s._split_with_separator("nodelimiter", regex_pattern=r"\n\n", min_len=1)
    assert out == ["nodelimiter"]


def test_split_with_separator_end_min_len_merge():
    # short trailing chunk (< min_len) merges into previous via "end" path
    s = RecursiveSplitter(attach_separator_to="end", length_function=len)
    out = s._split_with_separator("aaaaaaaa.b", regex_pattern=r"\.", min_len=5)
    # pieces: 'aaaaaaaa', '.', 'b' -> chunks ['aaaaaaaa.', 'b']; 'b' < 5 merges
    assert out == ["aaaaaaaa.b"]


def test_split_with_separator_start_min_len_merge():
    s = RecursiveSplitter(attach_separator_to="start", length_function=len)
    out = s._split_with_separator("aa.bbbbbbbb", regex_pattern=r"\.", min_len=5)
    # pieces: 'aa', '.', 'bbbbbbbb' -> chunks ['aa', '.bbbbbbbb']; 'aa' < 5 merges fwd
    assert out == ["aa.bbbbbbbb"]


# ---------------------------------------------------------------------------
# is_separator_regex
# ---------------------------------------------------------------------------

def test_is_separator_regex_true():
    text = "alpha123beta456gamma"
    s = RecursiveSplitter(
        chunk_size=6,
        chunk_overlap=0,
        is_separator_regex=True,
        separators=[r"\d+", ""],
        merging="small_only",
        min_chunk_tokens=0,
        attach_separator_to="end",
        length_function=len,
    )
    chunks = s.split_text(text)
    assert "".join(chunks) == text


def test_invalid_regex_raises_in_recursive_split():
    s = RecursiveSplitter(
        chunk_size=2,
        is_separator_regex=True,
        separators=["[unterminated", ""],
        length_function=len,
    )
    with pytest.raises(ValueError):
        s.split_text("some text that is definitely longer than two chars")


# ---------------------------------------------------------------------------
# empty-string separator hard split (binary search)
# ---------------------------------------------------------------------------

def test_empty_separator_hard_split():
    text = "abcdefghijklmnopqrstuvwxyz"  # no separators present
    s = RecursiveSplitter(
        chunk_size=5,
        chunk_overlap=0,
        separators=[""],
        merging="small_only",
        min_chunk_tokens=0,
        length_function=len,
    )
    chunks = s.split_text(text)
    assert "".join(chunks) == text
    assert all(len(c) <= 5 for c in chunks)


def test_hard_split_via_separator_fallthrough():
    # Long token with a space separator that doesn't help -> falls to [""] hard split
    text = "shortword " + "x" * 50
    s = RecursiveSplitter(
        chunk_size=8,
        chunk_overlap=0,
        separators=[" ", ""],
        merging="small_only",
        min_chunk_tokens=0,
        attach_separator_to="end",
        length_function=len,
    )
    chunks = s.split_text(text)
    assert "".join(chunks) == text
    assert all(len(c) <= 8 for c in chunks)


# ---------------------------------------------------------------------------
# invalid merging strategy
# ---------------------------------------------------------------------------

def test_invalid_merging_strategy_raises():
    s = RecursiveSplitter(chunk_size=10, merging="to_chunk_size", length_function=len)
    s.merging = "bogus"
    with pytest.raises(ValueError):
        s.split_text("this text is long enough to produce splits for sure now")


# ---------------------------------------------------------------------------
# group_chunks
# ---------------------------------------------------------------------------

def tok(s):
    return s.split()


def test_group_chunks_normal():
    blocks = ["one two ", "three four ", "five six ", "seven eight "]
    chunks, grouped = group_chunks(blocks, tokenizer_func=tok, max_tokens=4)
    assert chunks
    assert grouped
    assert len(chunks) == len(grouped)


def test_group_chunks_oversized_block_cropped_verbose(capsys):
    blocks = ["start ", "a b c d e f g h i j", "end "]
    chunks, grouped = group_chunks(
        blocks, tokenizer_func=tok, max_tokens=3, verbose=True
    )
    captured = capsys.readouterr()
    assert "exceed max_tokens" in captured.out
    # the oversized block got cropped to <= 3 tokens
    assert any(len(tok(c)) <= 3 for c in chunks)


def test_group_chunks_with_block_overlap():
    blocks = ["aa ", "bb ", "cc ", "dd ", "ee ", "ff "]
    chunks, grouped = group_chunks(
        blocks, tokenizer_func=tok, max_tokens=2, chunk_block_overlap=1
    )
    assert len(chunks) >= 2
    assert len(chunks) == len(grouped)


def test_group_chunks_empty():
    chunks, grouped = group_chunks([], tokenizer_func=tok, max_tokens=5)
    assert chunks == []
    assert grouped == []


def test_group_chunks_crop_empty_block():
    # An empty oversized block can't happen, but exercise final-chunk handling
    blocks = ["word "]
    chunks, grouped = group_chunks(blocks, tokenizer_func=tok, max_tokens=5)
    assert chunks == ["word "]


# ---------------------------------------------------------------------------
# group_pages
# ---------------------------------------------------------------------------

def test_group_pages_basic_with_overlap():
    pages = [
        "p1l1\np1l2\n",
        "p2l1\np2l2\n",
        "p3l1\np3l2\n",
        "p4l1\np4l2\n",
    ]
    out = group_pages(pages, pages_per_group=2, overlap_lines=1)
    assert len(out) == 2
    # second group should contain a context line from the first group
    assert "p2l2" in out[1]


def test_group_pages_no_overlap():
    pages = ["p1\n", "p2\n", "p3\n", "p4\n"]
    out = group_pages(pages, pages_per_group=2, overlap_lines=0)
    assert out == ["p1\np2\n", "p3\np4\n"]


def test_group_pages_trailing_remainder_with_overlap():
    pages = ["p1\np1b\n", "p2\np2b\n", "p3\np3b\n"]
    out = group_pages(pages, pages_per_group=2, overlap_lines=1)
    # one full group + a trailing remainder group
    assert len(out) == 2
    assert "p3" in out[-1]


def test_group_pages_trailing_remainder_no_overlap():
    pages = ["p1\n", "p2\n", "p3\n"]
    out = group_pages(pages, pages_per_group=2, overlap_lines=0)
    assert len(out) == 2
    assert out[-1] == "p3\n"


def test_group_pages_single_remainder_only():
    pages = ["only\n"]
    out = group_pages(pages, pages_per_group=2, overlap_lines=5)
    assert out == ["only\n"]


# ---------------------------------------------------------------------------
# combine_blocks
# ---------------------------------------------------------------------------

def test_combine_blocks_stops_at_max():
    blocks = ["aaa", "bbb", "ccc", "ddd"]
    out = combine_blocks(blocks, max_tokens=6, count_tokens_func=len)
    # 3 + 3 = 6 fits, next would be 9 > 6 -> stop
    assert out == "aaabbb"


def test_combine_blocks_all_fit():
    blocks = ["a", "b", "c"]
    out = combine_blocks(blocks, max_tokens=100, count_tokens_func=len)
    assert out == "abc"


# ---------------------------------------------------------------------------
# regex_splitter
# ---------------------------------------------------------------------------

def test_regex_splitter_start():
    out = regex_splitter("one\n\ntwo\n\nthree", r"\n\n", attach_to="start", min_len=1)
    assert out == ["one", "\n\ntwo", "\n\nthree"]


def test_regex_splitter_end():
    out = regex_splitter("one\n\ntwo\n\nthree", r"\n\n", attach_to="end", min_len=1)
    assert out == ["one\n\n", "two\n\n", "three"]


def test_regex_splitter_start_min_len_merge():
    out = regex_splitter("aa.bbbbbbbb", r"\.", attach_to="start", min_len=5)
    assert out == ["aa.bbbbbbbb"]


def test_regex_splitter_end_min_len_merge():
    out = regex_splitter("aaaaaaaa.b", r"\.", attach_to="end", min_len=5)
    assert out == ["aaaaaaaa.b"]


def test_regex_splitter_no_match():
    out = regex_splitter("nodelim", r"\n\n", attach_to="start", min_len=1)
    assert out == ["nodelim"]


def test_regex_splitter_invalid_regex_prints_and_returns(capsys):
    out = regex_splitter("some text", "[unterminated", attach_to="start")
    captured = capsys.readouterr()
    assert "Error compiling" in captured.out
    assert out == ["some text"]


def test_regex_splitter_invalid_attach_to_raises():
    with pytest.raises(ValueError):
        regex_splitter("one\n\ntwo", r"\n\n", attach_to="middle", min_len=1)
