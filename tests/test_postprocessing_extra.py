"""Extra coverage for the pure functions in adaptive_chunking.postprocessing.

Uses ``count_tokens_func=len`` for deterministic, tiktoken-free sizing.
"""

from adaptive_chunking.postprocessing import (
    get_page_info,
    get_title_info,
    check_chunk_gaps,
    find_chunks_start_and_end,
    repair_gaps_between_chunks,
    split_oversized_chunks,
    merge_small_chunks_smallest_first,
    merge_small_chunks_to_neighbours,
)

import pytest


# --------------------------------------------------------------------------
# get_page_info
# --------------------------------------------------------------------------
class TestGetPageInfo:
    def test_empty_chunks_returns_empty(self):
        assert get_page_info({1: "abc"}, [], "abc") == []

    def test_chunk_on_single_page(self):
        pages = {1: "Hello ", 2: "world."}
        text = "Hello world."
        # each chunk falls entirely within one page
        assert get_page_info(pages, ["Hello ", "world."], text) == [[1], [2]]

    def test_chunk_overlapping_two_pages(self):
        pages = {1: "Hello ", 2: "world."}
        text = "Hello world."
        # a single chunk that spans the page boundary overlaps both pages
        assert get_page_info(pages, ["Hello world."], text) == [[1, 2]]

    def test_chunks_spanning_multiple_pages(self):
        # three pages; chunks of varying spans
        pages = {1: "AAAA", 2: "BBBB", 3: "CCCC"}
        text = "AAAABBBBCCCC"
        # chunk0 covers page1 fully + start of page2; chunk1 covers rest
        result = get_page_info(pages, ["AAAAB", "BBBCCCC"], text)
        assert result == [[1, 2], [2, 3]]

    def test_string_page_keys_are_converted(self):
        # JSON-loaded pages use string keys; get_page_info int()-casts them
        pages = {"1": "Hello ", "2": "world."}
        text = "Hello world."
        assert get_page_info(pages, ["Hello world."], text) == [[1, 2]]


# --------------------------------------------------------------------------
# get_title_info
# --------------------------------------------------------------------------
class TestGetTitleInfo:
    def test_empty_chunks_returns_empty(self):
        assert get_title_info([{"title": "T", "start": 0, "end": 1}], [], "x") == []

    def test_present_title_not_added_missing_title_added(self):
        text = "Alpha Beta Gamma Delta"
        chunks = ["Alpha Beta ", "Gamma Delta"]
        titles = [
            # spans chunk0 start AND is present in chunk0 -> NOT added
            {"title": "Alpha", "start": 0, "end": 5, "level": 1},
            # spans both chunk starts but text absent from chunks -> added
            {"title": "SECTION", "start": 0, "end": 15, "level": 1},
        ]
        result = get_title_info(titles, chunks, text)
        assert result == ["SECTION", "SECTION"]

    def test_title_out_of_span_not_added(self):
        text = "Alpha Beta Gamma Delta"
        chunks = ["Alpha Beta ", "Gamma Delta"]
        # title span [0,5) only covers chunk0 start (0); chunk1 start (11) is out of span
        titles = [{"title": "ZZZ", "start": 0, "end": 5, "level": 1}]
        result = get_title_info(titles, chunks, text)
        assert result == ["ZZZ", ""]


# --------------------------------------------------------------------------
# check_chunk_gaps
# --------------------------------------------------------------------------
class TestCheckChunkGaps:
    def test_no_gaps_true(self):
        assert check_chunk_gaps(["Hello ", "world."], "Hello world.") is True

    def test_overlap_allowed_true(self):
        # overlapping chunks still cover all chars with no gap
        assert check_chunk_gaps(["ABCD", "CDEF"], "ABCDEF") is True

    def test_with_gaps_false(self):
        assert check_chunk_gaps(["Hello ", "world."], "Hello beautiful world.") is False

    def test_empty_chunks_empty_text_true(self):
        assert check_chunk_gaps([], "") is True

    def test_empty_chunks_nonempty_text_false(self):
        assert check_chunk_gaps([], "something") is False

    def test_forward_search_fallback_detects_gap_false(self):
        # "BBBB" is not in the local backward window after "AAAA";
        # the forward-search fallback finds it later -> gap -> False
        assert check_chunk_gaps(["AAAA", "BBBB"], "AAAAXXXXBBBB") is False

    def test_chunk_absent_returns_false(self):
        # forward fallback also fails to find the chunk -> False
        assert check_chunk_gaps(["AAAA", "ZZZZ"], "AAAABBBB") is False


# --------------------------------------------------------------------------
# find_chunks_start_and_end
# --------------------------------------------------------------------------
class TestFindChunksStartAndEnd:
    def test_basic(self):
        text = "Hello world."
        assert find_chunks_start_and_end(["Hello ", "world."], text) == [(0, 6), (6, 12)]

    def test_empty_returns_empty(self):
        assert find_chunks_start_and_end([], "abc") == []

    def test_forward_search_fallback(self):
        # "BBBB" sits beyond the backward window -> forward search finds it
        assert find_chunks_start_and_end(["AAAA", "BBBB"], "AAAAXXXXBBBB") == [(0, 4), (8, 12)]

    def test_chunk_absent_raises(self):
        with pytest.raises(ValueError, match="Chunk not found in text."):
            find_chunks_start_and_end(["ZZ"], "AAAA")


# --------------------------------------------------------------------------
# repair_gaps_between_chunks
# --------------------------------------------------------------------------
class TestRepairGaps:
    def test_empty_returns_empty(self):
        assert repair_gaps_between_chunks([], "anything") == []

    def test_fills_internal_gap(self):
        text = "AAAA BBBB CCCC DDDD"
        repaired = repair_gaps_between_chunks(["AAAA BBBB ", "DDDD"], text)
        assert repaired == ["AAAA BBBB ", "CCCC DDDD"]
        assert check_chunk_gaps(repaired, text) is True

    def test_appends_trailing_tail(self):
        text = "AAAA BBBB EXTRA"
        repaired = repair_gaps_between_chunks(["AAAA ", "BBBB "], text)
        assert repaired == ["AAAA ", "BBBB EXTRA"]
        assert check_chunk_gaps(repaired, text) is True


# --------------------------------------------------------------------------
# split_oversized_chunks
# --------------------------------------------------------------------------
class _FakeSplitter:
    """Splits a chunk into two halves."""

    def split_text(self, text):
        half = len(text) // 2
        return [text[:half], text[half:]]


class TestSplitOversizedChunks:
    def test_splits_oversized_keeps_others(self):
        chunks = ["short", "aaaaaaaaaa"]
        result = split_oversized_chunks(
            chunks, _FakeSplitter(), count_tokens_func=len, max_chunk_tokens=5
        )
        # "short" (len 5) kept as-is; "aaaaaaaaaa" (len 10 > 5) split in half
        assert result == ["short", "aaaaa", "aaaaa"]

    def test_nothing_oversized(self):
        chunks = ["a", "bb", "ccc"]
        result = split_oversized_chunks(
            chunks, _FakeSplitter(), count_tokens_func=len, max_chunk_tokens=100
        )
        assert result == chunks


# --------------------------------------------------------------------------
# merge_small_chunks_smallest_first
# --------------------------------------------------------------------------
class TestMergeSmallestFirst:
    def test_len_below_two_returns_input(self):
        assert merge_small_chunks_smallest_first(["x"], len, 3, 100) == ["x"]

    def test_merges_to_smaller_right_neighbour(self):
        # middle "x" (len 1) merges to the smaller (right, len 2) neighbour
        result = merge_small_chunks_smallest_first(["aaa", "x", "cc"], len, 3, 100)
        assert result == ["aaa", "xcc"]

    def test_merges_to_smaller_left_neighbour(self):
        # middle "x" merges to the smaller (left, len 3) neighbour
        result = merge_small_chunks_smallest_first(["aaa", "x", "cccccc"], len, 3, 100)
        assert result == ["aaax", "cccccc"]

    def test_cannot_merge_respects_max_limit(self):
        # max_limit=3 makes both neighbour merges exceed the limit -> no merge
        result = merge_small_chunks_smallest_first(["aaa", "x", "cccccc"], len, 3, 3)
        assert result == ["aaa", "x", "cccccc"]


# --------------------------------------------------------------------------
# merge_small_chunks_to_neighbours
# --------------------------------------------------------------------------
class TestMergeToNeighbours:
    def test_len_below_two_returns_input(self):
        assert merge_small_chunks_to_neighbours(["x"], len, 3, 100, "next") == ["x"]

    def test_merge_to_next(self):
        result = merge_small_chunks_to_neighbours(["aaa", "x", "ccc"], len, 3, 100, "next")
        assert result == ["aaa", "xccc"]

    def test_merge_to_previous(self):
        result = merge_small_chunks_to_neighbours(["aaa", "x", "ccc"], len, 3, 100, "previous")
        assert result == ["aaax", "ccc"]

    def test_next_falls_back_to_previous_for_last_chunk(self):
        # last small chunk has no next; "next" mode falls back to previous neighbour
        result = merge_small_chunks_to_neighbours(["aaa", "ccc", "x"], len, 3, 100, "next")
        assert result == ["aaa", "cccx"]

    def test_previous_falls_back_to_next_for_first_chunk(self):
        # first small chunk has no previous; "previous" mode falls back to next neighbour
        result = merge_small_chunks_to_neighbours(["x", "aaa", "ccc"], len, 3, 100, "previous")
        assert result == ["xaaa", "ccc"]

    def test_middle_chunk_unmergeable_both_sides_previous(self):
        # "previous" mode: small middle chunk, both neighbour merges exceed max ->
        # neither fallback fires, chunk left in place
        result = merge_small_chunks_to_neighbours(["aaaa", "x", "bbbb"], len, 3, 4, "previous")
        assert result == ["aaaa", "x", "bbbb"]

    def test_middle_chunk_unmergeable_both_sides_next(self):
        # "next" mode: same, both fallbacks fail max_limit -> chunk left in place
        result = merge_small_chunks_to_neighbours(["aaaa", "x", "bbbb"], len, 3, 4, "next")
        assert result == ["aaaa", "x", "bbbb"]

    def test_unmergeable_chunk_left_in_place(self):
        # min_limit small enough that nothing is "small" -> trailing special-cases
        # evaluate to False; input returned unchanged
        result = merge_small_chunks_to_neighbours(["aaa", "bbb", "ccc"], len, 1, 100, "next")
        assert result == ["aaa", "bbb", "ccc"]
