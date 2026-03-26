from adaptive_chunking.postprocessing import (
    find_chunks_start_and_end,
    check_chunk_gaps,
    repair_gaps_between_chunks,
)


class TestFindChunksStartAndEnd:
    def test_basic(self):
        text = "Hello world. This is a test. Final sentence."
        chunks = ["Hello world. ", "This is a test. ", "Final sentence."]
        result = find_chunks_start_and_end(chunks, text)

        assert result[0] == (0, 13)
        assert result[1] == (13, 29)
        assert result[2] == (29, 44)

    def test_contiguous_chunks(self):
        text = "AAAA BBBB CCCC"
        chunks = ["AAAA ", "BBBB ", "CCCC"]
        positions = find_chunks_start_and_end(chunks, text)

        for i in range(len(positions) - 1):
            assert positions[i][1] == positions[i + 1][0]


class TestCheckChunkGaps:
    def test_no_gaps(self):
        text = "Hello world."
        chunks = ["Hello ", "world."]
        has_no_gaps = check_chunk_gaps(chunks, text)
        assert has_no_gaps is True

    def test_with_gaps(self):
        text = "Hello beautiful world."
        chunks = ["Hello ", "world."]
        has_no_gaps = check_chunk_gaps(chunks, text)
        assert has_no_gaps is False


class TestRepairGaps:
    def test_repair_covers_full_text(self):
        text = "AAAA BBBB CCCC DDDD"
        chunks = ["AAAA BBBB ", "DDDD"]
        repaired = repair_gaps_between_chunks(chunks, text)

        assert check_chunk_gaps(repaired, text) is True
