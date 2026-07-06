import pytest
from adaptive_chunking.metrics import (
    compute_size_compliance,
    compute_filtered_missing_ref_error,
)


class TestSizeCompliance:
    def test_all_in_range(self, token_counter):
        chunks = ["word " * 200] * 5  # 200 words each
        score = compute_size_compliance(
            chunks, count_tokens_func=token_counter,
            min_tokens=100, max_tokens=300
        )
        assert score == 1.0

    def test_all_out_of_range(self, token_counter):
        chunks = ["word " * 5] * 3  # 5 words each, below min
        score = compute_size_compliance(
            chunks, count_tokens_func=token_counter,
            min_tokens=100, max_tokens=300
        )
        assert score == 0.0

    def test_partial(self, token_counter):
        chunks = [
            "word " * 200,  # in range
            "word " * 5,    # below min
        ]
        score = compute_size_compliance(
            chunks, count_tokens_func=token_counter,
            min_tokens=100, max_tokens=300
        )
        assert score == pytest.approx(0.5)

    def test_empty_chunks(self, token_counter):
        score = compute_size_compliance(
            [], count_tokens_func=token_counter,
            min_tokens=100, max_tokens=300
        )
        assert score is None

    def test_custom_bounds(self, token_counter):
        chunks = ["word " * 10] * 3  # 10 words each
        score = compute_size_compliance(
            chunks, count_tokens_func=token_counter,
            min_tokens=5, max_tokens=15
        )
        assert score == 1.0


class TestFilteredMissingRefError:
    # "Alice went home. " is 17 chars, so chunk boundary lands at index 17.
    full_text = "Alice went home. She slept."

    def test_boundary_splits_pair(self):
        chunks = ["Alice went home. ", "She slept."]
        # entity "Alice" (0-4) and pronoun "She" (17-19) straddle the chunk boundary
        pairs = [[(0, 4), (17, 19)]]
        assert compute_filtered_missing_ref_error(self.full_text, chunks, pairs) == 1.0

    def test_no_boundary_no_missing(self):
        chunks = [self.full_text]  # single chunk -> no internal boundaries
        pairs = [[(0, 4), (17, 19)]]
        assert compute_filtered_missing_ref_error(self.full_text, chunks, pairs) == 0.0

    def test_missing_chunk_raises(self):
        # Regression: str.find returns -1 for an absent chunk and never raises,
        # so the sentinel must be checked explicitly rather than via try/except.
        with pytest.raises(ValueError, match="Chunk not found"):
            compute_filtered_missing_ref_error(
                self.full_text, ["totally absent chunk"], [[(0, 4), (17, 19)]]
            )

    def test_no_pairs_returns_none(self):
        chunks = ["Alice went home. ", "She slept."]
        assert compute_filtered_missing_ref_error(self.full_text, chunks, []) is None
