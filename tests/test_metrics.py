import pytest
from adaptive_chunking.metrics import compute_size_compliance


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
