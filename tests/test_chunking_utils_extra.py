"""Extra coverage for chunking_utils: count_tokens and gpu_memory_stats."""

import pytest

from adaptive_chunking.chunking_utils import count_tokens, gpu_memory_stats


def test_count_tokens_nonempty_positive_int():
    n = count_tokens("hello world")
    assert isinstance(n, int)
    assert n > 0


def test_count_tokens_empty_is_zero():
    assert count_tokens("") == 0


def test_gpu_memory_stats_raises_importerror_without_torch():
    # torch is not installed in this environment, so the except-ImportError
    # branch must re-raise ImportError.
    with pytest.raises(ImportError):
        gpu_memory_stats()
