import pytest


def word_count(text: str) -> int:
    """Simple word-based token counter for tests (no tiktoken dependency)."""
    return len(text.split())


@pytest.fixture
def token_counter():
    return word_count
