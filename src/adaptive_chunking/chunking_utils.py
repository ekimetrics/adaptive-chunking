import tiktoken


def count_tokens(text: str, model: str = "gpt-4o") -> int:
    """
    Counts the number of tokens in the given text using the specified model's tokenizer.

    Parameters:
        text (str): The input text to count tokens for.
        model (str): The model name to use for tokenization.

    Returns:
        int: The number of tokens in the input text.
    """
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))


def is_high_confidence_non_english(text: str, threshold: float = 0.98) -> bool:
    """Return True only when `text` is detected as non-English with high confidence.

    Used to filter documents before English-only processing. When the language
    cannot be determined — no detection result, or langdetect raises — this
    returns False so the caller keeps the document rather than dropping it.

    This replaces an inline check that read `confidence`/`lang_code` outside the
    guard that assigned them, which raised NameError on the first empty result
    and otherwise leaked a previous document's values across loop iterations.
    """
    from langdetect import detect_langs, LangDetectException

    try:
        lang_probs = detect_langs(text[:50000])
    except LangDetectException:
        return False

    if not lang_probs:
        return False

    top = lang_probs[0]
    return top.prob >= threshold and top.lang != "en"


def gpu_memory_stats():
    try:
        import torch
    except ImportError:
        raise ImportError("torch is required for gpu_memory_stats(). Install with: pip install torch")
    print(f"\nAllocated memory: {torch.cuda.memory_allocated()/1024**2} MB")
    print(f"Reserved memory: {torch.cuda.memory_reserved()/1024**2} MB\n")
