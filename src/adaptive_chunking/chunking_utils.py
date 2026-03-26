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


def gpu_memory_stats():
    try:
        import torch
    except ImportError:
        raise ImportError("torch is required for gpu_memory_stats(). Install with: pip install torch")
    print(f"\nAllocated memory: {torch.cuda.memory_allocated()/1024**2} MB")
    print(f"Reserved memory: {torch.cuda.memory_reserved()/1024**2} MB\n")
