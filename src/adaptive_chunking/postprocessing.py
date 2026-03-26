from typing import Callable, Literal
from .chunking_utils import count_tokens
from pathlib import Path
import pandas as pd
import json
import time

def get_page_info(
    pages: dict[int, str],
    chunks: list[str],
    text: str) -> list[list[int]]:
    """Return, for each chunk, the list of page numbers it overlaps (ascending)."""

    if not chunks:
        return []

    # Use your helper to locate each chunk in the full text
    chunk_positions: list[tuple[int, int]] = find_chunks_start_and_end(chunks, text)

    # Build page boundaries in the SAME order the full text was constructed:
    # dicts preserve insertion order (Python 3.7+). Do NOT sort the keys.
    page_bounds: list[tuple[int, int, int]] = []  # (page_num, start, end)
    offset = 0
    for page_num, page_text in pages.items():
        start = offset
        end = start + len(page_text)
        page_bounds.append((page_num, start, end))
        offset = end

    # Interval overlap: [chunk_start, chunk_end) with [p_start, p_end)
    result: list[list[int]] = []
    for chunk_start, chunk_end in chunk_positions:
        pages_for_chunk = [
            int(page_num)
            for page_num, p_start, p_end in page_bounds
            if not (chunk_end <= p_start or chunk_start >= p_end)
        ]
        result.append(sorted(pages_for_chunk))

    return result

def get_title_info(titles: list[dict], chunks: list[str], text: str) -> list[str]:
    """Return title info (str) for each chunk containing the missing titles.
    A missing title is a title that spans across the chunk but is not present in the chunk.
    """

    if not chunks:
        return []

    sorted_titles = sorted(titles, key=lambda t: (t.get("level", 1), t["start"])) # sort by heading level (outer-most first) and break ties by document order

    chunk_positions = find_chunks_start_and_end(chunks, text)

    result: list[str] = []
    for chunk, (chunk_start, _chunk_end) in zip(chunks, chunk_positions):
        add_titles: list[str] = []
        for t in sorted_titles:
            if t["start"] <= chunk_start < t["end"]:
                title_text: str = t["title"]
                if chunk.find(title_text) == -1:
                    add_titles.append(title_text)
        result.append("\n".join(add_titles))

    return result

def check_chunk_gaps(chunks: list[str], text: str) -> bool:
    """
    Return True iff `chunks`, in order, cover every character of `text`
    with *no* gaps, allowing any amount of overlap.
    """

    if not chunks:
        return len(text) == 0

    end_of_previous = 0

    for chunk in chunks:
        # try a local backward search
        search_start_at = max(0, end_of_previous - len(chunk))
        search_stop_at  = end_of_previous + len(chunk)
        idx = text.rfind(chunk, search_start_at, search_stop_at)

        # if not found, fall back to a forward search 
        if idx == -1:
            search_start_at = search_stop_at
            idx = text.find(chunk, search_start_at)
            if idx == -1:
                return False

        # gap check
        if idx > end_of_previous:
            return False

        # extend the frontier as far as this chunk reaches
        end_of_previous = max(end_of_previous, idx + len(chunk))

    # check if there is no trailing tail
    return end_of_previous == len(text)

def find_chunks_start_and_end(chunks: list[str], text: str) -> list[tuple[int, int]]:
    if not chunks:
        return None

    end_of_previous_chunk = 0
    starts_and_ends = []

    for chunk in chunks:
        # try to find the start of the current chunk before the end of the previous chunk
        search_start_at = max(0, end_of_previous_chunk - len(chunk))
        search_stop_at = end_of_previous_chunk + len(chunk)
        current_start_index = text.rfind(chunk, search_start_at, search_stop_at) # backward search

        # if not found, try to find the start of the current chunk in the full text
        if current_start_index == -1:
            search_start_at = end_of_previous_chunk
            search_stop_at = len(text)
            current_start_index = text.find(chunk, search_start_at, search_stop_at) # forward search

            if current_start_index == -1:
                raise ValueError("Chunk not found in text.")
        
        starts_and_ends.append((current_start_index, current_start_index + len(chunk)))

        end_of_previous_chunk = current_start_index + len(chunk)

    return starts_and_ends

def repair_gaps_between_chunks(chunks: list[str], text: str) -> list[str] | None:
    """
    Repair gaps between consecutive chunks, if any.
    """

    if not chunks:
        return None

    starts_and_ends = find_chunks_start_and_end(chunks, text)

    repaired = []
    end_of_previous_chunk = 0

    for chunk, (chunk_start, chunk_end) in zip(chunks, starts_and_ends):

        text_gap = text[end_of_previous_chunk:chunk_start] if chunk_start > end_of_previous_chunk else ""

        repaired.append(text_gap + chunk)
        end_of_previous_chunk = chunk_end

    if repaired and end_of_previous_chunk < len(text):
        repaired[-1] += text[end_of_previous_chunk:]

    return repaired

def split_oversized_chunks(
    chunks: list[str],
    splitter: Callable[[str], list[str]],
    count_tokens_func: Callable[[str], int],
    max_chunk_tokens: int = 1200,
    ) -> list[str]:
    """
    Split chunks that are too long into smaller chunks using the given splitter.
    """
    final_splits = []
    for chunk in chunks:
        if count_tokens_func(chunk) > max_chunk_tokens:
            sub_chunks = splitter.split_text(chunk)
            final_splits.extend(sub_chunks)
        else:
            final_splits.append(chunk)

    return final_splits

def merge_small_chunks_smallest_first(
    chunks: list[str],
    count_tokens_func: Callable[[str], int],
    min_limit: int = 100,
    max_limit: int = 1200
    ) -> list[str]:
    """
    Merge chunks that are too small into larger chunks in a greedy fashion merging to the smallest neighbour first.
    """

    if len(chunks) < 2:
        return chunks

    merged_chunks = list(chunks)
    merged_chunks_lens = [count_tokens_func(chunk) for chunk in chunks]

    while True:
        merging_occurred_in_pass = False
        i = 0
        
        while i < len(merged_chunks):
            if merged_chunks_lens[i] >= min_limit:
                i += 1
                continue

            left_neighbor_size = float('inf')
            can_merge_left = (
                i > 0 and
                (merged_chunks_lens[i] + merged_chunks_lens[i - 1]) <= max_limit
            )
            if can_merge_left:
                left_neighbor_size = merged_chunks_lens[i - 1]

            right_neighbor_size = float('inf')
            can_merge_right = (
                i < len(merged_chunks) - 1 and
                (merged_chunks_lens[i] + merged_chunks_lens[i + 1]) <= max_limit
            )
            if can_merge_right:
                right_neighbor_size = merged_chunks_lens[i + 1]

            if not can_merge_left and not can_merge_right:
                i += 1
                continue

            if left_neighbor_size < right_neighbor_size: # merge to left 
                merged_chunks[i - 1] += merged_chunks[i]
                merged_chunks_lens[i - 1] += merged_chunks_lens[i]
                del merged_chunks[i]
                del merged_chunks_lens[i]
            else: # merge to right
                merged_chunks[i] += merged_chunks[i + 1]
                merged_chunks_lens[i] += merged_chunks_lens[i + 1]
                del merged_chunks[i + 1]
                del merged_chunks_lens[i + 1]
            
            merging_occurred_in_pass = True
        
        if not merging_occurred_in_pass:
            break

    return merged_chunks

def merge_small_chunks_to_neighbours(
    chunks: list[str],
    count_tokens_func: Callable[[str], int] = count_tokens,
    min_limit: int = 100,
    max_limit: int = 1200,
    merge_to: Literal["previous", "next"] = "next"
    ) -> list[str]:
    """
    Merge chunks that are too small to their previous or next neighbour in a greedy fashion.
    """

    if len(chunks) < 2:
        return chunks

    chunks = chunks[:]  # work on a copy to avoid mutating the input

    size = lambda idx: count_tokens_func(chunks[idx])

    changed = True
    while changed:
        changed = False
        i = 0
        while i < len(chunks):
            if size(i) < min_limit:
                if merge_to == "previous":
                    if i > 0 and size(i - 1) + size(i) <= max_limit:
                        chunks[i - 1] += chunks[i]
                        del chunks[i]
                        changed = True
                        continue
                    if i < len(chunks) - 1 and size(i) + size(i + 1) <= max_limit:
                        chunks[i] += chunks[i + 1]
                        del chunks[i + 1]
                        changed = True
                        continue
                else:  # merge_to == "next"
                    if i < len(chunks) - 1 and size(i) + size(i + 1) <= max_limit:
                        chunks[i] += chunks[i + 1]
                        del chunks[i + 1]
                        changed = True
                        continue
                    if i > 0 and size(i - 1) + size(i) <= max_limit:
                        chunks[i - 1] += chunks[i]
                        del chunks[i]
                        changed = True
                        continue
            i += 1

    if len(chunks) > 1 and size(0) < min_limit and size(0) + size(1) <= max_limit:
        chunks[0] += chunks[1]
        del chunks[1]

    if len(chunks) > 1 and size(-1) < min_limit and size(-1) + size(-2) <= max_limit:
        chunks[-2] += chunks[-1]
        chunks.pop()

    return chunks

def split_oversized_chunks_from_df(
    parsed_docs_dir: str | Path,
    chunks_path: str | Path,
    output_dir: str | Path,
    methods_to_be_regularized: set,
    split_oversized_func: Callable[[list[str]], list[str]] = split_oversized_chunks,
    count_tokens_func: Callable[[str], int] = count_tokens,
    replace_all_results: bool = False):

    # Read parsed docs dir
    parsed_docs_dir = Path(parsed_docs_dir)
    print(f"Loading parsed docs from {parsed_docs_dir}\n")

    parsed_docs = {}
    for json_path in parsed_docs_dir.glob("*.json"):
        with json_path.open("r") as f:
            parsed_docs[json_path.with_suffix('').name] = json.load(f)
    
    if len(parsed_docs) == 0:
        print(f"No parsed docs found in {parsed_docs_dir.name}")
        return None

    # Load chunks df
    chunks_path = Path(chunks_path)
    df = pd.read_parquet(chunks_path)
    print(f"Loading chunks from {chunks_path}\n")
    # dict to store timing per doc/method
    time_per_doc_per_method = {}

    raw_splits_per_doc = {}
    for doc_name, group in df.groupby("doc_name"):
        raw_splits_per_doc[doc_name] = {}
        for method, sub_group in group.groupby("method"):
            raw_splits_per_doc[doc_name][method] = sub_group.chunk_text.to_list()

    regularized_splits_per_doc = {}
    for doc_name in raw_splits_per_doc:
        regularized_splits_per_doc[doc_name] = {}
        for method, chunks in raw_splits_per_doc[doc_name].items():
            start_time_method = time.time()
            # Split oversized chunks and repair any chunking between chunks gaps
            if method in methods_to_be_regularized:
                print(f"Splitting oversized chunks: {doc_name}, {method}")
                full_text = parsed_docs[doc_name]["full_text"]
                repaired_splits = repair_gaps_between_chunks(
                    chunks=split_oversized_func(chunks), text=full_text)
                recover_success = check_chunk_gaps(repaired_splits, full_text)
                assert recover_success==True, "Oversized splitting gap recovery failed"
            else:
                repaired_splits = chunks

            regularized_splits_per_doc[doc_name][method] = repaired_splits
            time_spent = time.time() - start_time_method
            time_per_doc_per_method.setdefault(doc_name, {})[method] = time_spent

    # Format and save chunks parquet
    records = []
    for doc_name, methods in regularized_splits_per_doc.items():
        for method, chunks in methods.items():
            chunks_title_info = get_title_info(titles=parsed_docs[doc_name]["titles"], chunks=chunks, text=parsed_docs[doc_name]["full_text"])
            chunks_page_info = get_page_info(pages=parsed_docs[doc_name]["pages"], chunks=chunks, text=parsed_docs[doc_name]["full_text"])

            if chunks:
                for i, chunk_text in enumerate(chunks):
                    records.append({
                        "doc_name": doc_name,
                        "method": method,
                        "type": "no_oversizing",
                        "chunk_index": i,
                        "chunk_text": chunk_text,
                        "chunk_pages": chunks_page_info[i],
                        "titles_context": chunks_title_info[i],
                        "chunk_len": count_tokens_func(chunk_text),
                    })

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    chunks_output_path = output_dir / "chunks.parquet"
    performances_output_path = output_dir / "performances.parquet"
    chunks_df = pd.DataFrame(records)

    # Merge with existing results if we only want to replace selected methods (chunks)
    if not replace_all_results and chunks_output_path.exists():
        try:
            print(f"Found existing chunks parquet. Replacing results for methods: {methods_to_be_regularized}")
            existing_df = pd.read_parquet(chunks_output_path)
            existing_df = existing_df[~existing_df["method"].isin(methods_to_be_regularized)]
            chunks_df = pd.concat([existing_df, chunks_df], ignore_index=True)
        except Exception as e:
            print(f"Warning: failed to read/merge existing parquet at {chunks_output_path}. Overwriting. Error: {e}")
    chunks_df.to_parquet(chunks_output_path)

    # Build performances dataframe
    perf_records = []
    for doc_name, method_dict in time_per_doc_per_method.items():
        for method, t in method_dict.items():
            perf_records.append({"doc_name": doc_name, "method": method, "time": t})

    performances_df = pd.DataFrame(perf_records)

    # Merge with existing performances if needed
    if not replace_all_results and performances_output_path.exists():
        try:
            existing_perf_df = pd.read_parquet(performances_output_path)
            existing_perf_df = existing_perf_df[~existing_perf_df["method"].isin(methods_to_be_regularized)]
            performances_df = pd.concat([existing_perf_df, performances_df], ignore_index=True)
        except Exception as e:
            print(f"Warning: failed to merge performances parquet at {performances_output_path}. Overwriting. Error: {e}")

    performances_df.to_parquet(performances_output_path)

def merge_small_chunks_from_df(
    parsed_docs_dir: str | Path,
    chunks_path: str | Path,
    output_dir: str | Path,
    methods_to_be_regularized: set,
    merge_small_chunks_func: Callable[[list[str]], list[str]] = merge_small_chunks_to_neighbours,
    count_tokens_func: Callable[[str], int] = count_tokens,
    replace_all_results: bool = False):

    # Read parsed docs dir
    parsed_docs_dir = Path(parsed_docs_dir)
    print(f"Loading parsed docs from {parsed_docs_dir}\n")

    parsed_docs = {}
    for json_path in parsed_docs_dir.glob("*.json"):
        with json_path.open("r") as f:
            parsed_docs[json_path.with_suffix('').name] = json.load(f)
    
    if len(parsed_docs) == 0:
        print(f"No parsed docs found in {parsed_docs_dir.name}")
        return None
    
    # Load chunks df
    chunks_path = Path(chunks_path)
    df = pd.read_parquet(chunks_path)
    print(f"Loading chunks from {chunks_path}\n")

    raw_splits_per_doc = {}
    for doc_name, group in df.groupby("doc_name"):
        raw_splits_per_doc[doc_name] = {}
        for method, sub_group in group.groupby("method"):
            raw_splits_per_doc[doc_name][method] = sub_group.chunk_text.to_list()

    regularized_splits_per_doc = {}
    # dict to store timing per doc/method
    time_per_doc_per_method = {}
    for doc_name in raw_splits_per_doc:
        regularized_splits_per_doc[doc_name] = {}
        for method, chunks in raw_splits_per_doc[doc_name].items():
            start_time_method = time.time()
            if method in methods_to_be_regularized:
                # Merge small chunks and repair any chunking gaps
                print(f"Merging small chunks: {doc_name}, {method}")
                full_text = parsed_docs[doc_name]["full_text"]

                repaired_splits = repair_gaps_between_chunks(
                    chunks=merge_small_chunks_func(chunks), text=full_text)

                recover_success = check_chunk_gaps(repaired_splits, full_text)
                assert recover_success==True, "Small chunks merging gap recovery failed"
            else:
                repaired_splits = chunks
            
            regularized_splits_per_doc[doc_name][method] = repaired_splits
            time_spent = time.time() - start_time_method
            time_per_doc_per_method.setdefault(doc_name, {})[method] = time_spent

    # Format and save chunks parquet
    records = []
    for doc_name, methods in regularized_splits_per_doc.items():
        for method, chunks in methods.items():
            chunks_title_info = get_title_info(titles=parsed_docs[doc_name]["titles"], chunks=chunks, text=parsed_docs[doc_name]["full_text"])
            chunks_page_info = get_page_info(pages=parsed_docs[doc_name]["pages"], chunks=chunks, text=parsed_docs[doc_name]["full_text"])
            if chunks:
                for i, chunk_text in enumerate(chunks):
                    records.append({
                        "doc_name": doc_name,
                        "method": method,
                        "type": "no_small_chunks",
                        "chunk_index": i,
                        "chunk_text": chunk_text,
                        "chunk_pages": chunks_page_info[i],
                        "titles_context": chunks_title_info[i],
                        "chunk_len": count_tokens_func(chunk_text), # compute chunk lens
                    })

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    chunks_output_path = output_dir / "chunks.parquet"
    performances_output_path = output_dir / "performances.parquet"
    chunks_df = pd.DataFrame(records)

    # Merge with existing results for chunks
    if not replace_all_results and chunks_output_path.exists():
        try:
            print(f"Found existing chunks parquet. Replacing results for methods: {methods_to_be_regularized}")
            existing_df = pd.read_parquet(chunks_output_path)
            existing_df = existing_df[~existing_df["method"].isin(methods_to_be_regularized)]
            chunks_df = pd.concat([existing_df, chunks_df], ignore_index=True)
        except Exception as e:
            print(f"Warning: failed to read/merge existing parquet at {chunks_output_path}. Overwriting. Error: {e}")
    chunks_df.to_parquet(chunks_output_path)

    # Build performances df
    perf_records = []
    for doc_name, method_dict in time_per_doc_per_method.items():
        for method, t in method_dict.items():
            perf_records.append({"doc_name": doc_name, "method": method, "time": t})

    performances_df = pd.DataFrame(perf_records)

    if not replace_all_results and performances_output_path.exists():
        try:
            existing_perf_df = pd.read_parquet(performances_output_path)
            existing_perf_df = existing_perf_df[~existing_perf_df["method"].isin(methods_to_be_regularized)]
            performances_df = pd.concat([existing_perf_df, performances_df], ignore_index=True)
        except Exception as e:
            print(f"Warning: failed to merge performances parquet at {performances_output_path}. Overwriting. Error: {e}")

    performances_df.to_parquet(performances_output_path)