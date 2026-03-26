from typing import Callable, List, Literal, Tuple
import re

from .chunking_utils import count_tokens


class RecursiveSplitter:
    """
    Splits text recursively into chunks.

    This splitter works by recursively trying to split text using a list of
    separators. It can operate in two modes:
    1.  `merging="to_chunk_size"`: Merges small splits into larger chunks of approximately
        `chunk_size`, adding overlap between them if needed.
    2.  `merging="small_only"`: Keeps splits separated by semantic boundaries (the
        separators), only merging chunks that are smaller than
        `min_chunk_tokens` with their direct neighbor.
        Adds overlap between chunks if wanted.
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 0,
        length_function: Callable[[str], int] = count_tokens,
        attach_separator_to: Literal["start", "end"] = "start",
        is_separator_regex: bool = False,
        separators: List[str] = ["\n\n", "\n", " ", ""],
        merging: Literal["to_chunk_size", "small_only"] = "to_chunk_size",
        max_tokens_strategy: Literal["chunk_size", "chunk_size_plus_overlap"] = "chunk_size_plus_overlap",
        min_chunk_tokens: int = 100,
        merging_order: Literal["forward", "backward"] = "forward"):

        if chunk_overlap > chunk_size:
            raise ValueError(
                f"Chunk overlap ({chunk_overlap}) cannot be greater than chunk "
                f"size ({chunk_size})."
            )
        if min_chunk_tokens < 0:
            raise ValueError(
                f"Minimum chunk tokens ({min_chunk_tokens}) cannot be negative."
            )
        if merging_order not in ("forward", "backward"):
            raise ValueError(
                f"Invalid merging_order: {merging_order}. Expected 'forward' or 'backward'."
            )

        # set the maximum chunk size based on the max_tokens_strategy
        if max_tokens_strategy == "chunk_size_plus_overlap": # never produces splits bigger than chunk_size + chunk_overlap
            self.chunk_size  = chunk_size
        elif max_tokens_strategy == "chunk_size": # never produces splits bigger than chunk_size
            self.chunk_size = chunk_size - chunk_overlap
        else:
            raise ValueError(f"Invalid max_tokens_strategy: {max_tokens_strategy}")

        self.chunk_overlap = chunk_overlap
        self.length_function = length_function
        self.attach_separator_to = attach_separator_to
        self.is_separator_regex = is_separator_regex
        self.separators = separators
        self.merging = merging
        self.merging_order = merging_order
        self.min_chunk_tokens = min_chunk_tokens

    def _split_with_separator(
        self,
        text: str,
        regex_pattern: str,
        min_len: int = 10) -> List[str]:
        """
        Splits *text* on *regex_pattern*, keeps the separator with the chunk
        (appended to the **end** or prepended to the **start**), then merges any
        chunk whose length is strictly below *min_len* with its neighbour.

        Robust against patterns that contain capturing groups.
        """
        pieces = []
        last_idx = 0
        try:
            for match in re.finditer(regex_pattern, text):
                pieces.append(text[last_idx:match.start()])
                pieces.append(match.group(0))
                last_idx = match.end()
            pieces.append(text[last_idx:])
        except re.error as e:
            raise ValueError(f"Invalid regex pattern: {regex_pattern}") from e

        if len(pieces) <= 1:
            return [p for p in pieces if p]

        if self.attach_separator_to == "end":
            chunks, i = [], 0
            while i < len(pieces):
                separator = pieces[i + 1] if i + 1 < len(pieces) else ""
                chunk = pieces[i] + separator
                i += 2
                if chunk:
                    chunks.append(chunk)

            merged = []
            for c in chunks:
                if len(c) < min_len and merged:
                    merged[-1] += c
                else:
                    merged.append(c)
            return merged

        elif self.attach_separator_to == "start":
            chunks = [pieces[0]] if pieces[0] else []
            i = 1
            while i < len(pieces):
                text_segment = pieces[i + 1] if i + 1 < len(pieces) else ""
                chunk = pieces[i] + text_segment
                chunks.append(chunk)
                i += 2

            merged, idx = [], 0
            while idx < len(chunks):
                if len(chunks[idx]) < min_len and idx + 1 < len(chunks):
                    chunks[idx + 1] = chunks[idx] + chunks[idx + 1]
                else:
                    merged.append(chunks[idx])
                idx += 1
            return merged
        else:
            raise ValueError('attach_separator_to must be "start" or "end"')

    def _recursive_split(self, text: str, separators: List[str], max_length: int) -> List[str]:
        """Recursively splits text until all chunks are smaller than chunk_size."""
        if not text:
            return []

        # If text is already small enough, return it as a single chunk
        if self.length_function(text) <= max_length:
            return [text]

        # Use the first available separator
        current_separator = separators[0]
        remaining_separators = separators[1:]

        # Handle the special case of an empty string separator using binary search
        if current_separator == "":
            chunks = []
            text_to_split = text

            while text_to_split:
                # Use a binary search to find the optimal character index
                # 'low' is the minimum possible chunk length (0 chars)
                # 'high' is the maximum possible chunk length (all remaining chars)
                low, high = 0, len(text_to_split)
                best_end = 0 # Stores the end index of the best chunk found so far.

                while low <= high:
                    mid = (low + high) // 2
                    # Create a potential chunk from the start to the middle index.
                    potential_chunk = text_to_split[:mid]

                    # Measure the token count of this potential chunk.
                    if self.length_function(potential_chunk) <= max_length:
                        # This chunk is valid (doesn't exceed the size).
                        # It might be our best option so far, so we store its end index.
                        best_end = mid
                        # We try to find a larger valid chunk by searching in the upper half.
                        low = mid + 1
                    else:
                        # This chunk is too large. We need to search for a smaller one
                        # in the lower half.
                        high = mid - 1

                # If no valid chunk could be found (e.g., the first character is > chunk_size),
                # we must take at least one character to advance.
                if best_end == 0:
                    best_end = 1

                # Extract the optimal chunk.
                chunk = text_to_split[:best_end]
                chunks.append(chunk)

                # Prepare for the next iteration.
                text_to_split = text_to_split[best_end:]
            return chunks

        sep_pattern = (
            current_separator if self.is_separator_regex else re.escape(current_separator)
        )

        # Split the text using the current separator
        splits = self._split_with_separator(text=text, regex_pattern=sep_pattern)

        # Recurse on any split that is still too large
        final_chunks = []
        for split in splits:
            if self.length_function(split) > max_length:
                if remaining_separators:
                    # Recurse with the next level of separators
                    final_chunks.extend(
                        self._recursive_split(split, remaining_separators, max_length=max_length)
                    )
                else:
                    # No more separators, force a hard split
                    final_chunks.extend(self._recursive_split(split, [""], max_length=max_length))
            else:
                final_chunks.append(split)

        return final_chunks

    def _merge_splits(self, splits: List[str]) -> List[str]:
        """
        Merges small splits into larger chunks and introduces overlap if needed.
        This mode aims to create chunks of approximately `chunk_size`.
        """
        # Adjust processing order based on the desired merging direction
        if self.merging_order == "backward":
            splits = list(reversed(splits))

        final_chunks = []
        current_chunk_parts = []
        current_length = 0

        for split in splits:
            split_len = self.length_function(split)

            # If adding the new split exceeds the chunk size, finalize the current chunk
            if current_length + split_len > self.chunk_size and current_chunk_parts:
                # Preserve original text order inside each chunk when merging backward
                if self.merging_order == "backward":
                    chunk = "".join(reversed(current_chunk_parts))
                else:
                    chunk = "".join(current_chunk_parts)
                final_chunks.append(chunk)

                # Start a new chunk, creating overlap by backtracking
                overlap_parts = []
                overlap_len = 0
                min_overlap_len = int(0.5*self.chunk_overlap) # define the minimum overlap length

                for i, part in enumerate(reversed(current_chunk_parts)):
                    part_len = self.length_function(part)

                    # if a part is larger than chunk_overlap AND it is the first part or the current overlap is smaller
                    #  than min_overlap_len, we resplit that part recursively
                    if part_len > self.chunk_overlap:
                        if i == 0 or overlap_len < min_overlap_len:
                            remaining = self.chunk_overlap - overlap_len
                            subparts = self._recursive_split(text=part, separators=self.separators, max_length=remaining)
                            for subpart in reversed(subparts):
                                if overlap_len + self.length_function(subpart) > self.chunk_overlap:
                                    break
                                overlap_parts.insert(0, subpart)
                                overlap_len += self.length_function(subpart)
                            break

                    if overlap_len + part_len > self.chunk_overlap:
                        break

                    overlap_parts.insert(0, part)
                    overlap_len += part_len

                # The new chunk starts with the overlap and the current split
                current_chunk_parts = overlap_parts + [split]
                current_length = self.length_function("".join(current_chunk_parts))

            else:
                current_chunk_parts.append(split)
                current_length += split_len

        # Add the last remaining chunk
        if current_chunk_parts:
            if self.merging_order == "backward":
                final_chunks.append("".join(reversed(current_chunk_parts)))
            else:
                final_chunks.append("".join(current_chunk_parts))

        # If we processed splits backward, reverse the result to maintain original order
        if self.merging_order == "backward":
            final_chunks = list(reversed(final_chunks))

        return final_chunks

    def _merge_small_splits(self, splits: List[str]) -> List[str]:
        """
        Merges small chunks and adds overlap, preserving the exact overlap logic
        from _merge_splits by operating on the constituent parts of each chunk.
        """
        # Adjust processing order based on the desired merging direction
        if self.merging_order == "backward":
            splits = list(reversed(splits))

        if not splits:
            return []

        if len(splits) == 1:
            return splits

        # Group small splits with their neighbors, but keep them as parts
        grouped_parts: List[List[str]] = [[s] for s in splits]

        size = lambda idx: sum(self.length_function(p) for p in grouped_parts[idx])

        changed = True
        while changed:
            changed = False
            i = 0
            while i < len(grouped_parts):
                if size(i) < self.min_chunk_tokens:
                    # try to merge with the *next* group
                    if i < len(grouped_parts) - 1 and size(i) + size(i + 1) <= self.chunk_size:
                        grouped_parts[i].extend(grouped_parts[i + 1])
                        del grouped_parts[i + 1]
                        changed = True
                        continue
                    # fall back on the previous group
                    if i > 0 and size(i - 1) + size(i) <= self.chunk_size:
                        grouped_parts[i - 1].extend(grouped_parts[i])
                        del grouped_parts[i]
                        changed = True
                        continue
                i += 1

        # If no overlap, just join the groups and return
        if self.chunk_overlap < 1:
            return ["".join(group) for group in grouped_parts]

        # Join groups and create overlap from the parts, matching _merge_splits logic
        final_chunks = []
        # Start with no overlap for the first chunk
        overlap_parts = []

        for group in grouped_parts:
            # The parts for the current chunk consist of the overlap from the previous
            # chunk plus the new group of parts.
            parts_for_this_chunk = overlap_parts + group
            if self.merging_order == "backward":
                chunk = "".join(reversed(parts_for_this_chunk))
            else:
                chunk = "".join(parts_for_this_chunk)
            final_chunks.append(chunk)

            # Now, create the overlap FOR THE NEXT chunk by backtracking over the
            # parts we just used for THIS chunk.
            new_overlap_parts = []
            overlap_len = 0
            min_overlap_len = int(0.5*self.chunk_overlap) # define the minimum overlap length

            for i, part in enumerate(reversed(parts_for_this_chunk)):
                part_len = self.length_function(part)

                # if a part is larger than chunk_overlap AND it is the first part or the current overlap is smaller
                #  than min_overlap_len, we resplit that part recursively
                if part_len > self.chunk_overlap:
                    if i == 0 or overlap_len < min_overlap_len:
                        remaining = self.chunk_overlap - overlap_len
                        subparts = self._recursive_split(text=part, separators=self.separators, max_length=remaining)
                        for subpart in reversed(subparts):
                            if overlap_len + self.length_function(subpart) > self.chunk_overlap:
                                break
                            new_overlap_parts.insert(0, subpart)
                            overlap_len += self.length_function(subpart)
                        break

                if overlap_len + part_len > self.chunk_overlap:
                    break

                new_overlap_parts.insert(0, part)
                overlap_len += part_len

            # This overlap will be used in the next iteration of the loop
            overlap_parts = new_overlap_parts

        # If we processed splits backward, reverse the result to maintain original order
        if self.merging_order == "backward":
            final_chunks = list(reversed(final_chunks))

        return final_chunks

    def split_text(self, text: str) -> List[str]:
        """
        The main entry point for splitting text.
        """
        # First, perform the recursive split to get initial, non-overlapping chunks
        initial_splits = self._recursive_split(text=text, separators=self.separators, max_length=self.chunk_size)

        if not initial_splits:
            return []

        # Apply the selected strategy: merging or handling small chunks
        if self.merging == "to_chunk_size":
            return self._merge_splits(splits=initial_splits)
        elif self.merging == "small_only":
            return self._merge_small_splits(splits=initial_splits)
        else:
            raise ValueError(f"Invalid merging strategy: {self.merging}")


def group_chunks(blocks: list[str], tokenizer_func: Callable[[str], list], max_tokens: int, chunk_block_overlap: int = 0, verbose: bool = False) -> tuple[list[str], list[list[str]]]:
    """
    Group pre-chunked text blocks into overlapping chunks without cutting blocks.
    Assumes blocks are already pre-chunked.

    Parameters:
      blocks (List[str]): List of text blocks (e.g., sentences or paragraphs).
      max_tokens (int): Maximum tokens allowed in a chunk.
      tokenizer_func: A function that takes a string and returns a list of tokens.
      chunk_block_overlap (int): Number of blocks to overlap between chunks.

    Returns:
      List[str]: List of text chunks (each chunk is the concatenation of full blocks).
      List[str]: List of grouped blocks.

    """

    def _crop_block_to_max_tokens(
        block: str,
        max_tokens: int,
        tokenizer_func: Callable[[str], list]) -> str:
        """
        Crops a text block to a maximum token limit using a binary search.
        """
        if not block:
            return ""

        low = 0
        high = len(block)
        best_split = 0

        while low <= high:
            mid = (low + high) // 2
            substring = block[:mid]

            num_tokens = len(tokenizer_func(substring)) if substring else 0

            if num_tokens <= max_tokens:
                best_split = mid
                low = mid + 1
            else:
                high = mid - 1

        return block[:best_split]

    token_counts = [len(tokenizer_func(block)) for block in blocks]

    chunks = []
    grouped_blocks = []
    current_chunk = []
    current_chunk_token_count = 0

    for i, block in enumerate(blocks):
        block_token_count = token_counts[i]

        # Check if the block is too long
        if block_token_count > max_tokens:
            if verbose:
                print(f"Block exceed max_tokens with {block_token_count} tokens, block will be cropped to max_tokens. Increase max_tokens or reduce block size if needed.")
            # save any chunk that was being built
            if current_chunk:
                chunks.append("".join(current_chunk))
                grouped_blocks.append(current_chunk)

            # crop oversized block
            cropped_block = _crop_block_to_max_tokens(block, max_tokens, tokenizer_func)
            if cropped_block:
                chunks.append(cropped_block)
                grouped_blocks.append([cropped_block])

            # reset auxiliars
            current_chunk = []
            current_chunk_token_count = 0


        # Check if adding this block would exceed the max token count
        elif current_chunk_token_count + block_token_count > max_tokens:
            if current_chunk:
                chunks.append("".join(current_chunk))
                grouped_blocks.append(current_chunk)

            # Prepare the overlap for the next chunk
            overlap_start = max(0, len(current_chunk) - chunk_block_overlap)
            overlap = current_chunk[overlap_start:]

            # Start the new chunk with the overlap blocks
            current_chunk = overlap.copy()
            current_chunk_token_count = sum(token_counts[i - len(overlap) + j] for j in range(len(overlap)))

            # Add the current block if there's space after the overlap
            if current_chunk_token_count + block_token_count <= max_tokens:
                current_chunk.append(block)
                current_chunk_token_count += block_token_count
        else:
            current_chunk.append(block)
            current_chunk_token_count += block_token_count

    # Add the final chunk only if it contains more than just overlap blocks
    if current_chunk and (len(current_chunk) > chunk_block_overlap or chunk_block_overlap == 0):
        chunks.append("".join(current_chunk))
        grouped_blocks.append(current_chunk)

    return chunks, grouped_blocks


def group_pages(doc_pages: list[str], pages_per_group: int = 2, overlap_lines: int = 10) -> list[str]:

    grouped_pages = []
    current_group = []

    for page in doc_pages:
        current_group.append(page)
        if len(current_group) >= pages_per_group:
            if grouped_pages:
                past_group_lines = grouped_pages[-1].splitlines(keepends=True)
                if overlap_lines > 0:
                    # Add context from the last group to the current group
                    context_text = "".join(past_group_lines[-overlap_lines:])
                    grouped_pages.append(context_text + "".join(current_group))
                else:
                    grouped_pages.append("".join(current_group))
            else:
                grouped_pages.append("".join(current_group))
            current_group = []

    # Add any remaining pages
    if current_group:
        if grouped_pages:
            past_group_lines = grouped_pages[-1].splitlines(keepends=True)
            if overlap_lines > 0:
                # Add context from the last group to the current group
                context_text = "".join(past_group_lines[-overlap_lines:])
                grouped_pages.append(context_text + "".join(current_group))
            else:
                grouped_pages.append("".join(current_group))
        else:
            grouped_pages.append("".join(current_group))

    return grouped_pages


def combine_blocks(blocks: list[str], max_tokens: int, count_tokens_func: Callable[[str], int]) -> str:
    """
    Concatenates document blocks (i.g. lines, paragraphs, pages) into a single string,
    ensuring the total token count does not exceed max_tokens.
    """
    current_context = []
    current_tokens = 0

    for i, block in enumerate(blocks):
        block_tokens = count_tokens_func(block)

        if current_tokens + block_tokens <= max_tokens:
            current_context.append(block)
            current_tokens += block_tokens
        else:
            break

    return "".join(current_context)


def regex_splitter(text: str,
                   regex_pattern: str,
                   attach_to: str = "start",
                   min_len: int = 10) -> list[str]:
    """
    Splits *text* on *regex_pattern*, keeps the separator with the chunk
    (appended to the **end** or prepended to the **start**), then merges any
    chunk whose length is strictly below *min_len* with its neighbour,
    following the same **attach_to** rule.

    *robust against patterns that contain capturing groups.
    """
    pieces = []
    last_idx = 0
    try:
        for match in re.finditer(regex_pattern, text):
            pieces.append(text[last_idx:match.start()])
            pieces.append(match.group(0))
            last_idx = match.end()
        pieces.append(text[last_idx:])
    except re.error as e:
        print(f"Error compiling or using regex pattern: {e}")
        return [text]

    if len(pieces) <= 1:
        return [p for p in pieces if p]

    if attach_to == "end":
        chunks, i = [], 0
        while i < len(pieces):
            separator = pieces[i + 1] if i + 1 < len(pieces) else ""
            chunk = pieces[i] + separator
            i += 2
            if chunk:
                chunks.append(chunk)

        merged = []
        for c in chunks:
            if len(c) < min_len and merged:
                merged[-1] += c
            else:
                merged.append(c)
        return merged

    if attach_to == "start":
        chunks = [pieces[0]] if pieces[0] else []
        i = 1
        while i < len(pieces):
            text_segment = pieces[i + 1] if i + 1 < len(pieces) else ""
            chunk = pieces[i] + text_segment
            chunks.append(chunk)
            i += 2

        merged, idx = [], 0
        while idx < len(chunks):
            if len(chunks[idx]) < min_len and idx + 1 < len(chunks):
                chunks[idx + 1] = chunks[idx] + chunks[idx + 1]
            else:
                merged.append(chunks[idx])
            idx += 1
        return merged

    raise ValueError('attach_to must be "start" or "end"')
