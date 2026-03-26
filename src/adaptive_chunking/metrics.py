import numpy as np
from itertools import combinations
from typing import Callable
import re
from .chunking_utils import count_tokens
from .postprocessing import find_chunks_start_and_end

PERSONAL_PRONOUNS = { # default list of lowercased pronouns used in extract_entity_pronoun_pairs
    "i", "me", "my", "mine", "myself", "we", "us",
    "our", "ours", "ourselves", "you", "your", "yours",
    "yourself", "yourselves", "he", "him", "his", "himself",
    "she", "her", "hers", "herself", "it", "its", "itself", "they",
    "them","their", "theirs","themselves", "one", "oneself"
}

def compute_size_compliance(
    chunks: list[str],
    max_tokens: int = 1100,
    min_tokens: int = 100,
    count_tokens_func: Callable[[str], int] = count_tokens):
    """
    Measures how well the chunks comply to the size constraints.
    """

    if not chunks:
        return None
    
    out_of_span = 0
    for chunk in chunks:
        chunk_len = count_tokens_func(chunk)
        if chunk_len > max_tokens or chunk_len < min_tokens:
            out_of_span += 1

    return 1 - out_of_span / len(chunks)

def compute_chunk_embeddings(
    chunks: list[str],
    model: "SentenceTransformer",
    batch_size: int = 16,
    progress_bar: bool = False,
    normalize_embeddings: bool = True) -> np.ndarray:
    """
    Embed each chunk as a single sentence.
    """
    embeddings = model.encode(chunks,
        batch_size=batch_size,
        show_progress_bar=progress_bar,
        convert_to_numpy=True,
        normalize_embeddings=normalize_embeddings)

    return embeddings

def compute_intrachunk_cohesion(
    chunks: list[str],
    full_text: str,
    split_points: list[int],
    model: "SentenceTransformer",
    chunk_embeddings: np.ndarray | None = None,
    batch_size: int = 16,
    progress_bar: bool = False) -> float | None:
    """
    Computes cohesion inside chunks by comparing each chunk "sentence" block to the corresponding chunk embedding.
    The provided `chunk_embeddings` must be normalized and they are not provided, one embedding is generated per chunk via `compute_chunk_embeddings`.
    Sentence boundaries are provided in `split_points` as character offsets relative to `full_text`.
    The cohesion score for a chunk is defined as the mean cosine similarity between chunk sentences
    and the full chunk embedding.
    """

    # Prepare / compute chunk-level embeddings if not provided
    if chunk_embeddings is None:
        chunk_embeddings = compute_chunk_embeddings(
            chunks,
            model=model,
            batch_size=batch_size,
            progress_bar=progress_bar,
            normalize_embeddings=True,
        )

    # Guard against mismatched lengths
    if len(chunk_embeddings) != len(chunks):
        raise ValueError("chunk_embeddings length must equal number of chunks.")

    # Reconstruct sentences per chunk using the provided split points
    chunk_sentences: list[list[str]] = []
    sorted_splits = sorted(split_points)

    # Locate every chunk in the full_text (chunks are ordered but can overlap)
    chunk_starts_and_ends = find_chunks_start_and_end(chunks=chunks, text=full_text)

    for chunk, (chunk_start, chunk_end) in zip(chunks, chunk_starts_and_ends):
        local_split_points = [
            split_point - chunk_start
            for split_point in sorted_splits
            if chunk_start <= split_point < chunk_end
        ]

        boundaries = sorted({0, *local_split_points, chunk_end})

        sentences: list[str] = []
        for i in range(len(boundaries) - 1):
            start_idx, end_idx = boundaries[i], boundaries[i + 1]
            sentence = chunk[start_idx:end_idx]
            if sentence:
                sentences.append(sentence)

        if not sentences:
            sentences = [chunk]

        chunk_sentences.append(sentences)

    # Embed all sentences in one go for efficiency
    flattened_sentences: list[str] = [s for sent_list in chunk_sentences for s in sent_list]

    # Guard against empty sentences
    if not flattened_sentences:
        return None

    sentence_embeddings = model.encode(
        flattened_sentences,
        batch_size=batch_size,
        show_progress_bar=progress_bar,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    # Compute cohesion per chunk
    cohesion_scores: list[float] = []
    sent_idx = 0
    for idx, sentences in enumerate(chunk_sentences):
        num_sents = len(sentences)

        # if less than 2 sentences, do not include chunk in the cohesion score
        if num_sents < 2:
            sent_idx += num_sents
            continue

        sent_embeds = sentence_embeddings[sent_idx : sent_idx + num_sents]
        sent_idx += num_sents

        # compute cosine similarity between each sentence and the chunk embedding
        sims = np.dot(sent_embeds, chunk_embeddings[idx])
        cohesion_scores.append(np.mean(sims))

    if not cohesion_scores:
        return None

    # Clip to [0, 1]
    return np.clip(np.mean(cohesion_scores), 0.0, 1.0)

def compute_contextual_coherence(
    chunks: list[str],
    full_text: str,
    model: "SentenceTransformer",
    window_context_tokens: int = 3000,
    count_tokens_func: Callable[[str], int] = count_tokens,
    window_step: int = 1,
    batch_size: int = 16,
    chunk_embeddings: np.ndarray | None = None,
    progress_bar: bool = False) -> float | None:
    """
    Compute coherence by comparing chunk embeddings to fixed size window embeddings of text.
    Windows are built so the *sum* of non overlapping token counts in the window is < `window_context_tokens`, and
    the window slides forward `window_step` chunks at a time.

    If `chunk_embeddings` is `None` one embedding is generated per chunk via `compute_chunk_embeddings`, otherwise
    the provided embeddings are used. `chunk_embeddings` must be normalized.
    """
    n = len(chunks)
    if n < 2:
        return None

    if chunk_embeddings is None:
        chunk_embeddings = compute_chunk_embeddings(
            chunks,
            model=model,
            batch_size=batch_size,
            progress_bar=progress_bar,
            normalize_embeddings=True,
        )

    if len(chunk_embeddings) != len(chunks):
        raise ValueError("chunk_embeddings length must equal number of chunks.")

    # Locate each chunk in the full text so we can handle overlaps
    chunk_bounds = find_chunks_start_and_end(chunks=chunks, text=full_text)

    # Sliding‑window builder that never duplicates text
    text_windows              = []
    chunks_indices_per_window = []

    i = 0
    n = len(chunk_bounds)

    while i < n:
        current_end         = chunk_bounds[i][0]  # what we have already added to the window
        window_token_count  = 0 # sum of token counts of chunks in the window
        window_parts        = [] # list of chunks in the window
        chunks_in_window    = [] # list of indices of chunks in the window

        j = i
        while j < n:
            start_j, end_j = chunk_bounds[j]

            # tail that is NOT yet inside the window
            slice_start = max(current_end, start_j)
            slice_end   = end_j

            if slice_start < slice_end:          # something new to add
                tail_tokens = count_tokens_func(full_text[slice_start:slice_end])
            else:                                # chunk fully overlapped
                tail_tokens = 0

            # always keep at least ONE chunk, otherwise respect the limit
            if chunks_in_window and window_token_count + tail_tokens > window_context_tokens:
                break

            # append the unseen tail
            if tail_tokens:
                window_parts.append(full_text[slice_start:slice_end])
                window_token_count += tail_tokens
                current_end = slice_end          # extend covered region

            chunks_in_window.append(j)
            j += 1

        # only append windows if the biggest chunk index has changed (avoid subwindows) and if they have at least 2 chunks
        if len(chunks_indices_per_window) > 0:
            if chunks_in_window[-1] != chunks_indices_per_window[-1][-1] and len(chunks_in_window) > 1:
                text_windows.append("".join(window_parts))
                chunks_indices_per_window.append(chunks_in_window)
        elif len(chunks_in_window) > 1:
            text_windows.append("".join(window_parts))
            chunks_indices_per_window.append(chunks_in_window)

        # slide the window forward
        i += window_step

    if not text_windows:
        return None
    
    # embed all windows once via batching
    window_embeddings = model.encode(
        text_windows,
        batch_size=batch_size,
        show_progress_bar=progress_bar,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    
    # compute cosine similarity between each chunk in the window and the window embedding
    cohesion_scores = []
    for window_idx, window_chunks in enumerate(chunks_indices_per_window):
        window_embed = window_embeddings[window_idx]
        for chunk_idx in window_chunks:
            chunk_embed = chunk_embeddings[chunk_idx]
            sim = np.dot(window_embed, chunk_embed)
            cohesion_scores.append(sim)

    if not cohesion_scores:
        return None

    return np.clip(np.mean(cohesion_scores), 0.0, 1.0)

def compute_block_integrity(
    chunks: list[str],
    doc_split_points: list[int],
    full_text: str,
    tolerance_chars: int = 5) -> float | None:
    """
    Returns the fraction of gold-standard blocks that the predicted
    chunking did *not* cut in half.

    A gold block is intact if no predicted split lies strictly inside
    the block (allowing `tolerance_chars` leeway at both ends).
    """
    # sanity checks
    if not chunks:
        return None
    if len(chunks) == 1:
        return 1.0

    # locate every chunk in the document
    starts_and_ends = find_chunks_start_and_end(chunks=chunks, text=full_text)
    starts = [start for start, end in starts_and_ends]

    # remove duplicates
    predicted_split_points = sorted({s for s in starts[1:] if s is not None})

    # gold-standard block boundaries
    gold_sorted = sorted(doc_split_points)
    doc_len = len(full_text)
    block_bounds = [0] + gold_sorted + [doc_len]

    # count intact gold blocks
    intact = 0
    for left, right in zip(block_bounds, block_bounds[1:]):
        block_broken = any(
            (left < p < right)                       # inside the block
            and (p - left) > tolerance_chars         # not too close to left edge
            and (right - p) > tolerance_chars        # not too close to right edge
            for p in predicted_split_points
        )
        if not block_broken:
            intact += 1

    total = len(block_bounds) - 1
    return intact / total if total else None

def compute_semantic_dissimilarity(
    chunks: list[str],
    model: "SentenceTransformer",
    batch_size: int = 16,
    progress_bar: bool = False,
    window_size: int = 5,
    min_tokens: int = 100,
    small_chunk_penalty_coef: float = 1.0,
    ) -> float | None:
    """
    Sliding-window semantic dissimilarity with an extra penalty for
    over-fragmentation (too many chunks shorter than min_tokens).
    """

    from sentence_transformers import util as st_util

    n = len(chunks)
    if n < 2:
        return None

    embeddings = model.encode(
        chunks,
        batch_size=batch_size,
        show_progress_bar=progress_bar,
        convert_to_tensor=True,
    )

    chunk_lengths = [count_tokens(c) for c in chunks]

    total_weighted_similarity = 0.0
    total_weight = 0.0

    # Sliding-window neighbourhood comparison
    for i in range(n):
        upper = min(i + window_size + 1, n)
        for j in range(i + 1, upper):
            similarity = st_util.cos_sim(embeddings[i], embeddings[j]).item()

            len1, len2 = chunk_lengths[i], chunk_lengths[j]
            weight = len1 * len2
            total_weighted_similarity += similarity * weight
            total_weight += weight

    if total_weight == 0:
        return None

    weighted_avg_similarity = total_weighted_similarity / total_weight
    dissimilarity = 1.0 - weighted_avg_similarity  # base semantic score

    # Additional penalty for too small chunks using fraction of small chunks
    undersized_chunks = sum(1 for l in chunk_lengths if l < min_tokens)
    if undersized_chunks:
        fraction_small = undersized_chunks / n
        dissimilarity -= small_chunk_penalty_coef * fraction_small

    # Clip to [0, 1]
    return np.clip(dissimilarity, 0.0, 1.0)

def compute_lexical_dissimilarity(
    chunks: list[str],
    window_size: int = 5,
    min_tokens: int = 100,
    small_chunk_penalty_coef: float = 1.0,
    ) -> None | float:
    """
    Sliding-window lexical (TF-IDF) dissimilarity with an extra penalty
    for over-fragmentation (too many chunks shorter than `min_tokens`).
    Returns a score in [0, 1] where higher = more dissimilar.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    n = len(chunks)
    if n < 2:
        return None

    # TF-IDF vectors for each chunk
    tfidf = TfidfVectorizer(
        stop_words="english",
        max_df=0.95
    ).fit_transform(chunks)

    chunk_lengths = [count_tokens(c) for c in chunks]

    total_sim, total_wt = 0.0, 0.0
    for i in range(n):
        upper = min(i + window_size + 1, n)
        for j in range(i + 1, upper):
            sim = cosine_similarity(tfidf[i], tfidf[j])[0, 0]
            wt  = chunk_lengths[i] * chunk_lengths[j]
            total_sim += sim * wt
            total_wt  += wt

    if total_wt == 0:
        return None

    dissim = 1.0 - (total_sim / total_wt)        # base lexical score

    # penalty for excessive small chunks
    undersized = sum(1 for L in chunk_lengths if L < min_tokens)
    if undersized:
        frac_small = undersized / n
        dissim -= small_chunk_penalty_coef * frac_small

    return np.clip(dissim, 0.0, 1.0)

def compute_normalized_intrachunk_sim(chunk_sentences: list[list[str]], model, batch_size=16, progress_bar=False) -> float:
    """
    Normalized intrachunk cosine similarity, computed as:
    score  = max(intrachunk_similarity - doc_similarity, 0) / ( 1 - doc_similarity )
    """
    sim_scores_list = []
    if not chunk_sentences:
        return 0.0

    processed_flattened_sentences = [
        sentence_text
        for sentences_in_single_chunk in chunk_sentences
        if sentences_in_single_chunk
        for sentence_text in sentences_in_single_chunk
    ]

    if not processed_flattened_sentences:
        return 0.0

    all_sentence_embeddings = model.encode(
        processed_flattened_sentences,
        batch_size=batch_size,
        show_progress_bar=progress_bar,
        normalize_embeddings=True,
    )

    total_num_sentences_in_doc = len(all_sentence_embeddings)
    if total_num_sentences_in_doc < 2:
        return 0.0

    current_embedding_index = 0
    for sentences_in_single_chunk in chunk_sentences:
        num_sentences_in_chunk = len(sentences_in_single_chunk)
        if num_sentences_in_chunk < 2:
            current_embedding_index += num_sentences_in_chunk
            continue

        chunk_embeddings = all_sentence_embeddings[
            current_embedding_index : current_embedding_index + num_sentences_in_chunk
        ]

        sum_of_chunk_embeddings = chunk_embeddings.sum(axis=0)
        chunk_similarity_score = (np.dot(sum_of_chunk_embeddings, sum_of_chunk_embeddings) - num_sentences_in_chunk) / (num_sentences_in_chunk * (num_sentences_in_chunk - 1))
        
        sim_scores_list.append(chunk_similarity_score)
        current_embedding_index += num_sentences_in_chunk

    sim_scores = np.mean(sim_scores_list) if sim_scores_list else 0.0

    sum_of_all_doc_embeddings = all_sentence_embeddings.sum(axis=0)
    doc_overall_similarity_denominator = total_num_sentences_in_doc * (total_num_sentences_in_doc - 1)
    
    if doc_overall_similarity_denominator == 0:
        return max(0.0, sim_scores)

    doc_overall_similarity = (np.dot(sum_of_all_doc_embeddings, sum_of_all_doc_embeddings) - total_num_sentences_in_doc) / doc_overall_similarity_denominator

    denominator = 1.0 - doc_overall_similarity

    if denominator < 1e-9:
        return 0.0

    numerator = sim_scores - doc_overall_similarity
    
    normalized_score = numerator / denominator

    final_normalized_sim = max(0.0, normalized_score)

    return final_normalized_sim

def compute_missing_ref_error(chunks: list[str], mentions: list[list[tuple]]) -> float:
    """
    Computes the missing reference error (MRE) for a list of chunks.
    Assumes the original text is the simple concatenation of the chunks.
    Mentions are assumed to be clustered using Coreference Resolution and sorted.
    """
    # check if any mention is present
    if len(mentions) < 1:
        print("Warning: no mentions provided.")
        return None

    # compute chunk boundaries (as chunk start index) excluding the beginning of the text
    chunk_boundaries = []
    current_len = 0
    for i, chunk in enumerate(chunks):
        current_len += len(chunk)
        if current_len > 0 and i < len(chunks) - 1:
            chunk_boundaries.append(current_len)

    # compute the missing references
    missing_references = 0
    total_references = 0
    for cluster in mentions:
        total_references += len(cluster)
        for i in range(len(cluster)-1):
            start = cluster[i][0] # start of the first mention
            end = cluster[i+1][1] # end of the next mention
            for boundary in chunk_boundaries:
                if start < boundary <= end:
                    missing_references += 1
                    
    return missing_references / total_references

def compute_filtered_missing_ref_error(
    full_text: str,
    chunks: list[str],
    entity_pron_pairs: list[list[tuple]]) -> float | None:
    """Missing reference error for chunks (overlapping or not) based on chunk boundaries (start and end).

    A reference is considered *missing* when **any** chunk boundary
    (i.e. the start of a chunk other than the first) falls strictly between
    the entity mention and its pronoun.  Each entity-pronoun pair can be
    counted at most once, even if multiple boundaries lie in-between.
    """

    if len(entity_pron_pairs) < 1:
        print("Warning: no entity-pronoun pairs provided.")
        return None

    # Gather all chunk boundary positions (starts, ends)
    # We exclude the document start and final character 

    boundary_positions: set[int] = set()

    # locate every chunk once in order (works with overlaps)
    chunk_bounds: list[tuple[int, int]] = []
    search_pos = 0
    for idx, chunk in enumerate(chunks):
        try:
            start_idx = full_text.find(chunk, search_pos)
        except ValueError:
            raise ValueError("Chunk not found in full_text.")

        end_idx = start_idx + len(chunk)
        chunk_bounds.append((start_idx, end_idx))

        # Add boundaries except for very first start
        if idx > 0:
            boundary_positions.add(start_idx)

        # move cursor fwd to allow overlaps
        search_pos = start_idx + 1

    # Add all chunk end positions except the last chunk's end
    for i, (_, end_idx) in enumerate(chunk_bounds):
        if i < len(chunk_bounds) - 1:
            boundary_positions.add(end_idx)

    # Sort boundaries
    chunk_boundaries = sorted(boundary_positions)

    # Check each entity-pronoun pair against the boundaries
    missing_references = 0
    for entity_span, pronoun_span in entity_pron_pairs:
        # ensure chronological order
        if entity_span[0] > pronoun_span[0]:
            entity_span, pronoun_span = pronoun_span, entity_span

        span_start = entity_span[0]
        span_end   = pronoun_span[1]

        # Count **once** per pair if *any* boundary (start or end) splits them
        for boundary in chunk_boundaries:
            if span_start < boundary <= span_end:
                missing_references += 1
                break  # avoid double-counting the same pair

    total_references = len(entity_pron_pairs)
    return missing_references / total_references if total_references else None

def extract_entity_pronoun_pairs(text: str, clusters: list[list[tuple]], spacy_model) -> list[list[tuple]]:
    """
    Extracts entity-pronoun pairs, ensuring each pronoun is uniquely paired
    with its last found antecedent.
    """
    def is_pronoun(span):
        """Helper, returns True if *any* token in `span` is a pronoun."""
        if not span:
            return False
        for tok in span:
            if tok.pos_ == "PRON" or tok.text.lower() in PERSONAL_PRONOUNS:
                return True
        return False

    doc = spacy_model(text.lower())
    all_mentions = {} # Cache spans to avoid re-creating them

    for cluster in clusters:
        for begin, end in cluster:
            if (begin, end) not in all_mentions:
                span = doc.char_span(begin, end + 1, alignment_mode="expand")
                if span:
                    all_mentions[(begin, end)] = span

    # Use a dictionary to store pairs, with the pronoun span as the key.
    # This automatically handles duplicates, keeping only the last one encountered.
    pairs_dict = {}

    for cluster in clusters:
        # 1. Handle nested mentions
        sorted_mentions = sorted(cluster, key=lambda x: (x[0], x[0] - x[1]))
        
        deduplicated_mentions = []
        for i, (begin1, end1) in enumerate(sorted_mentions):
            is_nested = False
            for j, (begin2, end2) in enumerate(sorted_mentions):
                if i == j: continue
                if begin2 <= begin1 and end2 >= end1 and (end2 - begin2 > end1 - begin1):
                    is_nested = True
                    break
            if not is_nested:
                deduplicated_mentions.append((begin1, end1))
        
        if not deduplicated_mentions:
            continue

        entity_spans = []
        pronoun_spans = []

        # 2. Find pronoun and entity spans
        for begin, end in deduplicated_mentions:
            span = all_mentions.get((begin, end))
            if span:
                if is_pronoun(span):
                    for token in span:
                        if token.pos_ == "PRON" or token.text.lower() in PERSONAL_PRONOUNS:
                            pronoun_begin = token.idx
                            pronoun_end = token.idx + len(token.text) - 1
                            pronoun_spans.append((pronoun_begin, pronoun_end))
                else:
                    entity_spans.append((begin, end))

        if len(pronoun_spans) < 1 or len(entity_spans) < 1:
            continue
        
        pronoun_spans = sorted(list(set(pronoun_spans)))
        entity_spans = sorted(list(set(entity_spans)))

        # 3. Create pairs
        for pronoun_begin, pronoun_end in pronoun_spans:
            preceding_entities = sorted(
                [es for es in entity_spans if es[0] < pronoun_begin],
                key=lambda es: pronoun_begin - es[0]
            )

            if preceding_entities:
                entity_span = preceding_entities[0]
            else:
                entity_span = entity_spans[0]

            # Assign the pair to the dictionary. If the pronoun key already exists,
            # its value (the pair) will be updated.
            pronoun_tuple = (pronoun_begin, pronoun_end)
            pairs_dict[pronoun_tuple] = [entity_span, pronoun_tuple]

    # Return the unique pairs from the dictionary values
    return list(pairs_dict.values())

class CoreferenceSolver:
    def __init__(self,
                 pre_splitter,
                 device: str = "cpu",
                 model_name: str = "sapienzanlp/maverick-mes-ontonotes",
                 tokenizer_model_name: str = "microsoft/deberta-large",
                 max_context_tokens: int = 2000,
                 lines_overlap: int = 2
                 ):
        
        try:
            from maverick import Maverick
        except ImportError:
            raise ImportError(
                "maverick-coref is required for coreference resolution. "
                "Install it with: pip install adaptive-chunking[coref]"
            )
        from transformers import AutoTokenizer

        self.device = device
        self.model = Maverick(hf_name_or_path = model_name, device = device)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_model_name)
        self.count_tokens_func = lambda x: len(self.tokenizer(x)["input_ids"])
        self.max_context_tokens = max_context_tokens
        self.lines_overlap = lines_overlap
        self.pre_splitter = pre_splitter

    def find_mentions(self, text: str) -> list[list[tuple]]:
        # create context windows using contiguous blocks and saving the starting offsets
        print("Splitting text into blocks...")
        blocks = self.pre_splitter.split_text(text)
        block_start_offsets = []
        char_pos = 0
        for block in blocks:
            block_start_offsets.append(char_pos)
            char_pos += len(block)
        
        print("Merging blocks into context windows...")
        context_windows, starting_offsets = self._group_chunks_with_offsets(blocks, block_start_offsets)

        if len(starting_offsets) > len(context_windows): # remove the last offset if it doesn't correspond to a window
            starting_offsets.pop()
        
        # find mentions in each context window
        mentions = []
        print("Finding mentions using maverick...")
        for starting_offset, window in zip(starting_offsets, context_windows):
            window_mentions = self._find_mention_clusters(window, text_index_offset=starting_offset)
            mentions.extend(window_mentions)
        
        # merge clusters that contain a common mention
        print("Merging mention clusters...")
        merged_mentions = self._merge_mention_clusters(mentions)
            
        # filter mentions to remove useless cases
        print("Filtering mention clusters...")
        final_mentions = self._filter_mentions(merged_mentions, text)

        return final_mentions
        
    def _group_chunks_with_offsets(self, blocks: list[str], block_start_offsets: list[int]) -> tuple[list[str], list[int]]:
        """
        Groups text blocks (e.g., lines) into overlapping chunks and tracks offsets.

        This method takes pre-split text blocks and their corresponding starting
        character offsets. It groups these blocks into chunks that do not exceed
        `max_context_tokens`, managing an overlap of a specified number of blocks
        between consecutive chunks.

        Args:
            blocks (list[str]): A list of text blocks to be grouped.
            block_start_offsets (list[int]): A list of starting character offsets for each block.

        Returns:
            A tuple containing two lists:
            - A list of strings, where each string is a text chunk.
            - A list of integers, where each integer is the starting character
            offset of the corresponding chunk in the original text.
        """
        # --- Helper function to crop a single block if it's too long ---
        def _crop_block_to_max_tokens(block: str) -> str:
            if not block: return ""
            # Use the class's tokenizer
            tokens = self.tokenizer.tokenize(block)
            if len(tokens) > self.max_context_tokens:
                cropped_tokens = tokens[:self.max_context_tokens]
                return self.tokenizer.convert_tokens_to_string(cropped_tokens)
            return block
        
        # 1. Pre-calculate token counts for each block.
        block_token_counts = [self.count_tokens_func(block) for block in blocks]

        # 2. Group blocks into chunks
        chunks = []
        chunk_start_offsets = []
        
        start_block_index = 0
        while start_block_index < len(blocks):
            # Start building a new chunk from the start_block_index
            current_chunk_blocks = []
            current_chunk_token_count = 0
            # NOTE: The 'chunk_build_start_index' variable was redundant, 
            # as it was always equal to 'start_block_index' at the point of use.
            # It has been removed for clarity.
            
            for i in range(start_block_index, len(blocks)):
                block = blocks[i]
                block_token_count = block_token_counts[i]
                
                # Handle case where a single block is too long
                if block_token_count > self.max_context_tokens:
                    # If there's a pending chunk, save it first
                    if current_chunk_blocks:
                        chunks.append("".join(current_chunk_blocks))
                        chunk_start_offsets.append(block_start_offsets[start_block_index])

                    # Add the cropped oversized block as its own chunk
                    cropped_block = _crop_block_to_max_tokens(block)
                    chunks.append(cropped_block)
                    chunk_start_offsets.append(block_start_offsets[i])
                    
                    # Reset and continue from the next block
                    start_block_index = i + 1
                    break
                
                # If adding the next block fits, add it
                if current_chunk_token_count + block_token_count <= self.max_context_tokens:
                    current_chunk_blocks.append(block)
                    current_chunk_token_count += block_token_count
                else:
                    # The chunk is full, so finalize it
                    chunks.append("".join(current_chunk_blocks))
                    chunk_start_offsets.append(block_start_offsets[start_block_index])
                    
                    # --- START: INFINITE LOOP FIX ---
                    # Determine the number of blocks in the chunk we just made.
                    num_blocks_in_chunk = len(current_chunk_blocks)

                    # Calculate the desired start of the next chunk based on overlap.
                    # We subtract the overlap from the number of blocks in the current chunk
                    # and add it to the current start index.
                    rewind_offset = max(0, num_blocks_in_chunk - self.lines_overlap)
                    next_start_candidate = start_block_index + rewind_offset

                    # CRITICAL: The new start_block_index MUST be greater than the current
                    # one to prevent an infinite loop. If the overlap logic would cause
                    # a stall, we force an advancement of at least one block.
                    start_block_index = max(next_start_candidate, start_block_index + 1)
                    # --- END: INFINITE LOOP FIX ---
                    
                    break
            else:
                # This else block runs if the inner loop completed without a `break`
                # (i.e., we've reached the end of the text)
                if current_chunk_blocks:
                    chunks.append("".join(current_chunk_blocks))
                    chunk_start_offsets.append(block_start_offsets[start_block_index])
                break # Exit the main while loop

        return chunks, chunk_start_offsets

    def _filter_mentions(self, mentions: list[list[tuple]], original_text: str) -> list[list[tuple]]:
        """
        Remove clusters having only the exact same mention text multiple times.
        Remove mentions composed of numerical only values, e.g., "12345 7890".
        """
        filtered_mentions = []
        for cluster in mentions:
            clean_mention_texts = [re.sub(r"\s+", "", original_text[start:end+1]) for start, end in cluster] #ignore newline, tabs, whitespaces

            # if all mentions are the same, skip the cluster
            if len(set(clean_mention_texts)) == 1:
                continue

            # skip mentions composed of numerical values only
            keep = []
            for start, end in cluster:
                clean_mention_text = re.sub(r"\s+", "", original_text[start:end+1]) #ignore newline, tabs, whitespaces
                if not clean_mention_text.isdigit():
                    keep.append((start, end))
            if keep:
                filtered_mentions.append(keep)
        return filtered_mentions

    def _merge_mention_clusters(self, mention_clusters: list[list[tuple]]) -> list[list[tuple]]:
        """
        Merge clusters that have at least one mention in common.
        """
        if not mention_clusters:
            return []

        current_merged_sets = [set(cluster) for cluster in mention_clusters]

        while True: # loop until no fusions are performed
            merged_something_in_this_pass = False
            i = 0
            while i < len(current_merged_sets):
                j = i + 1
                while j < len(current_merged_sets):
                    if current_merged_sets[i] & current_merged_sets[j]:
                        current_merged_sets[i].update(current_merged_sets[j])
                        current_merged_sets.pop(j)
                        merged_something_in_this_pass = True
                    else: # only increment j if no fusion was performed, since we're popping one element
                        j += 1
                i += 1

            if not merged_something_in_this_pass:
                break

        merged_clusters = [list(sorted(list(s))) for s in current_merged_sets]
        return merged_clusters

    def _find_mention_clusters(self, text: str, text_index_offset = 0) -> list[list[tuple]]:
        tokens, clean_tokens = self._tokenize_by_word(text, return_clean=True)
        model_output = self.model.predict(clean_tokens)
        token_offsets = model_output['clusters_token_offsets']

        # convert token offsets to character offsets
        char_offsets = []

        # Create a mapping from token indices to character indices
        char_index = 0
        token_to_char_map = []

        for token in tokens:
            token_to_char_map.append(char_index)
            char_index += len(token)

        # Convert each cluster's token offsets to character offsets
        for cluster in token_offsets:
            char_cluster = []
            for mention in cluster:
                start_token, end_token = mention
                # Get the character start position from the start token
                char_start = token_to_char_map[start_token]
                # Get the character end position from the end token
                if end_token < len(tokens):
                    char_end = token_to_char_map[end_token] + len(tokens[end_token]) - 1
                else:
                    # If the end token is out of bounds, use the end of the text
                    char_end = char_index - 1
                # Adjust for the text index offset
                char_cluster.append((char_start + text_index_offset, char_end + text_index_offset))
            char_offsets.append(char_cluster)

        return char_offsets

    def _tokenize_by_word(self, text: str, return_clean = False) -> list|tuple:
        import spacy
        # use the small English model for faster processing
        nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
        doc = nlp(text)
        
        tokens_with_ws = []
        clean_tokens = [] if return_clean else None
        
        for token in doc:
            tokens_with_ws.append(token.text_with_ws)
            if return_clean:
                clean_tokens.append(token.text)
        
        return (tokens_with_ws, clean_tokens) if return_clean else tokens_with_ws