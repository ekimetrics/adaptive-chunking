import tiktoken
import spacy
from typing import Callable, List, Literal, Awaitable, Any, Tuple
import re
import stanza
import nltk
from nltk.tokenize import sent_tokenize
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import numpy as np
from matplotlib import pyplot as plt
import gc
from functools import lru_cache
from ..chunking_utils import count_tokens
import asyncio
from langchain_experimental.text_splitter import SemanticChunker

from ..splitters import combine_blocks, regex_splitter


class SemanticChunkerWrapper(SemanticChunker):
    def __init__(self, *args, lookback_min: int = 32, **kwargs):
        """
        A subclass of SemanticChunker that keeps the base semantic breakpoints but remaps each chunk back to the original text using a whitespace-tolerant match, then makes chunks contiguous.
        """
        super().__init__(*args, **kwargs)
        self._lookback_min = lookback_min

    @staticmethod
    def _relaxed_pattern(s: str) -> re.Pattern:
        s = re.sub(r"\s+", " ", s.strip())
        parts = [re.escape(p) for p in s.split(" ")] if s else []
        return re.compile(r"\s+".join(parts), re.DOTALL | re.MULTILINE)

    def _map_chunks(self, chunks: List[str], text: str) -> List[Tuple[int,int]]:
        spans, cur = [], 0
        for ch in chunks:
            if not ch.strip():
                spans.append((cur, cur)); continue
            pat = self._relaxed_pattern(ch)
            m = pat.search(text, cur) or pat.search(text, max(0, cur - max(self._lookback_min, len(ch)//4)))
            if not m:
                raise ValueError("Could not remap a chunk back to the source text.")
            s, e = m.span(); spans.append((s, e)); cur = e
        return spans

    def split_text(self, text: str) -> List[str]:
        mutated = super().split_text(text)
        if not mutated:
            return []
        spans = self._map_chunks(mutated, text)
        boundaries = [0] + [s for s, _ in spans[1:]] + [len(text)]
        for i in range(1, len(boundaries)):
            if boundaries[i] < boundaries[i-1]:
                boundaries[i] = boundaries[i-1]
        return [text[boundaries[i]:boundaries[i+1]] for i in range(len(mutated))]


class SentenceSplitter:
    """
    Split text into sentences using NLTK, Stanza, or SpaCy.
    """

    def __init__(self, method = "nltk", sentences_per_chunk = 1, device = "cpu"):
        if method == "nltk":
            # Download NLTK punkt tokenizer if not already downloaded
            try:
                nltk.download('punkt', quiet=True)
                nltk.download('punkt_tab', quiet=True)
            except:
                pass
            self.splitter = self._split_into_sentences_nltk
        elif method == "stanza":
            self.splitter = self._split_into_sentences_stanza
        elif method == "spacy-small":
            self.splitter = lambda text: self._split_into_sentences_spacy(text, modeltype="small")
        elif method == "spacy-medium":
            self.splitter = lambda text: self._split_into_sentences_spacy(text, modeltype="medium")
        elif method == "spacy-large":
            self.splitter = lambda text: self._split_into_sentences_spacy(text, modeltype="large")
        elif method == "lines":
            self.splitter = lambda text: self._split_by_lines(text)
        elif method == "blank_lines":
            self.splitter = lambda text: self._split_by_blank_lines(text)
        else:
            raise ValueError("Invalid method. Choose 'nltk', 'stanza', 'spacy-small', 'spacy-medium', or 'spacy-large'.")

        self.sentences_per_chunk = sentences_per_chunk
        self.device = device

    def _split_into_sentences_nltk(self, text: str) -> list[str]:
        """
        Splits text into sentences using NLTK's sent_tokenize while preserving all text
        including line separators between sentence boundaries.
        """

        # Get the raw sentences
        sentences = sent_tokenize(text)
        result = []

        # If no sentences were found, return the original text
        if not sentences:
            return [text]

        current_pos = 0

        for i, sentence in enumerate(sentences):
            # Find the start position of this sentence
            start_pos = text.find(sentence, current_pos)

            # If we couldn't find the sentence, try a more flexible search
            if start_pos == -1:
                start_pos = current_pos

            # For all sentences except the last one
            if i < len(sentences) - 1:
                # Find the start position of the next sentence
                next_sentence = sentences[i + 1]
                next_start = text.find(next_sentence, start_pos + len(sentence))

                if next_start == -1:
                    # If we can't find the next sentence, just use this one
                    segment = text[start_pos:]
                    current_pos = len(text)
                else:
                    # Include everything up to the start of the next sentence
                    segment = text[start_pos:next_start]
                    current_pos = next_start
            else:
                # For the last sentence, include everything until the end
                segment = text[start_pos:]

            result.append(segment)

        return result

    def _split_into_sentences_stanza(self, text: str) -> list[str]:
        try:
            nlp = stanza.Pipeline('en', processors='tokenize', tokenize_no_ssplit=False, verbose=False, device=self.device)
        except Exception as e:
            print("Stanza model not found. Downloading...")
            stanza.download('en')
            nlp = stanza.Pipeline('en', processors='tokenize', tokenize_no_ssplit=False, verbose=False, device=self.device)

        doc = nlp(text)
        sentences = doc.sentences
        result = []

        for i, sent in enumerate(sentences):
            # For every sentence except the last, extend to include separator
            if i < len(sentences) - 1:
                next_sent = sentences[i+1]
                sentence_with_sep = text[sent.tokens[0].start_char:next_sent.tokens[0].start_char]
            else:
                # For the last sentence, take until the end of text
                sentence_with_sep = text[sent.tokens[0].start_char:]

            result.append(sentence_with_sep)

        return result

    def _split_into_sentences_spacy(self, text: str, modeltype: str) -> list[str]:
        if modeltype == "small":
            nlp = spacy.load("en_core_web_sm")
        elif modeltype == "medium":
            nlp = spacy.load("en_core_web_md")
        elif modeltype == "large":
            nlp = spacy.load("en_core_web_lg")
        else:
            raise ValueError("Invalid model type. Choose 'small', 'medium', or 'large'.")

        if self.device.startswith("cuda"):
            device = self.device.split(":")[-1]
            spacy.prefer_gpu(device)

        doc = nlp(text)
        sentences = list(doc.sents)
        result = []

        for i, sent in enumerate(sentences):
            # For every sentence except the last, extend the sentence to include the separator
            if i < len(sentences) - 1:
                next_sent = sentences[i+1]
                sentence_with_sep = text[sent.start_char:next_sent.start_char]
            else:
                # For the last sentence, take until the end of the text.
                sentence_with_sep = text[sent.start_char:]
            result.append(sentence_with_sep)

        return result

    def _split_by_lines(self, text: str) -> list[str]:
        lines = text.splitlines()
        result = []

        current_blanklines = 0
        for i, line in enumerate(lines):
            if line == "":
                current_blanklines += 1
            else:
                if result:
                    result[-1] += '\n' * current_blanklines
                    current_blanklines = 0
                    result.append(line + '\n')
                else:
                    result.append(line + '\n')
                    result[-1] = '\n' * current_blanklines + result[-1]

        if current_blanklines:
            result[-1] += '\n' * current_blanklines

        return result

    def _split_by_blank_lines(self, text: str) -> list[str]:
        chunks = regex_splitter(text, r"\n\n")
        return chunks

    def split_text(self, text: str) -> list[str]:

        sentences = self.splitter(text)

        # If sentences_per_chunk is greater than 1, group sentences into chunks
        if self.sentences_per_chunk > 1:
            chunks = []
            for i in range(0, len(sentences), self.sentences_per_chunk):
                chunk = "".join(sentences[i:i + self.sentences_per_chunk])
                chunks.append(chunk)
            return chunks
        return sentences


class LongContextSemanticSplitter:
    """
    Experimental splitter.
    Split text into semantic chunks computing cosine similarity between consecutive sentences.
    The raw text is split into sentences using a sentence splitter, default is NLTK which is fast.
    The sentences are grouped into overlapping blocks each having at most max_tokens tokens.
    All tokens in a block are embedded together using a long context embedding model in order to increase contextuality.
    Default embedding model: Qwen/Qwen3-Embedding-0.6B with max 32k context tokens.

    """
    def __init__(self,
                 sentence_splitter: Callable[[str], list[str]] = None,
                 threshold: float|str = "max_tokens",
                 max_context_tokens: int = 8000,
                 max_chunk_tokens: int = 1200,
                 quantile_value: float = 0.90,
                 sentence_overlap: int = 5,
                 model_name: str = "Qwen/Qwen3-Embedding-0.6B",
                 device: str = "cpu",
                 batch_size: int = 2,
                 visualize_splitting: bool = False):

        self.device = torch.device(device)
        self.batch_size = batch_size
        self.tokenizer = self._get_tokenizer(model_name)
        self.tokenizer_func = lambda x: self.tokenizer(x)["input_ids"]
        self.model = self._get_model(model_name, self.device)
        self.max_context_tokens = max_context_tokens

        if sentence_splitter is None:
            sent_splitter_obj = SentenceSplitter(method="nltk")
            self.sentence_splitter = sent_splitter_obj.split_text
        else:
            self.sentence_splitter = sentence_splitter

        self.threshold = threshold
        self.quantile_value = quantile_value
        self.max_chunk_tokens = max_chunk_tokens
        self.sentence_overlap = sentence_overlap
        self.visualize_splitting = visualize_splitting

    @staticmethod
    @lru_cache(maxsize=None)
    def _get_tokenizer(model_name: str):
        """Load once per process, then get from cache."""
        return AutoTokenizer.from_pretrained(model_name)

    @staticmethod
    @lru_cache(maxsize=None)
    def _get_model(model_name: str, device: torch.device):
        """Load model once, put it on the requested device, keep it forever."""
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        return model.to(device).eval()

    @torch.inference_mode()
    def split_text(self, text: str) -> list[str]:
        # Use the provided non-destructive sentence splitter. This is the source of truth.
        sentences = self.sentence_splitter(text)
        assert "".join(sentences) == text, "The sentence splitter is destructive, please use a non-destructive one."

        if not sentences:
            return [text] if text else []

        # Use group_chunks for semantic analysis, but be aware its output can be cropped.
        grouped_sentences_text, grouped_sentences = self._group_chunks(
            blocks=sentences,
            max_tokens=self.max_context_tokens,
            tokenizer_func=self.tokenizer_func,
            chunk_block_overlap=self.sentence_overlap,
            verbose=True
        )

        reconstructed = ""
        for i, block in enumerate(grouped_sentences):        # grouped_sentences is a list[list[str]]
            if i == 0:                                       # keep the whole first chunk
                reconstructed += "".join(block)
            else:                                            # skip the first N (= overlap) blocks
                reconstructed += "".join(block[self.sentence_overlap:])
        assert reconstructed == text, "grouped sentences reconstruction failed"

        reconstructed = ""
        for i, (chunk_text, chunk_blocks) in enumerate(zip(grouped_sentences_text, grouped_sentences)):
            if i == 0 or self.sentence_overlap == 0:         # first chunk (or no overlap): take all
                reconstructed += chunk_text
            else:
                # number of characters to cut from the front of this chunk
                chars_to_skip = sum(len(b) for b in chunk_blocks[:self.sentence_overlap])
                reconstructed += chunk_text[chars_to_skip:]
        assert reconstructed == text, "grouped sentences text reconstruction failed"

        if not grouped_sentences:
            return ["".join(sentences)]

        absolute_split_indices = set()

        # Keep track of where the last group started in the original sentences list
        last_group_start_index = 0

        for i, group in enumerate(grouped_sentences):
            # Perform semantic analysis on the potentially cropped text from group_chunks
            batch_inputs = self.tokenizer(grouped_sentences_text[i], return_tensors="pt", padding=True).to(self.device)
            batch_embeds = self.model(**batch_inputs)["last_hidden_state"].to("cpu")
            batch_input_ids = batch_inputs["input_ids"].to("cpu")

            token_embeds = batch_embeds[0]
            input_ids = batch_input_ids[0]

            special_ids = torch.tensor(self.tokenizer.all_special_ids, device=input_ids.device)
            mask = ~torch.isin(input_ids, special_ids)
            token_embeds = token_embeds[mask]

            sentence_vectors, n_tokens_per_sentence = [], []
            begin = 0
            for sentence in group:
                n_tokens = sum(
                    1
                    for t in self.tokenizer(sentence)["input_ids"]
                    if t not in self.tokenizer.all_special_ids
                )
                n_tokens_per_sentence.append(n_tokens)
                end = begin + n_tokens
                sentence_vectors.append(token_embeds[begin:end].mean(dim=0, keepdim=True))
                begin = end

            dissimilarities = [
                torch.clamp(1 - F.cosine_similarity(sentence_vectors[j], sentence_vectors[j - 1]), -1, 1).item()
                for j in range(1, len(sentence_vectors))
            ]

            # Determine threshold
            if self.threshold == "quantile":
                current_threshold = np.quantile(dissimilarities, self.quantile_value) if len(dissimilarities) > 1 else 0.05
            elif self.threshold == "max_tokens":
                current_threshold = 0.0
                if dissimilarities:
                    for cand in sorted(set(dissimilarities), reverse=True) + [0.0]:
                        split_points_cand = [p for p, d in enumerate(dissimilarities) if d > cand]

                        max_len = 0
                        last_p = -1
                        for p in split_points_cand + [len(group) - 1]:
                            chunk_tokens = sum(n_tokens_per_sentence[last_p+1:p+1])
                            if chunk_tokens > max_len:
                                max_len = chunk_tokens
                            last_p = p

                        if max_len <= self.max_chunk_tokens:
                            current_threshold = cand
                            break
            else:
                current_threshold = self.threshold

            if self.visualize_splitting:
                self._plot_splitting(dissimilarities, current_threshold, f"group {i+1}/{len(grouped_sentences)}")

            # Find split points relative to the current group
            relative_split_points = [p for p, d in enumerate(dissimilarities) if d > current_threshold]

            # Find the starting index of the current group in the original sentences list
            # This is a robust way to map relative indices to absolute ones
            group_start_index = -1
            if group:
                try:
                    group_start_index = sentences.index(group[0], last_group_start_index)
                except ValueError:
                    # This fallback is unlikely but safe, in case a sentence appears multiple times
                    group_start_index = -1

            if group_start_index != -1:
                # Map relative split points to absolute indices in the original sentences list
                for p in relative_split_points:
                    absolute_idx = group_start_index + p
                    absolute_split_indices.add(absolute_idx)
                last_group_start_index = group_start_index


        # Build final chunks from original sentences using absolute split indices
        splits = []
        last_split = -1
        for split_idx in sorted(list(absolute_split_indices)):
            # Ensure we don't create empty chunks if splits are consecutive
            if split_idx > last_split:
                chunk_sentences = sentences[last_split + 1 : split_idx + 1]
                splits.append("".join(chunk_sentences))
            last_split = split_idx

        # Add the final remaining chunk
        final_chunk_sentences = sentences[last_split + 1 :]
        if final_chunk_sentences:
            splits.append("".join(final_chunk_sentences))

        # If no splits were made, return the whole text as one chunk
        if not splits and sentences:
            return ["".join(sentences)]

        return splits

    def _group_chunks(
        self,
        blocks: list[str],
        tokenizer_func: Callable[[str], list[str]],
        max_tokens: int,
        chunk_block_overlap: int = 0,
        verbose: bool = False
        ) -> tuple[list[str], list[list[str]]]:
        """
        Group pre-chunked text blocks into overlapping chunks without cutting blocks.
        Returns:
            block_texts      : list[str]          – the joined text of each chunk
            grouped_blocks   : list[list[str]]    – the blocks that make up every chunk
        """

        token_counts = [len(tokenizer_func(b)) for b in blocks]

        block_texts = []
        grouped_blocks = []

        current = []
        cur_tokens = 0
        i = 0

        while i < len(blocks):
            block = blocks[i]
            block_tokens = token_counts[i]

            if block_tokens > max_tokens:
                if verbose:
                    print(f"Warning: block with {block_tokens} tokens exceeds max_tokens "
                        f"({max_tokens}); placed in its own chunk.")
                if current:
                    block_texts.append("".join(current))
                    grouped_blocks.append(current)
                    current, cur_tokens = [], 0
                block_texts.append(block)
                grouped_blocks.append([block])
                i += 1
                continue

            if cur_tokens + block_tokens <= max_tokens or not current:
                current.append(block)
                cur_tokens += block_tokens
                i += 1
                continue

            block_texts.append("".join(current))
            grouped_blocks.append(current)

            if chunk_block_overlap > 0:
                overlap_blocks = current[-chunk_block_overlap:]
            else:
                overlap_blocks = []

            current = overlap_blocks.copy()
            cur_tokens = sum(len(tokenizer_func(b)) for b in current)

        if current:
            block_texts.append("".join(current))
            grouped_blocks.append(current)

        return block_texts, grouped_blocks

    def _plot_splitting(self, dissimilarities: list[float], threshold: float, title_context: str = None):

        # plot dissimilarities and threshold
        plt.figure(figsize=(6, 3))
        plt.scatter(range(len(dissimilarities)), dissimilarities, color='b', s=10, zorder=1)
        above_threshold_indices = [i for i, similarity in enumerate(dissimilarities) if similarity > threshold]
        above_threshold_values = [dissimilarities[i] for i in above_threshold_indices]
        plt.scatter(above_threshold_indices, above_threshold_values, color='r', s=10, zorder=1, label='Splitting points')
        plt.axhline(y=threshold, color='g', linestyle='--', label=f'Threshold = {threshold:.3f}')
        plt.title('Dissimilarity between consecutive sentences' + " " + (title_context if title_context else ''))
        plt.xlabel('sentence index')
        plt.ylabel('1 - cosine similarity')
        plt.legend(fontsize='small')
        plt.grid()
        plt.show()

    def __del__(self):
        try:
            if hasattr(self, "model") and self.model is not None:
                self.model.to("cpu")

            for attr in ("model", "tokenizer", "tokenizer_func"):
                if hasattr(self, attr):
                    delattr(self, attr)

            try:
                LongContextSemanticSplitter._get_model.cache_clear()
                LongContextSemanticSplitter._get_tokenizer.cache_clear()
            except:
                pass

            gc.collect()
            with torch.no_grad():
                torch.cuda.empty_cache()
        except:
            pass


class LLMRegexSplitter():
    def __init__(self,
                 base_prompt: str,
                 async_client_completion_func: Awaitable[str],
                 count_tokens_func: Callable[[str], int] = count_tokens,
                 context_tokens: int = 8000,
                 ):

        self.base_prompt = base_prompt
        self.count_tokens_func = count_tokens_func
        self.async_client_completion_func = async_client_completion_func
        self.context_tokens = context_tokens

    async def split_text(self, text: str) -> list[str]:
        lines = text.splitlines(keepends=True)
        document_context = combine_blocks(lines, max_tokens=self.context_tokens, count_tokens_func=self.count_tokens_func)
        prompt = self.base_prompt + "<Input>" + document_context + "</Input>"
        response = await self.async_client_completion_func(prompt)
        regex_pattern = extract_llm_regex(response)
        if regex_pattern is None:
            return [text]
        splits = regex_splitter(text, regex_pattern, attach_to="start")
        return splits


def extract_llm_regex(answer: str) -> str:
    """
    Given an LLM answer containing a <regex>...</regex> block,
    returns the raw pattern string inside those tags and None if the pattern is not valid.
    """
    m = re.search(r"<regex>\s*([\s\S]+?)\s*</regex>", answer, re.IGNORECASE)
    if not m:
        return None
    pattern = m.group(1).strip()
    try:
        re.compile(pattern)
        return pattern
    except re.error:
        def _esc_hyphen(match: re.Match) -> str:
            return re.sub(r"(?<!\\)(?<=.)-(?=[^]])", r"\-", match.group(0))
        pattern = re.sub(r"\[[^\]]+\]", _esc_hyphen, pattern)
        try:
            re.compile(pattern)
            return pattern
        except re.error:
            return None
