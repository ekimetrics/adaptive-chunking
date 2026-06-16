"""Extra coverage tests for adaptive_chunking.metrics.

sentence-transformers, spacy and maverick are NOT installed, so anything that
depends on them is mocked (a fake embedding model, an injected fake
``sentence_transformers`` module, and ``CoreferenceSolver.__new__`` to bypass
the heavy ``__init__``). numpy and scikit-learn are real.

Functions intentionally left uncovered (need a real spaCy model / maverick):
``extract_entity_pronoun_pairs``, ``CoreferenceSolver.__init__``,
``CoreferenceSolver.find_mentions``, ``_tokenize_by_word``,
``_load_word_tokenizer``.
"""

import sys
import types

import numpy as np
import pytest

from adaptive_chunking.metrics import (
    CoreferenceSolver,
    compute_block_integrity,
    compute_chunk_embeddings,
    compute_contextual_coherence,
    compute_intrachunk_cohesion,
    compute_lexical_dissimilarity,
    compute_missing_ref_error,
    compute_normalized_intrachunk_sim,
    compute_semantic_dissimilarity,
)


class FakeModel:
    """Deterministic stand-in for a SentenceTransformer (no torch needed)."""

    def __init__(self, dim=4):
        self.dim = dim

    def encode(self, texts, batch_size=16, show_progress_bar=False,
               convert_to_numpy=True, normalize_embeddings=False,
               convert_to_tensor=False):
        arr = np.array(
            [[float((len(t) + k) % 7 + 1) for k in range(self.dim)] for t in texts],
            dtype=float,
        )
        if normalize_embeddings:
            n = np.linalg.norm(arr, axis=1, keepdims=True)
            n[n == 0] = 1.0
            arr = arr / n
        return arr


def _inject_fake_sentence_transformers():
    """Install a fake ``sentence_transformers`` with a working ``util.cos_sim``."""
    st = types.ModuleType("sentence_transformers")
    util = types.ModuleType("sentence_transformers.util")

    class _Sim:
        def __init__(self, v):
            self._v = float(v)

        def item(self):
            return self._v

    util.cos_sim = lambda a, b: _Sim(
        float(np.dot(np.asarray(a), np.asarray(b))
              / ((np.linalg.norm(a) * np.linalg.norm(b)) or 1))
    )
    st.util = util
    sys.modules["sentence_transformers"] = st
    sys.modules["sentence_transformers.util"] = util


# --------------------------------------------------------------------------- #
# compute_chunk_embeddings
# --------------------------------------------------------------------------- #
class TestComputeChunkEmbeddings:
    def test_shape(self):
        chunks = ["alpha", "beta beta", "gamma gamma gamma"]
        emb = compute_chunk_embeddings(chunks, model=FakeModel(dim=4))
        assert isinstance(emb, np.ndarray)
        assert emb.shape == (3, 4)


# --------------------------------------------------------------------------- #
# compute_block_integrity
# --------------------------------------------------------------------------- #
class TestBlockIntegrity:
    def test_empty(self):
        assert compute_block_integrity([], [], "") is None

    def test_single_chunk(self):
        assert compute_block_integrity(["only chunk"], [], "only chunk") == 1.0

    def test_block_broken(self):
        # one gold block (no split points) cut by the predicted split at 10
        chunks = ["aaaaaaaaaa", "bbbbbbbbbb"]
        full_text = "".join(chunks)
        assert compute_block_integrity(chunks, [], full_text) == 0.0

    def test_block_intact(self):
        # gold split aligns with the predicted split -> nothing broken
        chunks = ["aaaaaaaaaa", "bbbbbbbbbb"]
        full_text = "".join(chunks)
        assert compute_block_integrity(chunks, [10], full_text) == 1.0

    def test_fraction(self):
        # 3 chunks, predicted splits at 10 & 20; gold split at 20.
        # block (0,20) is broken by split 10; block (20,30) intact -> 0.5
        chunks = ["0000000000", "1111111111", "2222222222"]
        full_text = "".join(chunks)
        assert compute_block_integrity(chunks, [20], full_text) == 0.5


# --------------------------------------------------------------------------- #
# compute_missing_ref_error
# --------------------------------------------------------------------------- #
class TestMissingRefError:
    def test_no_mentions(self, capsys):
        assert compute_missing_ref_error(["a", "b"], []) is None
        assert "no mentions" in capsys.readouterr().out.lower()

    def test_cluster_split_by_boundary(self):
        # text = concatenation; boundaries at cumulative lengths [5, 10]
        chunks = ["aaaaa", "bbbbb", "ccccc"]
        # one cluster whose span (0..9) crosses boundary 5
        mentions = [[(0, 2), (7, 9)]]
        assert compute_missing_ref_error(chunks, mentions) == 0.5


# --------------------------------------------------------------------------- #
# compute_intrachunk_cohesion
# --------------------------------------------------------------------------- #
class TestIntrachunkCohesion:
    def test_computed_embeddings_path(self):
        chunks = ["Hello world.", "Foo bar baz."]
        full_text = "".join(chunks)
        # split at char 6 inside chunk0 -> chunk0 has 2 sentences
        score = compute_intrachunk_cohesion(
            chunks, full_text, [6], model=FakeModel()
        )
        assert score is not None
        assert 0.0 <= float(score) <= 1.0

    def test_provided_embeddings_path(self):
        chunks = ["Hello world.", "Foo bar baz."]
        full_text = "".join(chunks)
        model = FakeModel()
        emb = compute_chunk_embeddings(chunks, model=model)
        score = compute_intrachunk_cohesion(
            chunks, full_text, [6], model=model, chunk_embeddings=emb
        )
        assert 0.0 <= float(score) <= 1.0

    def test_mismatched_length_raises(self):
        chunks = ["Hello world.", "Foo bar baz."]
        full_text = "".join(chunks)
        bad = np.zeros((1, 4))
        with pytest.raises(ValueError):
            compute_intrachunk_cohesion(
                chunks, full_text, [6], model=FakeModel(), chunk_embeddings=bad
            )

    def test_all_single_sentence_returns_none(self):
        # no split points inside any chunk -> every chunk has 1 sentence -> None
        chunks = ["Hello world.", "Foo bar baz."]
        full_text = "".join(chunks)
        assert compute_intrachunk_cohesion(
            chunks, full_text, [], model=FakeModel()
        ) is None


# --------------------------------------------------------------------------- #
# compute_contextual_coherence
# --------------------------------------------------------------------------- #
class TestContextualCoherence:
    def test_n_less_than_two(self):
        assert compute_contextual_coherence(
            ["only one"], "only one", model=FakeModel(), count_tokens_func=len
        ) is None

    def test_normal_returns_float(self):
        chunks = ["abcdefghij", "klmnopqrst"]
        full_text = "".join(chunks)
        score = compute_contextual_coherence(
            chunks, full_text, model=FakeModel(), count_tokens_func=len
        )
        assert score is not None
        assert 0.0 <= float(score) <= 1.0

    def test_mismatched_length_raises(self):
        chunks = ["abcdefghij", "klmnopqrst"]
        full_text = "".join(chunks)
        with pytest.raises(ValueError):
            compute_contextual_coherence(
                chunks, full_text, model=FakeModel(),
                count_tokens_func=len, chunk_embeddings=np.zeros((1, 4)),
            )


# --------------------------------------------------------------------------- #
# compute_normalized_intrachunk_sim
# --------------------------------------------------------------------------- #
class TestNormalizedIntrachunkSim:
    def test_empty(self):
        assert compute_normalized_intrachunk_sim([], FakeModel()) == 0.0

    def test_single_sentence_total(self):
        # only one sentence in the whole document -> total < 2 -> 0.0
        assert compute_normalized_intrachunk_sim([["only one"]], FakeModel()) == 0.0

    def test_normal(self):
        chunk_sentences = [
            ["first sentence here", "second sentence here"],
            ["third one", "fourth one"],
        ]
        score = compute_normalized_intrachunk_sim(chunk_sentences, FakeModel())
        assert isinstance(float(score), float)
        assert score >= 0.0


# --------------------------------------------------------------------------- #
# compute_semantic_dissimilarity (fake sentence_transformers)
# --------------------------------------------------------------------------- #
class TestSemanticDissimilarity:
    def test_n_less_than_two(self):
        _inject_fake_sentence_transformers()
        assert compute_semantic_dissimilarity(["only one"], model=FakeModel()) is None

    def test_normal(self):
        _inject_fake_sentence_transformers()
        chunks = ["the quick brown fox", "jumps over the lazy dog", "more text here"]
        score = compute_semantic_dissimilarity(
            chunks, model=FakeModel(), min_tokens=1
        )
        assert score is not None
        assert 0.0 <= float(score) <= 1.0

    def test_small_chunk_penalty(self):
        _inject_fake_sentence_transformers()
        chunks = ["the quick brown fox", "jumps over the lazy dog"]
        # huge min_tokens => every chunk counts as undersized -> penalty branch
        score = compute_semantic_dissimilarity(
            chunks, model=FakeModel(), min_tokens=10_000
        )
        assert 0.0 <= float(score) <= 1.0


# --------------------------------------------------------------------------- #
# compute_lexical_dissimilarity (real sklearn)
# --------------------------------------------------------------------------- #
class TestLexicalDissimilarity:
    def test_n_less_than_two(self):
        assert compute_lexical_dissimilarity(["only one"]) is None

    def test_normal(self):
        chunks = [
            "apples bananas cherries fruit basket",
            "engines pistons gears torque machinery",
            "violins cellos trumpets orchestra music",
        ]
        score = compute_lexical_dissimilarity(chunks, min_tokens=1)
        assert score is not None
        assert 0.0 <= float(score) <= 1.0

    def test_small_chunk_penalty(self):
        chunks = [
            "apples bananas cherries fruit basket",
            "engines pistons gears torque machinery",
        ]
        score = compute_lexical_dissimilarity(chunks, min_tokens=10_000)
        assert 0.0 <= float(score) <= 1.0


# --------------------------------------------------------------------------- #
# CoreferenceSolver pure methods
# --------------------------------------------------------------------------- #
def _bare_solver():
    return CoreferenceSolver.__new__(CoreferenceSolver)


class TestMergeMentionClusters:
    def test_empty(self):
        assert _bare_solver()._merge_mention_clusters([]) == []

    def test_shared_mention_merged(self):
        solver = _bare_solver()
        merged = solver._merge_mention_clusters(
            [[(0, 1), (2, 3)], [(2, 3), (4, 5)]]
        )
        assert merged == [[(0, 1), (2, 3), (4, 5)]]

    def test_disjoint_stay_separate(self):
        solver = _bare_solver()
        merged = solver._merge_mention_clusters([[(0, 1)], [(2, 3)]])
        assert sorted(merged) == [[(0, 1)], [(2, 3)]]


class TestFilterMentions:
    def test_drops_identical_and_numeric(self):
        solver = _bare_solver()
        text = "aaa aaa bbb 999"
        clusters = [
            [(0, 2), (4, 6)],   # both "aaa" -> dropped (all identical)
            [(8, 10), (12, 14)],  # "bbb" + "999" -> numeric removed, keep "bbb"
        ]
        assert solver._filter_mentions(clusters, text) == [[(8, 10)]]

    def test_all_numeric_cluster_dropped(self):
        solver = _bare_solver()
        text = "999 888"
        clusters = [[(0, 2), (4, 6)]]  # "999","888": distinct but both digits
        assert solver._filter_mentions(clusters, text) == []


class TestFindMentionClusters:
    def test_token_to_char_conversion(self):
        solver = _bare_solver()

        class FakeMaverick:
            def predict(self, tokens):
                return {"clusters_token_offsets": [[(0, 0), (2, 2)]]}

        solver.model = FakeMaverick()
        tokens = ["Alice ", "and ", "Bob"]
        clean = ["Alice", "and", "Bob"]
        solver._tokenize_by_word = lambda text, return_clean=False: (
            (tokens, clean) if return_clean else tokens
        )

        result = solver._find_mention_clusters("ignored", text_index_offset=100)
        # char map: "Alice "->0, "and "->6, "Bob"->10
        # mention (0,0): start 0, end 0+6-1=5 ; mention (2,2): start 10, end 10+3-1=12
        assert result == [[(100, 105), (110, 112)]]


class TestGroupChunksWithOffsets:
    def _solver(self, max_context_tokens, lines_overlap):
        solver = _bare_solver()

        class FakeTok:
            def tokenize(self, s):
                return s.split()

            def convert_tokens_to_string(self, toks):
                return " ".join(toks)

        solver.tokenizer = FakeTok()
        solver.count_tokens_func = lambda s: len(s.split())
        solver.max_context_tokens = max_context_tokens
        solver.lines_overlap = lines_overlap
        return solver

    def test_small_blocks_grouped(self):
        solver = self._solver(max_context_tokens=10, lines_overlap=1)
        blocks = ["a a", "b b", "c c"]
        offsets = [0, 3, 6]
        chunks, starts = solver._group_chunks_with_offsets(blocks, offsets)
        assert chunks == ["a ab bc c"]
        assert starts == [0]

    def test_oversized_block_cropped(self):
        solver = self._solver(max_context_tokens=3, lines_overlap=1)
        blocks = ["x x x x x"]  # 5 tokens > 3 -> cropped to 3
        offsets = [0]
        chunks, starts = solver._group_chunks_with_offsets(blocks, offsets)
        assert chunks == ["x x x"]
        assert starts == [0]

    def test_overlap_and_advance(self):
        solver = self._solver(max_context_tokens=4, lines_overlap=1)
        blocks = ["a a", "b b", "c c", "d d"]  # 2 tokens each
        offsets = [0, 4, 8, 12]
        chunks, starts = solver._group_chunks_with_offsets(blocks, offsets)
        assert chunks == ["a ab b", "b bc c", "c cd d"]
        assert starts == [0, 4, 8]
