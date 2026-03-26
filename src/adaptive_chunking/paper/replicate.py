"""Replicate the LREC 2026 paper results from parsed CLAIR documents.

Usage:
    python -m adaptive_chunking.paper.replicate --data-dir data/clair/ --output-dir results/

Expected data directory layout:
    data/clair/
    ├── adi_parsed/       # 33 parsed JSON documents
    └── mentions/         # Pre-computed coreference clusters (optional)

Steps (run individually or all at once):
    1. chunking     -- Split documents using 8 methods + postprocessing
    2. mentions     -- Extract coreference mentions (needs GPU for maverick-coref)
    3. metrics      -- Compute 5 intrinsic quality metrics (post-processed chunks)
    4. raw_metrics  -- Compute metrics on raw (unprocessed) chunks for † methods
    5. analysis     -- Print Tables 1-2 and Figure 1
    6. table3       -- Print full Table 3: locally-computed scores vs paper values
    7. rag          -- Full RAG evaluation (needs OPENAI_API_KEY, GPU)

Typical workflow for full paper reproduction:
    python -m adaptive_chunking.paper.replicate \\
        --data-dir data/clair/ --output-dir results/ --device cuda:0 \\
        --steps chunking mentions metrics raw_metrics analysis table3

Steps 1-6 replicate the chunking evaluation (Tables 1-3, Figure 1).
Step 7 replicates the RAG evaluation (Tables 4-5). It requires an OpenAI API
key and is expensive to run.

Pre-computed coreference mentions are shipped in data/clair/mentions/ so users
without a GPU can skip the ``mentions`` step entirely.

Embedding model (metrics step):
    By default uses jinaai/jina-embeddings-v3 loaded locally via SentenceTransformer.
    Set JINA_API_KEY in the environment (or a .env file) to use the Jina REST API
    instead — much faster for large corpora and avoids downloading the model.
"""

import argparse
import asyncio
import json
import os
import tempfile
from functools import partial
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from ..chunking_utils import count_tokens

# ---------------------------------------------------------------------------
# Constants matching the paper's experimental setup
# ---------------------------------------------------------------------------

SEPARATORS = [
    r"(?<=\n)#{1}\s",
    r"(?<=\n)#{2}\s",
    r"(?<=\n)#{3}\s",
    r"(?<=\n)#{4}\s",
    r"(?<=\n)#{5}\s",
    r"(?<=\n)#{6}\s",
    r"(?<=\n)\s*\(?[A-Za-z0-9]{1,4}[.)]\s+",
    r"(?<=\n)\s*[-*·•●▪◦‣▸▹○◯‒–—]\s+",
    r"\n{2,}",
    r"\n",
    r"[.!?]\s+",
    r",\s+",
    r"\s+",
    r"",
]

CHUNKING_METHODS = [
    "page",
    "sentence",
    "langch_recurs_default",
    "langch_recurs_1100",
    "our_recurs_1100",
    "our_recurs_600",
    "semantic",
    "llm_regex",
]

METRICS = [
    "size_compliance",
    "block_integrity",
    "intrachunk_cohesion",
    "document_contextual_coherence",
    "references_completeness",
]

WEIGHTS = {m: 0.2 for m in METRICS}

count_tokens_func = partial(count_tokens, model="gpt-4o")


def _build_few_shot_prompt() -> str:
    """Load the few-shot example and build the LLM regex prompt."""
    example_path = Path(__file__).parent.parent / "few_shot_examples" / "hccr_report_md_adi.json"
    with open(example_path, "r", encoding="utf-8") as f:
        example = json.load(f)

    return f"""<Task>
Your task is to split a long document into self-contained and logically complete chunks to be used in a Retrieval Augmented Generation (RAG) system. Given a document text, choose the best **unique** regular-expression to be used as a *delimiter* to split it into small chunks using the Python `re` engine and the `re.split` function.
</Task>

<Output requirements>
You **must** return only the answer in this format:
    <regex>regex pattern here</regex>
</Output requirements>

<Splitting guidelines>
    - The regex pattern **must** be valid.
    - The chunks should be self-contained, logically complete and not too large.
    - Do not split paragraphs.
    - Do not split tables, marked between <Table> </Table> tags.
    - Do not split figures, marked between <Figure> </Figure> tags.
    - Do not split lists of short elements.
    - Do not split titles from the text that follows them.
    - Do not split footnotes from their parent text.
</Splitting guidelines>

<Splitting example>
    <Example of input text>
{example["input"]}
    </Example of input text>

    <Expected answer>
        <regex>{example["output"]}</regex>
    </Expected answer>
</Splitting example>

Now, please apply this method to the following text between <Input> and </Input> markers:
"""


# ---------------------------------------------------------------------------
# Step 1: Chunking
# ---------------------------------------------------------------------------

async def run_chunking(
    data_dir: Path,
    output_dir: Path,
    device: str = "cpu",
    openai_model: str = "gpt-4o",
    skip_llm_regex: bool = False,
    skip_semantic: bool = False,
):
    """Split all documents using 8 methods, then postprocess."""
    from ..splitters import RecursiveSplitter
    from ..split_documents import split_documents_from_dir
    from ..postprocessing import (
        split_oversized_chunks,
        split_oversized_chunks_from_df,
        merge_small_chunks_from_df,
        merge_small_chunks_to_neighbours,
    )

    parsed_docs_dir = data_dir / "adi_parsed"
    raw_dir = output_dir / "chunks" / "raw"
    no_oversizing_dir = output_dir / "chunks" / "no_oversizing"
    small_merged_dir = output_dir / "chunks" / "small_merged"

    # -- Build splitters --
    sync_splitters: dict = {}

    # page
    sync_splitters["page"] = None  # handled specially in split_documents_from_dir

    # sentence (stanza, 5 per chunk)
    from .splitters import SentenceSplitter
    sync_splitters["sentence"] = SentenceSplitter(
        method="stanza", sentences_per_chunk=5, device=device,
    )

    # langchain recursive default
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    sync_splitters["langch_recurs_default"] = RecursiveCharacterTextSplitter()

    # langchain recursive 1100
    sync_splitters["langch_recurs_1100"] = RecursiveCharacterTextSplitter(
        separators=SEPARATORS,
        chunk_size=1100,
        chunk_overlap=0,
        is_separator_regex=True,
        keep_separator="start",
        length_function=count_tokens_func,
    )

    # our recursive 1100
    sync_splitters["our_recurs_1100"] = RecursiveSplitter(
        separators=SEPARATORS,
        chunk_size=1100,
        chunk_overlap=0,
        is_separator_regex=True,
        attach_separator_to="start",
        length_function=count_tokens_func,
        merging="to_chunk_size",
        merging_order="forward",
    )

    # our recursive 600
    sync_splitters["our_recurs_600"] = RecursiveSplitter(
        separators=SEPARATORS,
        chunk_size=600,
        chunk_overlap=0,
        is_separator_regex=True,
        attach_separator_to="start",
        length_function=count_tokens_func,
        merging="to_chunk_size",
        merging_order="forward",
    )

    # semantic chunker
    if not skip_semantic:
        import torch
        from langchain_community.embeddings import HuggingFaceEmbeddings
        from .splitters import SemanticChunkerWrapper

        embeddings = HuggingFaceEmbeddings(
            model_name="Qwen/Qwen3-Embedding-0.6B",
            model_kwargs={
                "device": device,
                "model_kwargs": {
                    "attn_implementation": "flash_attention_2",
                    "torch_dtype": torch.bfloat16,
                },
                "tokenizer_kwargs": {"padding_side": "left"},
            },
            encode_kwargs={"batch_size": 16},
        )
        sync_splitters["semantic"] = SemanticChunkerWrapper(
            embeddings=embeddings,
            breakpoint_threshold_type="gradient",
        )

    # LLM regex (async, requires OpenAI key)
    async_splitters: dict = {}
    if not skip_llm_regex:
        import openai
        from .splitters import LLMRegexSplitter

        client = openai.AsyncOpenAI()

        async def _llm_completion(prompt: str) -> str:
            resp = await client.chat.completions.create(
                model=openai_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            return resp.choices[0].message.content

        async_splitters["llm_regex"] = LLMRegexSplitter(
            base_prompt=_build_few_shot_prompt(),
            async_client_completion_func=_llm_completion,
            count_tokens_func=count_tokens_func,
            context_tokens=8000,
        )

    # -- Run chunking --
    print("\n=== Step 1a: Chunking (raw) ===\n")
    await split_documents_from_dir(
        parsed_docs_dir=parsed_docs_dir,
        sync_splitters=sync_splitters,
        async_splitters=async_splitters,
        output_dir=raw_dir,
        count_tokens_func=count_tokens_func,
        skip_non_english=False,
        replace_all_results=True,
    )

    # -- Postprocessing: split oversized --
    print("\n=== Step 1b: Split oversized chunks ===\n")
    oversized_splitter = RecursiveSplitter(
        separators=SEPARATORS,
        chunk_size=1100,
        chunk_overlap=0,
        is_separator_regex=True,
        attach_separator_to="start",
        length_function=count_tokens_func,
        merging="to_chunk_size",
    )
    split_oversized_func = lambda chunks: split_oversized_chunks(
        chunks, oversized_splitter, count_tokens_func, max_chunk_tokens=1100,
    )

    methods_to_regularize = {"page", "sentence", "semantic", "llm_regex"}
    if skip_semantic:
        methods_to_regularize.discard("semantic")
    if skip_llm_regex:
        methods_to_regularize.discard("llm_regex")

    split_oversized_chunks_from_df(
        parsed_docs_dir=parsed_docs_dir,
        chunks_path=raw_dir / "chunks.parquet",
        output_dir=no_oversizing_dir,
        methods_to_be_regularized=methods_to_regularize,
        split_oversized_func=split_oversized_func,
        count_tokens_func=count_tokens_func,
        replace_all_results=True,
    )

    # -- Postprocessing: merge small chunks --
    print("\n=== Step 1c: Merge small chunks ===\n")
    merge_func = lambda chunks: merge_small_chunks_to_neighbours(
        chunks, count_tokens_func, min_limit=100, max_limit=1150, merge_to="next",
    )

    all_methods = set(sync_splitters.keys()) | set(async_splitters.keys())
    merge_small_chunks_from_df(
        parsed_docs_dir=parsed_docs_dir,
        chunks_path=no_oversizing_dir / "chunks.parquet",
        output_dir=small_merged_dir,
        methods_to_be_regularized=all_methods,
        merge_small_chunks_func=merge_func,
        count_tokens_func=count_tokens_func,
        replace_all_results=True,
    )

    print("\nChunking complete. Results saved to:", output_dir / "chunks")


# ---------------------------------------------------------------------------
# Step 2: Mention extraction
# ---------------------------------------------------------------------------

def run_mentions(data_dir: Path, output_dir: Path):
    """Extract coreference mentions for the references completeness metric."""
    from ..extract_mentions import find_mentions_per_origin

    print("\n=== Step 2: Mention extraction ===\n")

    import spacy
    from ..metrics import CoreferenceSolver

    parsed_docs_dir = data_dir / "adi_parsed"

    coref_solver = CoreferenceSolver(max_context_tokens=2000)
    spacy_model = spacy.load("en_core_web_sm")
    spacy_model.max_length = 10_000_000

    mentions_dir = output_dir / "mentions"
    find_mentions_per_origin(
        parsed_docs_dir=parsed_docs_dir,
        models={"coref_solver": coref_solver, "spacy_model": spacy_model},
        output_dir=mentions_dir,
        skip_non_english=False,
    )
    print("\nMentions saved to:", mentions_dir)


# ---------------------------------------------------------------------------
# Step 3: Metrics
# ---------------------------------------------------------------------------

def _make_embedder(device: str = "cpu"):
    """Return a sentence embedder: Jina API if JINA_API_KEY is set, else local model."""
    if os.environ.get("JINA_API_KEY"):
        from ..jina_embedder import JinaEmbedder
        print("JINA_API_KEY found — using Jina Embeddings API (fast, no GPU needed)")
        return JinaEmbedder()
    from sentence_transformers import SentenceTransformer
    print("Loading jinaai/jina-embeddings-v3 locally (set JINA_API_KEY to use the API instead) ...")
    m = SentenceTransformer("jinaai/jina-embeddings-v3", trust_remote_code=True)
    return m.to(device)


def _resolve_mentions_dir(data_dir: Path, output_dir: Path) -> Path:
    mentions_dir = output_dir / "mentions"
    if not mentions_dir.exists() or not any(mentions_dir.glob("*.parquet")):
        precomputed = data_dir / "mentions"
        if precomputed.exists() and any(precomputed.glob("*.parquet")):
            mentions_dir = precomputed
            print(f"Using pre-computed mentions from {mentions_dir}")
        else:
            print(
                "WARNING: No mentions found. Run the 'mentions' step first "
                "or provide pre-computed mentions in data_dir/mentions/. "
                "The references_completeness metric will be skipped."
            )
    return mentions_dir


def run_metrics(data_dir: Path, output_dir: Path, device: str = "cpu", batch_size: int = 32):
    """Compute the 5 intrinsic quality metrics on post-processed (small_merged) chunks."""
    from ..compute_metrics import compute_metrics_per_origin

    print("\n=== Step 3: Metrics computation ===\n")

    embedder = _make_embedder(device)
    mentions_dir = _resolve_mentions_dir(data_dir, output_dir)
    metrics_dir = output_dir / "results"

    compute_metrics_per_origin(
        chunks_dir=output_dir / "chunks" / "small_merged",
        mentions_dir=mentions_dir,
        parsed_docs_dir=data_dir / "adi_parsed",
        models={"sentence_embedder": embedder},
        output_dir=metrics_dir,
        batch_size=batch_size,
    )
    print("\nMetrics saved to:", metrics_dir)


# ---------------------------------------------------------------------------
# Step 4: Raw metrics for † methods (needed for Table 3)
# ---------------------------------------------------------------------------

# Methods shown without post-processing in paper Table 3 (marked with †)
_DAGGER_METHODS = {"page", "sentence", "semantic", "langch_recurs_1100", "langch_recurs_default"}


def run_raw_metrics(data_dir: Path, output_dir: Path, device: str = "cpu", batch_size: int = 32):
    """Compute metrics on raw (unprocessed) chunks for the † methods in Table 3.

    The paper shows some methods without post-processing to preserve their
    original design. This step produces results/results_raw/ which is required
    by the ``table3`` step.
    """
    from ..compute_metrics import compute_metrics_per_origin
    import pandas as pd

    print("\n=== Step 4: Raw metrics for † methods ===\n")
    print(f"  Methods: {sorted(_DAGGER_METHODS)}\n")

    embedder = _make_embedder(device)
    mentions_dir = _resolve_mentions_dir(data_dir, output_dir)
    raw_chunks_path = output_dir / "chunks" / "raw" / "chunks.parquet"
    raw_output_dir = output_dir / "results_raw"

    df_all = pd.read_parquet(raw_chunks_path)
    df_filtered = df_all[df_all["method"].isin(_DAGGER_METHODS)].copy()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_chunks = Path(tmpdir) / "chunks"
        tmp_chunks.mkdir()
        df_filtered.to_parquet(tmp_chunks / "chunks.parquet")
        compute_metrics_per_origin(
            chunks_dir=tmp_chunks,
            mentions_dir=mentions_dir,
            parsed_docs_dir=data_dir / "adi_parsed",
            models={"sentence_embedder": embedder},
            output_dir=raw_output_dir,
            batch_size=batch_size,
        )
    print("\nRaw metrics saved to:", raw_output_dir)


# ---------------------------------------------------------------------------
# Step 5: Analysis (Tables 1-2, Figure 1)
# ---------------------------------------------------------------------------

def run_analysis(output_dir: Path, methods: list[str] | None = None):
    """Print summary tables matching the paper's Tables 1-2 and Figure 1."""
    from .analysis import (
        show_chunking_overall_report,
        show_chunking_overall_metametrics,
        plot_metric_correlations,
    )

    print("\n=== Step 5: Analysis ===\n")

    methods = methods or CHUNKING_METHODS
    metrics_path = output_dir / "results" / "chunking_metrics.parquet"
    chunks_path = output_dir / "chunks" / "small_merged" / "chunks.parquet"

    print("--- Table 1: Overall chunking report ---\n")
    show_chunking_overall_report(
        df_path=metrics_path,
        chunking_methods=methods,
        metrics=METRICS,
        weights=WEIGHTS,
    )

    print("\n--- Table 2: Meta metrics ---\n")
    show_chunking_overall_metametrics(
        df_chunks_path=chunks_path,
        df_metrics_path=metrics_path,
        chunking_methods=methods,
        weights=WEIGHTS,
    )

    print("\n--- Figure 1: Metric correlations ---\n")
    plot_metric_correlations(
        df_path=metrics_path,
        metrics=METRICS,
        chunking_methods=methods,
    )


# ---------------------------------------------------------------------------
# Step 6: Table 3 — full comparison of locally-computed scores vs paper values
# ---------------------------------------------------------------------------

# Paper Table 3 published values: (tag, RC, ICC, DCC, BI, SC, mean)
# * = post-processed (small_merged), † = no postprocessing (raw)
_PAPER_TABLE3 = {
    "our_recurs_1100":       ("*",  99.0, 66.6, 89.7, 98.1, 100.0, 90.68),
    "our_recurs_600":        ("*",  97.2, 69.6, 84.7, 94.8, 100.0, 89.24),
    "page":                  ("*",  97.2, 69.2, 86.4, 99.9,  99.9, 90.52),
    "llm_regex":             ("*",  98.0, 70.9, 82.4, 98.1,  99.6, 89.80),
    "langch_recurs_1100":    ("†",  98.4, 64.7, 86.8, 98.6,  93.3, 88.37),
    "langch_recurs_default": ("†",  96.1, 65.6, 88.8, 95.0,  97.7, 88.62),
    "page_raw":              ("†",  97.1, 69.3, 86.1, 100.0, 92.7, 89.03),
    "semantic":              ("†",  97.5, 69.3, 76.3, 91.3,  48.1, 76.49),
    "sentence":              ("†",  86.3, 78.4, 72.5, 61.9,  67.2, 73.26),
}

_TABLE3_METRICS = [
    "references_completeness",
    "intrachunk_cohesion",
    "document_contextual_coherence",
    "block_integrity",
    "size_compliance",
]
_TABLE3_SHORT = ["RC", "ICC", "DCC", "BI", "SC"]

_TABLE3_DISPLAY_ORDER = [
    "our_recurs_1100", "our_recurs_600", "page", "llm_regex",
    "langch_recurs_1100", "langch_recurs_default", "page_raw",
    "semantic", "sentence",
]

_TABLE3_DISPLAY_NAMES = {
    "our_recurs_1100":       "our recursive (s=1100)",
    "our_recurs_600":        "our recursive (s=600)",
    "page":                  "page (post-processed)",
    "llm_regex":             "LLM regex",
    "langch_recurs_1100":    "LC recursive (s=1100)",
    "langch_recurs_default": "LC recursive (default)",
    "page_raw":              "page (raw)",
    "semantic":              "semantic",
    "sentence":              "sentence",
}


def run_table3(output_dir: Path):
    """Print full Table 3: locally-computed scores vs published paper values.

    Requires both the ``metrics`` step (results/results/) and the
    ``raw_metrics`` step (results/results_raw/) to have been completed.
    """
    import numpy as np
    import pandas as pd
    from tabulate import tabulate

    print("\n=== Step 6: Table 3 reproduction ===\n")

    merged_path = output_dir / "results" / "chunking_metrics.parquet"
    raw_path = output_dir / "results_raw" / "chunking_metrics.parquet"

    # Load available parquets
    dfs: dict = {}
    for path in [merged_path, raw_path]:
        if path.exists():
            dfs[path] = pd.read_parquet(path)

    # Which parquet + column name to use for each display key
    source = {
        "our_recurs_1100":       (merged_path, "our_recurs_1100"),
        "our_recurs_600":        (merged_path, "our_recurs_600"),
        "page":                  (merged_path, "page"),
        "llm_regex":             (merged_path, "llm_regex"),
        "langch_recurs_1100":    (raw_path,    "langch_recurs_1100"),
        "langch_recurs_default": (raw_path,    "langch_recurs_default"),
        "page_raw":              (raw_path,    "page"),
        "semantic":              (raw_path,    "semantic"),
        "sentence":              (raw_path,    "sentence"),
    }

    headers = ["Method", "tag"] + _TABLE3_SHORT + ["our mean", "paper mean", "delta"]
    rows = []
    pending = []

    for key in _TABLE3_DISPLAY_ORDER:
        tag, *paper_vals = _PAPER_TABLE3[key]
        paper_mean = paper_vals[5]
        src_path, col = source[key]
        display = _TABLE3_DISPLAY_NAMES[key]

        df = dfs.get(src_path)
        if df is None:
            rows.append([display, tag] + ["N/A"] * 5 + ["—", f"{paper_mean:.2f}", "pending"])
            pending.append(key)
            continue

        sub = df[df["chunking_method"] == col]
        if sub.empty:
            rows.append([display, tag] + ["N/A"] * 5 + ["—", f"{paper_mean:.2f}", "pending"])
            pending.append(key)
            continue

        agg = sub.groupby("metric_name")["score"].agg(["mean", "std"])
        cells = []
        our_mean = 0.0
        for m in _TABLE3_METRICS:
            mv = agg["mean"].get(m, np.nan) * 100
            sv = agg["std"].get(m, np.nan) * 100
            cells.append(f"{mv:.1f}±{sv:.1f}" if not np.isnan(mv) else "N/A")
            our_mean += (agg["mean"].get(m, np.nan) * 100) / len(_TABLE3_METRICS)

        delta = our_mean - paper_mean
        rows.append([display, tag] + cells + [f"{our_mean:.2f}", f"{paper_mean:.2f}", f"{delta:+.2f}%"])

    print("=" * 110)
    print("PAPER TABLE 3 REPRODUCTION — all scores computed locally")
    print("  * = post-processed (small_merged)   † = no postprocessing (raw)")
    print("=" * 110)
    print(tabulate(rows, headers=headers, tablefmt="simple"))

    if pending:
        missing_steps = []
        if any(source[k][0] == merged_path for k in pending):
            missing_steps.append("metrics")
        if any(source[k][0] == raw_path for k in pending):
            missing_steps.append("raw_metrics")
        print(f"\n  Pending methods: {[_TABLE3_DISPLAY_NAMES[k] for k in pending]}")
        print(f"  Run missing steps first: --steps {' '.join(missing_steps)}")
    else:
        print("\n  All methods computed locally.")


# ---------------------------------------------------------------------------
# Step 7: RAG evaluation (optional, requires OpenAI API key)
# ---------------------------------------------------------------------------

async def run_rag(
    data_dir: Path,
    output_dir: Path,
    device: str = "cpu",
    openai_model: str = "gpt-4.1",
    batch_size: int = 16,
):
    """Run the full RAG pipeline: QA generation, indexing, retrieval, generation, evaluation."""
    import openai
    from .rag_utils import (
        generate_qa_pairs,
        load_json_chunks_from_dir,
        index_documents,
        create_retrieval_pipeline,
        run_retrieval_for_generated_questions,
    )
    from .rag_eval import (
        generate_answers,
        evaluate_rag_results_generated_questions,
        show_rag_results_generated_questions,
    )
    from .analysis import output_best_chunks, output_selected_chunks
    from deepeval.models import GPTModel

    print("\n=== Step 7: RAG evaluation ===\n")

    parsed_docs_dir = data_dir / "adi_parsed"
    chunks_path = output_dir / "chunks" / "small_merged" / "chunks.parquet"
    metrics_path = output_dir / "results" / "chunking_metrics.parquet"
    rag_dir = output_dir / "rag"

    # 5a: Select best chunks + baseline chunks for RAG comparison
    print("--- 5a: Selecting chunks for RAG ---")

    best_chunks_dir = rag_dir / "chunks" / "best"
    output_best_chunks(
        chunks_df_path=chunks_path,
        metrics_df_path=metrics_path,
        weights=WEIGHTS,
        output_dir=best_chunks_dir,
    )

    langch_chunks_dir = rag_dir / "chunks" / "langch_recurs_default"
    page_chunks_dir = rag_dir / "chunks" / "page"
    output_selected_chunks(
        chunks_df_paths={"small_merged": chunks_path},
        selection=[
            {"chunks_df": "small_merged", "chunking_method": "langch_recurs_default", "output_dir": langch_chunks_dir},
            {"chunks_df": "small_merged", "chunking_method": "page", "output_dir": page_chunks_dir},
        ],
    )

    # 5b: Generate QA pairs
    print("\n--- 5b: Generating QA pairs ---")
    client = openai.AsyncOpenAI()

    qa_generation_prompt = """You are a domain expert creating evaluation questions for a retrieval-augmented generation system.
Given the following document excerpt, generate {qa_pairs_per_document} diverse question-answer pairs that:
- Test understanding of key concepts, facts, and relationships in the document
- Have clear, specific answers that can be found in the text
- Vary in difficulty and type (factual, analytical, comparative)

Document excerpt:
{document_context}

Generate the question-answer pairs."""

    async def _qa_completion(prompt: str):
        from .rag_utils import QuestionAnswerPairList
        resp = await client.responses.parse(
            model=openai_model,
            input=[{"role": "user", "content": prompt}],
            text_format=QuestionAnswerPairList,
        )
        return resp

    qa_dir = rag_dir / "queries"
    await generate_qa_pairs(
        parsed_docs_dir=parsed_docs_dir,
        outputs_dir=qa_dir,
        client_completion_func=_qa_completion,
        qa_generation_prompt=qa_generation_prompt,
        qa_pairs_per_document=3,
        max_context_tokens=10000,
    )

    # 5c: Index and retrieve for each method
    rag_methods = {
        "best": best_chunks_dir,
        "langch_recurs_default": langch_chunks_dir,
        "page": page_chunks_dir,
    }

    for method_name, chunks_dir in rag_methods.items():
        print(f"\n--- 5c: Indexing & retrieval for '{method_name}' ---")

        method_rag_dir = rag_dir / "retrieval" / method_name
        docs = load_json_chunks_from_dir(chunks_dir)

        index_documents(
            documents=docs,
            output_dir=method_rag_dir,
            embedding_model_name="Qwen/Qwen3-Embedding-4B",
            device=device,
            batch_size=batch_size,
        )

        retrieval_pipeline = create_retrieval_pipeline(
            document_store_path=method_rag_dir / "document_store.json",
            embedding_model="Qwen/Qwen3-Embedding-4B",
            reranker_model="Snowflake/snowflake-arctic-embed-l-v2.0",
            device=device,
        )

        run_retrieval_for_generated_questions(
            qa_pairs_json_path=qa_dir / "generated_qa_pairs.json",
            output_dir=method_rag_dir,
            retrieval_pipeline=retrieval_pipeline,
        )

    # 5d: Generate answers
    qa_prompt = """You are a helpful assistant. Answer the question based ONLY on the provided context.
If the context does not contain enough information to answer the question, respond with "I don't know based on the provided context."

Context:
{context_str}

Question: {question_str}

Answer:"""

    async def _gen_completion(prompt: str) -> str:
        resp = await client.chat.completions.create(
            model=openai_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            top_p=1,
        )
        return resp.choices[0].message.content

    for method_name in rag_methods:
        print(f"\n--- 5d: Generating answers for '{method_name}' ---")
        method_rag_dir = rag_dir / "retrieval" / method_name

        await generate_answers(
            retrieval_results_path=method_rag_dir / "generated_questions_retrieval_results.json",
            output_dir=method_rag_dir,
            output_file_name="generated_questions_generation_results.json",
            async_client_completion_func=_gen_completion,
            qa_prompt=qa_prompt,
        )

    # 5e: Evaluate
    gpt_model = GPTModel(model=openai_model)

    for method_name in rag_methods:
        print(f"\n--- 5e: Evaluating '{method_name}' ---")
        method_rag_dir = rag_dir / "retrieval" / method_name

        evaluate_rag_results_generated_questions(
            generation_results_path=method_rag_dir / "generated_questions_generation_results.json",
            gpt_model=gpt_model,
            output_dir=method_rag_dir,
            skip_correctness_with="I don't know based on the provided context.",
        )

    # 5f: Show results
    print("\n--- Table 5: RAG results ---\n")
    eval_paths = {
        method_name: rag_dir / "retrieval" / method_name / "generated_questions_evaluation_results.json"
        for method_name in rag_methods
    }
    show_rag_results_generated_questions(eval_paths, evaluation_llm=openai_model)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Replicate LREC 2026 paper results.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--data-dir", type=Path, required=True,
        help="Root data directory (e.g. data/clair/) containing adi_parsed/ and optionally mentions/.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results"),
        help="Directory to write all outputs (default: results/).",
    )
    parser.add_argument(
        "--steps", nargs="+", default=["all"],
        choices=["all", "chunking", "mentions", "metrics", "raw_metrics", "analysis", "table3", "rag"],
        help="Which steps to run (default: all). For full Table 3 reproduction, run: metrics raw_metrics analysis table3.",
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        help="Device for ML models, e.g. 'cuda:0' (default: cpu).",
    )
    parser.add_argument(
        "--skip-llm-regex", action="store_true",
        help="Skip the LLM regex splitter (requires OpenAI API key).",
    )
    parser.add_argument(
        "--skip-semantic", action="store_true",
        help="Skip the semantic chunker (requires GPU + flash_attention_2).",
    )
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="Batch size for embedding computations (default: 32).",
    )

    args = parser.parse_args()
    steps = set(args.steps)
    run_all = "all" in steps

    if run_all or "chunking" in steps:
        asyncio.run(run_chunking(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            device=args.device,
            skip_llm_regex=args.skip_llm_regex,
            skip_semantic=args.skip_semantic,
        ))

    if run_all or "mentions" in steps:
        run_mentions(data_dir=args.data_dir, output_dir=args.output_dir)

    if run_all or "metrics" in steps:
        run_metrics(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            device=args.device,
            batch_size=args.batch_size,
        )

    if run_all or "raw_metrics" in steps:
        run_raw_metrics(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            device=args.device,
            batch_size=args.batch_size,
        )

    if run_all or "analysis" in steps:
        run_analysis(output_dir=args.output_dir)

    if run_all or "table3" in steps:
        run_table3(output_dir=args.output_dir)

    if run_all or "rag" in steps:
        asyncio.run(run_rag(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            device=args.device,
        ))


if __name__ == "__main__":
    main()
