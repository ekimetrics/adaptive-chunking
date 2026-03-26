# LLM.md

This file provides guidance to LLM-based coding tools when working with this repository.

## Project Overview

A multi-domain document chunking evaluation framework for comparing chunking strategies using NLP metrics and RAG evaluation. Associated with the LREC 2026 paper *"Adaptive Chunking: Optimizing Chunking-Method Selection for RAG"*.

The pipeline:
1. Parses PDF documents to Markdown (multiple backends)
2. Chunks them with 8 methods (page, sentence, LangChain recursive default/1100, our recursive 1100/600, semantic, LLM-regex)
3. Applies post-processing (split oversized, merge tiny)
4. Evaluates chunks with 5 intrinsic metrics: References Completeness, Intrachunk Cohesion, Document Contextual Coherence, Block Integrity, Size Compliance
5. Selects best chunking method per document (Adaptive Chunking)
6. Runs RAG evaluation

## Setup & Environment

- **Python**: >=3.11
- **Package name**: `adaptive-chunking` (import as `adaptive_chunking`)
- **Install**:
  - Core: `pip install -e .`
  - With parsing backends: `pip install -e ".[parsing]"`
  - Paper reproduction: `pip install -e ".[paper]"`
  - Development: `pip install -e ".[dev]"`
- **Environment variables** (`.env` at project root):
  - `ADI_ENDPOINT`, `ADI_KEY` — Azure Document Intelligence (only for `AzureDIParser`)
  - `OPENAI_API_KEY` — OpenAI embeddings (RAG and semantic chunking)
  - `GROQ_API_KEY` — Groq LLM (RAG evaluation and coreference resolution)
  - `JINA_API_KEY` — Jina embeddings API (optional, speeds up ICC/DCC metrics from ~9h to ~30min)
- **spaCy models**: Some metrics require `python -m spacy download en_core_web_sm`
- **Tests**: `pytest`

## Stability Constraint

**Avoid updating libraries or models** — changing versions may alter results. Key pinned dependencies:
- `jinaai/jina-embeddings-v3` — ICC and DCC metrics
- `Qwen/Qwen3-Embedding-0.6B` — semantic chunking
- `maverick-coref` — References Completeness
- `torch==2.6.0` — reproducibility

If a dependency change is necessary, explain the impact on reproducibility first.

## Architecture

### Package Structure (`src/adaptive_chunking/`)

Core and paper modules are separated: core installs by default, paper requires `[paper]` extras.

#### Core Modules

| Module | Purpose |
|--------|---------|
| `splitters.py` | `RecursiveSplitter` (adaptive recursive chunking with configurable separators, merge modes, overlap), plus `group_chunks()`, `combine_blocks()`, `regex_splitter()`. |
| `metrics.py` | Quality metrics: size compliance, intrachunk cohesion, contextual coherence, block integrity, filtered missing reference error. |
| `parsing.py` | `BaseParser` ABC with three backends: `DoclingParser`, `PyMuPDFParser`, `AzureDIParser`. Plus `ExcelParser`. |
| `postprocessing.py` | Chunk location in source text, gap detection/repair, page/title metadata. |
| `compute_metrics.py` | Orchestrates metric computation with incremental save + resumability. Entry point: `compute_metrics_per_origin()`. |
| `split_documents.py` | Orchestrates splitting across documents. Entry point: `split_documents_from_dir()`. |
| `extract_mentions.py` | Coreference resolution and entity-pronoun pair extraction. Entry point: `find_mentions_per_origin()`. |
| `chunking_utils.py` | Token counting (tiktoken). |
| `jina_embedder.py` | Jina REST API drop-in for `SentenceTransformer`. Auto-used when `JINA_API_KEY` is set. |

#### Paper Reproduction Modules (`paper/`)

| Module | Purpose |
|--------|---------|
| `paper/replicate.py` | End-to-end CLI: `python -m adaptive_chunking.paper.replicate` |
| `paper/splitters.py` | Baseline splitters: `SemanticChunkerWrapper`, `SentenceSplitter`, `LongContextSemanticSplitter`, `LLMRegexSplitter`. |
| `paper/rag_utils.py` | Haystack-based RAG pipeline with hybrid retrieval. |
| `paper/rag_eval.py` | Custom RAG evaluation metrics. |
| `paper/analysis.py` | Aggregation, statistical summaries, Tables 1–3, Figure 1. |
| `paper/visualization.py` | HTML overlays for split visualization. |

### Key Patterns

- **Lazy imports**: Heavy ML deps (sentence-transformers, sklearn, spacy, maverick, transformers) are lazy-imported inside functions, not at module level. Follow this pattern for any new heavy dependency.
- **Data format**: Parquet for chunks/mentions/metrics, JSON for parsed documents.
- **Parsed document JSON format**: `{"document_name": str, "pages": {page_num: markdown}, "full_text": str, "split_points": [int], "titles": [{title, start, end, level}]}`
- **Token counting**: via `tiktoken` with `o200k_base` encoding.

### Adding New Components

- **New chunking method**: Add to `splitters.py` (core) or `paper/splitters.py` (experimental), integrate in `split_documents.py`
- **New metric**: Add to `metrics.py` (use lazy imports), wire into `compute_metrics.py`
- **New parser**: Extend `BaseParser` in `parsing.py`, implement `parse_docs_in_dir()` and `convert_raw_results_to_markdown()`

## Replicating the Paper

```bash
# Full Table 3 reproduction (recommended)
python -m adaptive_chunking.paper.replicate \
  --data-dir data/clair/ --output-dir results/ --device cuda:0 \
  --steps chunking mentions metrics raw_metrics analysis table3

# Individual steps: chunking | mentions | metrics | raw_metrics | analysis | table3 | rag
```

The `metrics` step takes ~9 hours on an RTX 4090 with local model, or ~30 min with the Jina API. Both are resumable — if interrupted, rerun and already-computed documents are skipped.

### Post-processing design (CRITICAL for reproduction)

Paper Table 3 deliberately mixes two post-processing levels:

- **`*` methods** (our_recurs_1100, our_recurs_600, page post-processed, llm_regex): scored **after full post-processing** (oversized split + tiny-chunk merge) → `results/results/`
- **`†` methods** (sentence, semantic, page raw, langch_recurs_default, langch_recurs_1100): scored **without any post-processing (raw chunks)** → `results/results_raw/`

The `†` methods are shown as-designed to preserve how they work out-of-the-box. The `raw_metrics` step computes these, and `table3` prints the full comparison.

### Jina embedder notes

`jina_embedder.py` wraps the Jina REST API with the same `encode()` interface as `SentenceTransformer`. Key settings: `max_concurrent=3` (avoids 429 thundering-herd), jitter on retries, `MAX_CHARS=20000` truncation (jina-embeddings-v3 max ~8192 tokens).

## Key Data

- `data/clair/` — 33 parsed CLAIR corpus documents + pre-computed coreference mentions
- `results/` — generated outputs (not committed)
