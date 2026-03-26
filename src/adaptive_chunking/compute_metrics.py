import json
import pandas as pd
from time import time
import numpy as np
from pathlib import Path

def compute_metrics_per_origin(
    chunks_dir: str | Path,
    mentions_dir: str | Path,
    parsed_docs_dir: str | Path,
    models: dict,
    output_dir: str,
    batch_size: int = 32,
    ):

    from .metrics import (
        compute_intrachunk_cohesion,
        compute_filtered_missing_ref_error,
        compute_contextual_coherence,
        compute_block_integrity,
        compute_chunk_embeddings,
        compute_size_compliance,
    )

    # load chunks parquet file
    chunks_dir = Path(chunks_dir)
    chunks_path = chunks_dir / "chunks.parquet"
    df = pd.read_parquet(chunks_path)
    
    splits_per_doc = {}
    split_lens_per_doc = {}
    for doc_name, group in df.groupby("doc_name"):
        splits_per_doc[doc_name] = {}
        split_lens_per_doc[doc_name] = {}
        for method, group_data in group.groupby("method"):
            splits_per_doc[doc_name][method] = group_data["chunk_text"].to_list()
            split_lens_per_doc[doc_name][method] = group_data["chunk_len"].to_list()
    
    # load mentions parquet files
    mentions_dir = Path(mentions_dir)

    entity_pron_pairs_per_doc = {}

    for pq_file in mentions_dir.glob("*.parquet"):
        df = pd.read_parquet(pq_file)
        doc_name = df["doc_name"].iloc[0]
        entity_pron_pairs_per_doc[doc_name] = df["entity_pron_mentions"].iloc[0]

    # load split points per doc and full text from parsed jsons
    parsed_docs_dir = Path(parsed_docs_dir)
    parser_splitpoints_per_doc = {}
    full_text_per_doc = {}
    for doc_path in parsed_docs_dir.glob("*.json"):
        with open(doc_path, "r") as file:
            doc = json.load(file)

        doc_name = doc_path.with_suffix("").name
        parser_splitpoints_per_doc[doc_name] = doc["split_points"]
        full_text_per_doc[doc_name] = doc["full_text"]

    # compute scores for each document
    sentence_embedder = models["sentence_embedder"]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_output_path = output_dir / "chunking_metrics.parquet"
    perf_output_path = output_dir / "metrics_performance.parquet"

    # load existing results for resumability
    existing_docs = set()
    if metrics_output_path.exists():
        existing_df = pd.read_parquet(metrics_output_path)
        existing_docs = set(existing_df["doc_name"].unique())
        print(f"Found existing results for {len(existing_docs)} docs, will skip them")

    scores_per_doc = {}
    times_per_doc = {}
    for doc_name in splits_per_doc:
        if doc_name in existing_docs:
            print(f"\nSkipping already-computed document {doc_name}")
            continue

        print(f"\nProcessing document {doc_name}")

        splits_per_method = splits_per_doc[doc_name]
        split_lens_per_method = split_lens_per_doc[doc_name]
        
        scores_per_doc[doc_name] = {}
        times_per_doc[doc_name] = {}
        
        # compute size compliance
        start_time = time()
        print("Computing size compliance")
        scores_per_doc[doc_name]["size_compliance"] = {}
        for method in splits_per_method:
            size_compliance = compute_size_compliance(
                chunks=splits_per_method[method],
                min_tokens=100,
                max_tokens=1100)
            scores_per_doc[doc_name]["size_compliance"][method] = size_compliance
        times_per_doc[doc_name]["size_compliance"] = time() - start_time

        # compute block integrity
        start_time = time()
        print("Computing block integrity")
        scores_per_doc[doc_name]["block_integrity"] = {}
        for method in splits_per_method:
            block_integrity = compute_block_integrity(
                chunks=splits_per_method[method],
                doc_split_points=parser_splitpoints_per_doc[doc_name],
                full_text=full_text_per_doc[doc_name],
                tolerance_chars=5)

            scores_per_doc[doc_name]["block_integrity"][method] = block_integrity
        times_per_doc[doc_name]["block_integrity"] = time() - start_time
        
        # precompute chunk embeddings
        start_time = time()
        print("Computing chunk embeddings")
        embeddings_per_method = {}
        for method in splits_per_method:
            chunk_embeddings = compute_chunk_embeddings(
                chunks=splits_per_method[method],
                model=sentence_embedder,
                batch_size=batch_size)
            embeddings_per_method[method] = chunk_embeddings
        times_per_doc[doc_name]["chunk_embeddings"] = time() - start_time

        # compute intrachunk cohesion
        start_time = time()
        print("Computing intrachunk cohesion")
        scores_per_doc[doc_name]["intrachunk_cohesion"] = {}
        for method in splits_per_method:
            intrachunk_cohesion = compute_intrachunk_cohesion(
                chunk_embeddings=embeddings_per_method[method],
                split_points=parser_splitpoints_per_doc[doc_name],
                chunks=splits_per_method[method],
                full_text=full_text_per_doc[doc_name],
                model=sentence_embedder,
                batch_size=batch_size,
            )
            scores_per_doc[doc_name]["intrachunk_cohesion"][method] = intrachunk_cohesion
        times_per_doc[doc_name]["intrachunk_cohesion"] = time() - start_time

        # compute contextual coherence
        start_time = time()
        print("Computing document contextual coherence")
        scores_per_doc[doc_name]["document_contextual_coherence"] = {}
        for method in splits_per_method:           
            contextual_coherence = compute_contextual_coherence(
                chunks=splits_per_method[method],
                chunk_embeddings=embeddings_per_method[method],
                full_text=full_text_per_doc[doc_name],
                model=sentence_embedder,
                window_context_tokens = 3000,
                window_step = 1,
                batch_size = batch_size)
            
            scores_per_doc[doc_name]["document_contextual_coherence"][method] = contextual_coherence
        times_per_doc[doc_name]["document_contextual_coherence"] = time() - start_time

        # compute references completeness
        start_time = time()
        print("Computing references completeness")
        scores_per_doc[doc_name]["references_completeness"] = {}
        
        if doc_name in entity_pron_pairs_per_doc:
            entity_pronoun_mention_pairs = entity_pron_pairs_per_doc[doc_name]
        else:
            entity_pronoun_mention_pairs = None

        for method in splits_per_method:
            filtered_missing_ref_error = compute_filtered_missing_ref_error(
                full_text=full_text_per_doc[doc_name],
                chunks=splits_per_method[method],
                entity_pron_pairs=entity_pronoun_mention_pairs,
            )
            if filtered_missing_ref_error != None:
                filtered_missing_ref_score = 1 - filtered_missing_ref_error
                scores_per_doc[doc_name]["references_completeness"][method] = filtered_missing_ref_score
            else:
                scores_per_doc[doc_name]["references_completeness"][method] = None

        times_per_doc[doc_name]["references_completeness"] = time() - start_time

        # compute basic metrics related to chunk lengths
        start_time = time()
        print("Computing basic chunk token metrics")
        scores_per_doc[doc_name]["avg_chunk_tokens"] = {}
        scores_per_doc[doc_name]["stddev_chunk_tokens"] = {}
        scores_per_doc[doc_name]["max_chunk_tokens"] = {}
        scores_per_doc[doc_name]["min_chunk_tokens"] = {}
        scores_per_doc[doc_name]["num_chunks"] = {}

        for method in split_lens_per_method:
            chunk_lens = np.array(split_lens_per_method[method])

            avg_chunk_tokens = np.mean(chunk_lens)
            max_chunk_tokens = np.max(chunk_lens)
            min_chunk_tokens = np.min(chunk_lens)
            stddev_chunk_tokens = np.std(chunk_lens)
            num_chunks = len(chunk_lens)

            scores_per_doc[doc_name]["num_chunks"][method] = num_chunks
            scores_per_doc[doc_name]["avg_chunk_tokens"][method] = avg_chunk_tokens
            scores_per_doc[doc_name]["stddev_chunk_tokens"][method] = stddev_chunk_tokens
            scores_per_doc[doc_name]["max_chunk_tokens"][method] = max_chunk_tokens
            scores_per_doc[doc_name]["min_chunk_tokens"][method] = min_chunk_tokens

        times_per_doc[doc_name]["basic_chunk_token_metrics"] = time() - start_time

        # save incrementally after each document
        new_records = []
        for metric in scores_per_doc[doc_name]:
            for method in scores_per_doc[doc_name][metric]:
                new_records.append({
                    "doc_name": doc_name,
                    "chunking_method": method,
                    "metric_name": metric,
                    "score": scores_per_doc[doc_name][metric][method],
                })
        new_df = pd.DataFrame(new_records)

        if metrics_output_path.exists():
            existing_df = pd.read_parquet(metrics_output_path)
            new_df = pd.concat([existing_df, new_df], ignore_index=True)
        new_df.to_parquet(metrics_output_path)

        new_perf_records = []
        for metric in times_per_doc[doc_name]:
            new_perf_records.append({"doc_name": doc_name, "metric": metric, "time": times_per_doc[doc_name][metric]})
        new_perf_df = pd.DataFrame(new_perf_records)

        if perf_output_path.exists():
            existing_perf_df = pd.read_parquet(perf_output_path)
            new_perf_df = pd.concat([existing_perf_df, new_perf_df], ignore_index=True)
        new_perf_df.to_parquet(perf_output_path)

        print(f"Saved results for {doc_name}")