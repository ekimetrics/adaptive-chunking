import os
import pandas as pd
import json
import asyncio
from typing import Any, Callable
from pathlib import Path
import time
from .chunking_utils import count_tokens, is_high_confidence_non_english
from .postprocessing import (
    repair_gaps_between_chunks,
    check_chunk_gaps,
    build_chunk_records,
    upsert_parquet,
)

async def split_documents_from_dir(
    parsed_docs_dir: str | Path,
    sync_splitters: dict[str, Any],
    async_splitters: dict[str, Any],
    output_dir: str | Path,
    count_tokens_func: Callable[[str], int] = count_tokens,
    skip_non_english: bool = True,
    replace_all_results: bool = False,
    ):
    
    # read parsed docs dir
    parsed_docs_dir = Path(parsed_docs_dir)
    print(f"\nLoading parsed docs from {parsed_docs_dir.name}\n")

    parsed_docs = {}
    for json_path in parsed_docs_dir.glob("*.json"):
        print(f"Document: {json_path.name}")
        with json_path.open("r") as f:
            parsed_docs[json_path.with_suffix('').name] = json.load(f)
    
    if len(parsed_docs) == 0:
        print(f"No parsed docs found in {parsed_docs_dir.name}")
        return None

    # creates dicts for storing splits and timing
    splits_per_doc = {}
    time_per_doc_per_splitter = {}

    # create dicts of texts to use in splitting loops
    doc_texts_dict = {}
    doc_pages_dict = {}

    for doc_name in parsed_docs:
        # load document text
        parsed_doc_pages_dict = parsed_docs[doc_name]["pages"]
        doc_text = parsed_docs[doc_name]["full_text"]
        doc_titles = parsed_docs[doc_name]["titles"]
        doc_pages = []
        for pg in parsed_doc_pages_dict:
            page_text = parsed_doc_pages_dict[pg]
            doc_pages.append(page_text)
        
        if not doc_text:
            print(f"Skipping document {doc_name}: empty")
            continue

        # detect language and filter out non english if desirable
        if skip_non_english and is_high_confidence_non_english(doc_text):
            print(f"Skipping document {doc_name}: detected non-English")
            continue
        
        doc_texts_dict[doc_name] = doc_text
        doc_pages_dict[doc_name] = doc_pages
        splits_per_doc[doc_name] = {}
        time_per_doc_per_splitter[doc_name] = {}

    # chunking loop for sync splitting
    if sync_splitters:
        print("\nSync splitting documents...\n")

        for doc_name in doc_texts_dict:
            # chunk by page
            if 'page' in sync_splitters:
                print(f"Chunking: {doc_name}, page")
                start_time = time.time()

                splits_per_doc[doc_name]["page"] = doc_pages_dict[doc_name]
                recover_success = check_chunk_gaps(splits_per_doc[doc_name]["page"], doc_texts_dict[doc_name])
                assert recover_success==True, f"raw page gap recovery failed"
                time_per_doc_per_splitter[doc_name]["page"] = time.time() - start_time

            # chunk by reckitt method, special case: split by pages then recursive
            if 'reckitt' in sync_splitters:
                start_time = time.time()

                print(f"Chunking: {doc_name}, reckitt")
                chunks = []
                for page in doc_pages_dict[doc_name]:
                    chunks.extend(sync_splitters["reckitt"].split_text(page))

                repaired_chunks = repair_gaps_between_chunks(chunks=chunks, text=doc_texts_dict[doc_name])
                splits_per_doc[doc_name]["reckitt"] = repaired_chunks

                time_per_doc_per_splitter[doc_name]["reckitt"] = time.time() - start_time

                recover_success = check_chunk_gaps(splits_per_doc[doc_name]["reckitt"], doc_texts_dict[doc_name])
                assert recover_success==True, "sync chunking gap recovery failed"
                
            # chunk using standard methods
            splitters = {k:v for k,v in sync_splitters.items() if k not in {"page", "reckitt"}}

            for method, splitter in splitters.items():
                start_time = time.time()

                print(f"Chunking: {doc_name}, {method}")

                # chunk using sync splitter
                splits = splitter.split_text(doc_texts_dict[doc_name])

                # repair any gaps
                repaired_splits = repair_gaps_between_chunks(chunks=splits, text=doc_texts_dict[doc_name])

                splits_per_doc[doc_name][method] = repaired_splits
                time_per_doc_per_splitter[doc_name][method] = time.time() - start_time

                # recheck gaps
                recover_success = check_chunk_gaps(splits_per_doc[doc_name][method], doc_texts_dict[doc_name])
                assert recover_success==True, "sync chunking gap recovery failed"

    # chunking loop for async splitting
    if async_splitters:
        print("\nAsync splitting documents...\n")
        
        for method, splitter in async_splitters.items():
            start_time = time.time()
            # chunk using async splitter
            tasks = [splitter.split_text(doc_text) for doc_text in doc_texts_dict.values()]
            results = await asyncio.gather(*tasks)


            for doc_name, splits in zip(doc_texts_dict, results):
                print(f"Chunking: {doc_name}, {method}")
                # repair any gaps
                splits_per_doc[doc_name][method] = repair_gaps_between_chunks(chunks=splits, text=doc_texts_dict[doc_name])
                time_per_doc_per_splitter[doc_name][method] = time.time() - start_time

                # recheck gaps
                recover_success = check_chunk_gaps(splits_per_doc[doc_name][method], doc_texts_dict[doc_name])
                assert recover_success==True, "async chunking gap recovery failed"

    # save chunks parquet
    records = build_chunk_records(
        splits_per_doc, parsed_docs, record_type="raw", count_tokens_func=count_tokens_func
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # methods being (re)written = every supplied sync/async splitter
    methods_to_replace = set(sync_splitters or {}) | set(async_splitters or {})

    upsert_parquet(pd.DataFrame(records), output_dir / "chunks.parquet",
                   methods_to_replace, replace_all_results)

    # Build performances dataframe with times (records carry sync/async type)
    perf_records = []
    for doc_name, method_dict in time_per_doc_per_splitter.items():
        for method, time_spent in method_dict.items():
            method_type = "async" if (async_splitters and method in async_splitters) else "sync"
            perf_records.append({
                "doc_name": doc_name,
                "method": method,
                "method_type": method_type,
                "time": time_spent,
            })

    upsert_parquet(pd.DataFrame(perf_records), output_dir / "performances.parquet",
                   methods_to_replace, replace_all_results)
