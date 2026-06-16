import json
import os
import pandas as pd
from pathlib import Path
from time import time
from .chunking_utils import is_high_confidence_non_english

def find_mentions_per_origin(parsed_docs_dir: str|Path, models: dict, output_dir: str|Path, skip_non_english: bool = True):
    from .metrics import extract_entity_pronoun_pairs

    # load parsed documents
    parsed_docs_dir = Path(parsed_docs_dir)

    parsed_docs = {}
    for file_path in parsed_docs_dir.iterdir():
        if file_path.suffix == '.json':
            with open(file_path, 'r') as f:
                parsed_docs[file_path.with_suffix('').name] = json.load(f)
    
    # extract mentions
    coref_solver = models["coref_solver"]
    spacy_model = models["spacy_model"]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    times_per_doc = {}
    for doc_name in parsed_docs:
        full_text = parsed_docs[doc_name]["full_text"]

        # skip non english documents, the current coref solver is english only
        if skip_non_english and is_high_confidence_non_english(full_text):
            print(f"\nSkipping document {doc_name}: detected non-English")
            continue

        # Extract mentions
        start_time = time()
        print(f"\nExtracting mentions from document {doc_name}")

        print("Building clusters using coreference solver...")
        mentions = coref_solver.find_mentions(text=full_text)

        print("Building pronoun-entity pairs...")
        entity_pronoun_mention_pairs = extract_entity_pronoun_pairs(
            text=full_text,
            clusters=mentions,
            spacy_model=spacy_model
        )
        times_per_doc[doc_name] = time() - start_time

        # Create a DataFrame with the document data
        df = pd.DataFrame({
            "doc_name": [doc_name],
            "mentions": [mentions],
            "entity_pron_mentions": [entity_pronoun_mention_pairs]
        })
        
        # Save as parquet file
        doc_output_path = output_dir / f"{doc_name}.parquet"
        df.to_parquet(doc_output_path)
        print(f"Saved mentions to {doc_output_path}")

    # save times to file system
    perf_records = []
    for doc_name in times_per_doc:
        perf_records.append({"doc_name": doc_name, "time": times_per_doc[doc_name]})
    perf_df = pd.DataFrame(perf_records)
    perf_output_dir = output_dir / "performances"
    perf_output_dir.mkdir(parents=True, exist_ok=True)
    perf_output_path = perf_output_dir / "mentions_performance.parquet"
    perf_df.to_parquet(perf_output_path)