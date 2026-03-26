from pathlib import Path
import pandas as pd
import json
from typing import Callable, List, Optional, Awaitable
from ..splitters import combine_blocks
from ..chunking_utils import count_tokens
from pydantic import BaseModel, Field
from openai.types.responses.parsed_response import ParsedResponse
from haystack import Document, Pipeline
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.embedders import (
    SentenceTransformersDocumentEmbedder,
    SentenceTransformersTextEmbedder)
from haystack.components.writers import DocumentWriter
from haystack.utils import ComponentDevice
from haystack.components.retrievers.in_memory import (
    InMemoryBM25Retriever,
    InMemoryEmbeddingRetriever,
)
from haystack.components.joiners import DocumentJoiner
from haystack.components.rankers import SentenceTransformersSimilarityRanker
from haystack.core.component import component
from tqdm import tqdm
import asyncio
import torch

@component
class PromptableTextEmbedder(SentenceTransformersTextEmbedder):
    """
    A custom text embedder that allows passing a 'prompt_name'
    to the underlying sentence-transformer's encode method.

    This is useful for instruction-tuned models like Qwen3-Embedding
    that expect different prompts for queries vs. documents.
    """
    
    @component.output_types(embedding=List[float])
    def run(self, text: str, prompt_name: Optional[str] = None):
        """
        Embeds a single string of text.

        :param text: The text to embed.
        :param prompt_name: The name of the prompt to use for the embedding model,
                            e.g., "query" or "passage".
        :return: A dictionary containing the "embedding" of the text.
        """
        if not isinstance(text, str):
            # Fallback for old Haystack versions that send a list
            if isinstance(text, list) and len(text) == 1 and isinstance(text[0], str):
                text = text[0]
            else:
                 raise TypeError(
                    "PromptableTextEmbedder expects a single string as input. "
                    "If you want to embed multiple documents, please use the PromptableDocumentEmbedder."
                )

        encode_kwargs = dict(self.encode_kwargs or {})

        # Try to pass `prompt_name` directly if caller provided it.
        if prompt_name is not None:
            encode_kwargs["prompt_name"] = prompt_name

        # Determine prefix based on prompt_name.
        if prompt_name == "query":
            prefix = "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: "
        elif prompt_name:
            prefix = f"{prompt_name}: "
        else:
            prefix = ""

        text_to_embed = f"{prefix}{text}" if prefix else text

        # Ensure the backend is ready
        if self.embedding_backend is None:
            raise RuntimeError("The embedding model has not been loaded. Please call warm_up() before running.")

        embedding = self.embedding_backend.embed(
            [text_to_embed],
            batch_size=self.batch_size,
            show_progress_bar=self.progress_bar,
            normalize_embeddings=self.normalize_embeddings,
            precision=self.precision,
            **encode_kwargs,
        )[0]

        return {"embedding": embedding}

def run_retrieval_for_generated_questions(
    qa_pairs_json_path: str | Path,
    output_dir: str | Path,
    retrieval_pipeline: Pipeline,
    output_file_name: str = "generated_questions_retrieval_results.json"
    ):
    """
    Runs retrieval for generated question-answer pairs and stores the ranked
    documents for each question.
    """

    qa_pairs_json_path = Path(qa_pairs_json_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load QA pairs
    with open(qa_pairs_json_path, "r", encoding="utf-8") as f:
        qa_pairs_raw = json.load(f)

    # Validate structure (basic)
    qa_pairs: list[dict] = []
    for i, pair in enumerate(qa_pairs_raw):
        qa_pairs.append(pair)

    if not qa_pairs:
        print("No QA pairs found for retrieval.")
        return

    print(f"Running retrieval for {len(qa_pairs)} generated QA pairs.")

    retrieval_outputs: list[dict] = []
    for pair in tqdm(qa_pairs):
        question_text = pair.get("question")
        id = pair.get("id")
        if not question_text:
            print(f"Warning: missing 'question' in pair index {id}, skipping.")
            continue

        pipeline_result = retrieval_pipeline.run({
            "text_embedder": {"text": question_text, "prompt_name": "query"},
            "bm25_retriever": {"query": question_text},
            "ranker": {"query": question_text}
        })

        documents = pipeline_result.get("ranker", {}).get("documents", [])

        docs_serialised: list[dict] = []
        for doc in documents:
            docs_serialised.append({
                "content": doc.content,
                "meta": doc.meta,
                "score": float(doc.score),
            })

        retrieval_outputs.append({
            "query_id": id,
            "query_text": question_text,
            "reference_answer": pair.get("answer"),
            "reference_answer_doc_name": pair.get("doc_name"),
            "results": docs_serialised,
        })

    save_path = output_dir / output_file_name
    with open(save_path, "w", encoding="utf-8") as fp:
        json.dump(retrieval_outputs, fp, indent=4)

    print(f"Saved retrieval results to {save_path} with {len(retrieval_outputs)} entries.")

def run_retrieval_for_real_questions(
    queries_json_path: str | Path,
    output_dir: str | Path,
    retrieval_pipeline: Pipeline,
    output_file_name: str = "real_questions_retrieval_results.json",
    skip_non_relevant: bool = False
    ):
    """
    Runs retrieval for a list of queries and saves the results to a JSON file.
    """

    queries_json_path = Path(queries_json_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load and filter queries
    with open(queries_json_path, "r", encoding="utf-8") as f:
        raw_queries = json.load(f)

    if skip_non_relevant:
        queries: list[dict] = [q for q in raw_queries if q.get("relevant")]
    else:
        queries = raw_queries

    if skip_non_relevant and not queries:
        print("No relevant queries found.")
        return

    print(f"Running retrieval for {len(queries)} relevant queries.")

    # Execute retrieval for one query at a time
    retrieval_outputs: list[dict] = []
    for q in tqdm(queries):
        query_text = q["query_text"]
        query_id = q["query_id"]

        pipeline_result = retrieval_pipeline.run({
            "text_embedder": {"text": query_text, "prompt_name": "query"},
            "bm25_retriever": {"query": query_text},
            "ranker": {"query": query_text}
        })

        documents = pipeline_result.get("ranker", {}).get("documents", [])

        docs_serialised: list[dict] = []
        for doc in documents:
            docs_serialised.append({
                "content": doc.content,
                "meta": doc.meta,
                "score": float(doc.score),
            })

        retrieval_outputs.append({
            "query_id": query_id,
            "query_text": query_text,
            "results": docs_serialised,
        })

    # Save results
    save_path = output_dir / output_file_name
    with open(save_path, "w", encoding="utf-8") as fp:
        json.dump(retrieval_outputs, fp, indent=4)

    print(f"Saved retrieval results to {save_path} with {len(retrieval_outputs)} entries.")

def create_retrieval_pipeline(
    document_store_path: str | Path,
    embedding_model: str = "Qwen/Qwen3-Embedding-0.6B",
    embedder_config_kwargs: dict | None = {"attn_implementation": "flash_attention_2"},
    embedder_model_kwargs: dict | None = {"torch_dtype": torch.bfloat16},
    embedder_batch_size: int = 32,
    reranker_model: str = "Snowflake/snowflake-arctic-embed-l-v2.0",
    reranker_batch_size: int = 32,
    top_k_semantic_search: int = 50,
    top_k_keyword_search: int = 50,
    top_k_reranker: int = 10,
    device: str = "cpu",
    ) -> Pipeline:
    """
    Creates a retrieval pipeline by loading an `InMemoryDocumentStore` from the
    JSON file persisted with `save_to_disk()`.
    """

    document_store_path = Path(document_store_path)
    if not document_store_path.exists():
        raise FileNotFoundError(f"Document store file not found: {document_store_path}")

    try:
        document_store = InMemoryDocumentStore.load_from_disk(str(document_store_path))
        print(f"Loaded document store with {document_store.count_documents()} documents from {document_store_path}")
    except Exception as e:
        raise RuntimeError(f"Failed to load document store: {e}")

    text_embedder = PromptableTextEmbedder(
        model=embedding_model,
        device=ComponentDevice.from_str(device),
        progress_bar=False,
        config_kwargs=embedder_config_kwargs,
        model_kwargs=embedder_model_kwargs,
        batch_size=embedder_batch_size
    )

    embedding_retriever = InMemoryEmbeddingRetriever(document_store=document_store, top_k=top_k_semantic_search)

    bm25_retriever = InMemoryBM25Retriever(document_store=document_store, top_k=top_k_keyword_search)

    document_joiner = DocumentJoiner()  # join_mode="concatenate"

    ranker = SentenceTransformersSimilarityRanker(
        model=reranker_model,
        device=ComponentDevice.from_str(device),
        top_k=top_k_reranker,
        batch_size=reranker_batch_size
    )

    text_embedder.warm_up()
    ranker.warm_up()

    hybrid_retrieval = Pipeline()
    hybrid_retrieval.add_component("text_embedder", text_embedder)
    hybrid_retrieval.add_component("embedding_retriever", embedding_retriever)
    hybrid_retrieval.add_component("bm25_retriever", bm25_retriever)
    hybrid_retrieval.add_component("document_joiner", document_joiner)
    hybrid_retrieval.add_component("ranker", ranker)

    hybrid_retrieval.connect(
        "text_embedder.embedding", "embedding_retriever.query_embedding"
    )
    hybrid_retrieval.connect("bm25_retriever.documents", "document_joiner.documents")
    hybrid_retrieval.connect("embedding_retriever.documents", "document_joiner.documents")
    hybrid_retrieval.connect("document_joiner.documents", "ranker.documents")

    return hybrid_retrieval

def load_json_chunks_from_dir(documents_dir: str | Path) -> List[Document]:
    """
    Reads JSON files from a directory and converts them into a list of Haystack Documents.
    """
    documents_dir = Path(documents_dir)
    docs_to_index = []
    
    files = list(documents_dir.glob("*.json"))
    print(f"Found {len(files)} files to process...")

    for filename in files:
        with open(filename, 'r', encoding='utf-8') as f:
            doc_data = json.load(f)
            
        for i, chunk in enumerate(doc_data["chunks"]):
            document = Document(
                content=chunk["chunk_text"],
                meta={
                    'chunk_id': i,
                    'doc_name': doc_data["doc_name"],
                    'chunking_method': doc_data["method"],
                    'chunk_pages': chunk["chunk_pages"],
                }
            )
            docs_to_index.append(document)
            
    print(f"Created {len(docs_to_index)} Haystack Document objects.")
    return docs_to_index

def index_documents(
    documents: List[Document],
    output_dir: str | Path,
    embedding_model_name: str = "Qwen/Qwen3-Embedding-0.6B",
    embedder_config_kwargs: dict | None = {"attn_implementation": "flash_attention_2"},
    embedder_model_kwargs: dict | None = {"torch_dtype": torch.bfloat16},
    device: str = "cuda:0",
    batch_size: int = 32,
    progress_bar: bool = True,
    ):
    """
    Builds and runs an indexing pipeline to embed and store documents.
    """
    # Prepare output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    document_store = InMemoryDocumentStore()

    # Components
    document_embedder = SentenceTransformersDocumentEmbedder(
        model=embedding_model_name,
        device=ComponentDevice.from_str(device),
        batch_size=batch_size,
        config_kwargs=embedder_config_kwargs,
        model_kwargs=embedder_model_kwargs,
        progress_bar=progress_bar,
    )

    # Warm up model (loads it into memory and ensures faster first pass)
    document_embedder.warm_up()

    # document_store: FAISSDocumentStore = FAISSDocumentStore(
    #     embedding_dim=embedding_dim or 768,
    #     faiss_index_factory_str=faiss_index_factory,
    #     sql_url="sqlite:///:memory:",
    # )

    if not documents:
        print("No documents provided to index.")
        return document_store

    document_writer = DocumentWriter(document_store=document_store)

    # Build the pipeline
    indexing_pipeline = Pipeline()
    indexing_pipeline.add_component("embedder", document_embedder)
    indexing_pipeline.add_component("writer", document_writer)
    indexing_pipeline.connect("embedder.documents", "writer.documents")

    # Run the pipeline
    indexing_pipeline.run({"embedder": {"documents": documents}})

    print("Indexing complete!")

    # Persist the entire in-memory store to disk using Haystack's helper
    store_path = output_dir / "document_store.json"
    try:
        document_store.save_to_disk(str(store_path))
        print(f"Saved document store with {document_store.count_documents()} documents to {store_path}")
    except Exception as e:
        print(f"Warning: Could not save document store: {e}")

def read_queries_csv(file_path: str | Path) -> list[str]:
    file_path = Path(file_path)
    df = pd.read_csv(file_path, sep=";")
    queries = df["English Question"]
    return queries.tolist()

class QuestionAnswerPair(BaseModel):
    question_text: str = Field(
        ..., description="The generated question text."
    )
    answer_text: str = Field(
        ...,
        description="The answer text for the generated question.",
    )

class QuestionAnswerPairList(BaseModel):
    question_answer_pairs: list[QuestionAnswerPair] = Field(
        ...,
        description="A list of question-answer pairs.",
    )

async def generate_qa_pairs(
    parsed_docs_dir: str | Path,
    outputs_dir: str | Path,
    client_completion_func: Awaitable[str],
    qa_generation_prompt: str,
    qa_pairs_per_document: int = 3,
    output_filename: str = "generated_qa_pairs.json",
    max_context_tokens: int = 5000,
    count_tokens_func: Callable[[str], int] = count_tokens):

    parsed_docs_dir = Path(parsed_docs_dir)
    outputs_dir = Path(outputs_dir)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    # Build prompts
    doc_names: list[str] = []
    prompts: list[str] = []

    for doc_file in parsed_docs_dir.glob("*.json"):
        with open(doc_file, "r", encoding="utf-8") as f:
            doc_data = json.load(f)

        doc_name = doc_file.with_suffix("").name.split("_", 1)[-1]
        text = doc_data["full_text"]

        lines = text.splitlines(keepends=True)
        document_context = combine_blocks(
            lines,
            max_tokens=max_context_tokens,
            count_tokens_func=count_tokens_func,
        )
        prompt = qa_generation_prompt.format(document_context=document_context.strip(), qa_pairs_per_document=qa_pairs_per_document)

        doc_names.append(doc_name)
        prompts.append(prompt)
    
    print("Generating question answer pairs for documents...")
    tasks = [client_completion_func(p) for p in prompts]
    responses_list = await asyncio.gather(*tasks) if tasks else []

    outputs: list[dict] = []
    qa_counter = 1  # global sequential ID for all generated QA pairs

    for doc_name, response in zip(doc_names, responses_list):
        qa_items: QuestionAnswerPairList = response.output_parsed 
        for pair in qa_items.question_answer_pairs:
            outputs.append(
                {
                    "id": qa_counter,
                    "doc_name": doc_name,
                    "question": pair.question_text,
                    "answer": pair.answer_text,
                }
            )
            qa_counter += 1

    out_path = outputs_dir / output_filename
    try:
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(outputs, f, indent=4)
        print(f"Saved generated question answer pairs for {len(outputs)} documents to {out_path}")
    except IOError as e:
        print(f"Warning: could not write generated question answer pairs file '{out_path}': {e}")

async def generate_doc_summaries(
    parsed_docs_dir: str | Path,
    outputs_dir: str | Path,
    client_completion_func: Awaitable[str],
    summarization_prompt: str,
    max_context_tokens: int = 5000,
    count_tokens_func: Callable[[str], int] = count_tokens):
    """Generate summaries for documents concurrently.

    *client_completion_func* can be either synchronous or asynchronous.
    """

    parsed_docs_dir = Path(parsed_docs_dir)
    outputs_dir = Path(outputs_dir)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    # Load documents and build prompts
    doc_names: list[str] = []
    prompts: list[str] = []

    for doc_file in parsed_docs_dir.glob("*.json"):
        with open(doc_file, "r", encoding="utf-8") as f:
            doc_data = json.load(f)

        doc_name = doc_file.with_suffix("").name.split("_", 1)[-1]
        text = doc_data["full_text"]

        lines = text.splitlines(keepends=True)
        document_context = combine_blocks(
            lines,
            max_tokens=max_context_tokens,
            count_tokens_func=count_tokens_func,
        )
        prompt = summarization_prompt.format(document_context=document_context)

        doc_names.append(doc_name)
        prompts.append(prompt)

    # Run client completion function async for each prompt
    tasks = [client_completion_func(p) for p in prompts]
    summaries = await asyncio.gather(*tasks)

    # Attach summaries to output files
    for doc_name, summary in zip(doc_names, summaries):
        print("Generating summary for document:", doc_name)
        out_path = outputs_dir / f"{doc_name}.json"
        try:
            with out_path.open("w", encoding="utf-8") as f:
                json.dump({"doc_name": doc_name, "summary": summary}, f, indent=4)
        except IOError as e:
            print(f"Warning: could not write summary for '{doc_name}': {e}")

class QueryEvaluation(BaseModel):
    query_id: int = Field(
        ..., description="The unique identifier for the user's query."
    )
    relevant: bool = Field(
        ...,
        description="Set to true if one or more document summaries suggest the full document contains the relevant information. Otherwise, set to false.",
    )
    justification: str = Field(
        ..., description="A brief paragraph justifying your decision."
    )

class EvaluationResult(BaseModel):
    evaluations: List[QueryEvaluation]

def get_filtering_results(response: EvaluationResult):
    """
    Get the filtering results from the response and returns them as a list of dict.
    """

    group_responses = []
    for query_filtering_result in response.evaluations:
        group_responses.append(
            {
                "query_id": query_filtering_result.query_id,
                "relevant": query_filtering_result.relevant,
                "justification": query_filtering_result.justification
            }
        )
    return group_responses

def group_queries(
    queries: list[str],
    queries_per_group: int = 5,
    ):
    """
    Group queries into groups of size queries_per_group.
    """

    queries_groups = [queries[i:i+queries_per_group] for i in range(0, len(queries), queries_per_group)]
    return queries_groups

async def filter_relevant_queries(
    queries_csv_path: str | Path,
    doc_summaries_dir: str | Path,
    outputs_dir: str | Path,
    client_completion_func: Awaitable[str],
    filtering_prompt: str
    ):
    """
    Filter queries based on a prompt so that we only keep the queries that are relevant to the documents.
    """

    queries_csv_path = Path(queries_csv_path)
    doc_summaries_dir = Path(doc_summaries_dir)
    outputs_dir = Path(outputs_dir)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    queries = read_queries_csv(queries_csv_path)
    doc_summaries = {}
    for doc_summary_file in doc_summaries_dir.glob("*.json"):
        with open(doc_summary_file, "r") as f:
            doc_data = json.load(f)
            doc_name = doc_data["doc_name"]
            doc_summaries[doc_name] = doc_data["summary"]
    
    doc_summaries_str = ""
    for i, (doc_name, doc_summary) in enumerate(doc_summaries.items()):
        doc_summaries_str += f"Document ID: {i}\nDocument name: {doc_name}\nSummary: {doc_summary}\n\n"

    grouped_queries = group_queries(queries, queries_per_group=5)

    # Build prompts and maintain mapping info
    prompts: list[str] = []
    group_query_id_maps: list[list[int]] = []  # Within each prompt, list of query_ids
    query_id_to_text: list[str] = []
    query_id_counter = 0

    for group in grouped_queries:
        queries_str = ""
        ids_in_group: list[int] = []
        for query in group:
            queries_str += f"Query ID: {query_id_counter}\nQuery: {query}\n\n"
            ids_in_group.append(query_id_counter)
            query_id_to_text.append(query)
            query_id_counter += 1

        prompt = filtering_prompt.format(
            doc_summaries_str=doc_summaries_str, queries_str=queries_str
        )
        prompts.append(prompt)
        group_query_id_maps.append(ids_in_group)

    # Launch async model calls
    tasks = [client_completion_func(p) for p in prompts]
    responses = await asyncio.gather(*tasks) if tasks else []

    outputs: list[dict] = []

    for resp, ids_in_group in zip(responses, group_query_id_maps):
        # We assume resp has attribute output_parsed
        result = get_filtering_results(resp.output_parsed)  # type: ignore[attr-defined]

        for q_out in result:
            qid = q_out["query_id"]
            q_out["query_text"] = query_id_to_text[qid]
        outputs.extend(result)

    save_path = outputs_dir / "filtered_queries.json"
    with open(save_path, "w", encoding="utf-8") as fp:
        json.dump(outputs, fp, indent=4)

def combine_queries(
    generated_queries_dir: str | Path,
    filtered_queries_dir: str | Path,
    output_dir: str | Path,
    filtered_filename: str = "filtered_queries.json",
    generated_filename: str = "generated_queries.json",
    output_filename: str = "all_queries.json"):
    """Combine generated and filtered (real) queries into a single JSON file.

    Each record in the resulting JSON list contains every key from the filtered
    queries plus an extra boolean field *generated* indicating whether the
    query was produced automatically (``True``) or originates from the original
    dataset (``False``).
    """

    generated_queries_dir = Path(generated_queries_dir)
    filtered_queries_dir = Path(filtered_queries_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    filtered_path = filtered_queries_dir / filtered_filename
    if filtered_path.exists():
        with filtered_path.open("r", encoding="utf-8") as fp:
            filtered_queries: list[dict] = json.load(fp)
    else:
        print(f"Warning: filtered queries file not found: {filtered_path}. Proceeding with empty list.")
        filtered_queries = []

    combined: list[dict] = []

    # Add real queries and tag them
    for q in filtered_queries:
        q_copy = q.copy()
        q_copy["generated"] = False
        combined.append(q_copy)

    # Determine next query_id to ensure uniqueness across combined list
    existing_ids = {q.get("query_id") for q in combined if q.get("query_id") is not None}
    next_id = (max(existing_ids) + 1) if existing_ids else 0

    generated_path = generated_queries_dir / generated_filename

    if generated_path.exists():
        with generated_path.open("r", encoding="utf-8") as fp:
            generated_data = json.load(fp)

        if isinstance(generated_data, list):
            for item in generated_data:
                queries = item.get("queries", [])
                # Handle str vs list
                if isinstance(queries, str):
                    queries = [q.strip() for q in queries.splitlines() if q.strip()]
                if not isinstance(queries, list):
                    print(f"Warning: unexpected 'queries' format in consolidated file. Skipping entry {item}.")
                    continue
                for q_text in queries:
                    combined.append({
                        "query_id": next_id,
                        "query_text": q_text,
                        "relevant": True,
                        "justification": f"Generated from document {item.get('doc_name', 'unknown')}",
                        "generated": True,
                    })
                    next_id += 1

    output_path = output_dir / output_filename
    with output_path.open("w", encoding="utf-8") as fp:
        json.dump(combined, fp, indent=4)

    print(f"Saved {len(combined)} queries (real + generated) to {output_path}")