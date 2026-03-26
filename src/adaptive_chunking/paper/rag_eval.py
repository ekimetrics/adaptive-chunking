from deepeval import evaluate
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualRelevancyMetric,
    GEval,
    BaseMetric
)
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.models import GPTModel
from typing import Awaitable
from pathlib import Path
import json
from ..chunking_utils import count_tokens
import asyncio
from statistics import mean, stdev
from tabulate import tabulate
from pydantic import BaseModel, Field

def check_generation_stats(
    generation_outputs_path: Path|str,
    skipped_answer_string: str = "I don't know based on the provided context."):
    
    generation_outputs_path = Path(generation_outputs_path)
    with open(generation_outputs_path, "r") as f:
        gen_outputs = json.load(f)

    answered_count = len(gen_outputs)
    for output in gen_outputs:
        if skipped_answer_string in output["generated_output"]:
            answered_count -= 1
            
    print(f"Number of answered queries: {answered_count}")
    print(f"Total number of queries: {len(gen_outputs)}")
    print(f"Percentage of answered queries: {int(answered_count / len(gen_outputs) * 100)}%")

class RetrievalCompletenessMetric(BaseMetric):
    """Custom metric evaluating whether reference answer is fully supported by the retrieved context."""

    RETRIEVAL_COMPLETENESS_PROMPT = """<Task>
    You are an expert fact-checker. Your task is to evaluate how completely the provided Context supports a Reference Answer.
    </Task>

    <Instructions>
    1.  Read the Context thoroughly.
    2.  Read the Reference Answer carefully.
    3.  Compare the Reference Answer against the Context.
    4.  Classify the completeness level:
        - 0 = Incomplete: Context does not support key claims.
        - 1 = Partially Complete: Context supports some but not all claims.
        - 2 = Complete: Context fully supports all claims.
    5.  Provide a brief one-sentence reason.
    </Instructions>

    <Context>
    {context}
    </Context>

    <Reference Answer>
    {reference_answer}
    </Reference Answer>
    """

    class RetrievalCompletenessVerdict(BaseModel):
        completeness_level: int = Field(
            ..., ge=0, le=2,
            description="0 = Incomplete, 1 = Partially Complete, 2 = Complete."
        )
        reason: str = Field(..., description="A brief, one-sentence explanation for your decision.")

    def __init__(self, model: GPTModel, threshold: float = 0.5):
        self.name = "RetrievalCompleteness"
        self.model = model
        self.threshold = threshold
        self.score: float | None = None
        self.success: bool | None = None
        self.reason: str | None = None
        self.evaluation_cost: float | None = None

        # Set evaluation_model for DeepEval progress messages
        self.evaluation_model = (
            model.get_model_name() if hasattr(model, "get_model_name") else str(model)
        )

    def _evaluate_completeness(self, test_case: LLMTestCase) -> dict:
        if not test_case.expected_output or not test_case.retrieval_context:
            return {"score": 0.0, "reason": "Missing expected_output or retrieval_context"}

        full_context = "\n\n".join(test_case.retrieval_context)
        prompt = self.RETRIEVAL_COMPLETENESS_PROMPT.format(
            context=full_context,
            reference_answer=test_case.expected_output,
        )

        try:
            gen_result = self.model.generate(
                prompt, schema=self.RetrievalCompletenessVerdict
            )

            # model.generate returns (parsed, cost) when schema is provided
            if isinstance(gen_result, tuple):
                parsed, cost = gen_result
                self.evaluation_cost = cost
            else:
                parsed = gen_result

            if isinstance(parsed, self.RetrievalCompletenessVerdict):
                level = parsed.completeness_level
                reason = parsed.reason
            elif isinstance(parsed, dict):
                level = int(parsed.get("completeness_level", 0))
                reason = parsed.get("reason", "No reason returned")
            else:
                raise ValueError("Unexpected parsed output type from model.generate")
        except Exception as e:
            level = 0
            reason = f"Model error: {e}"

        level = max(0, min(level, 2))
        score = level / 2  # map 0->0.0, 1->0.5, 2->1.0
        return {"score": score, "reason": reason, "level": level}

    def measure(self, test_case: LLMTestCase, *_args, **_kwargs) -> float:
        res = self._evaluate_completeness(test_case)
        self.score = res["score"]
        self.reason = res["reason"]
        self.evaluation_model = self.model.get_model_name() if hasattr(self.model, "get_model_name") else str(self.model)
        self.success = self.score >= self.threshold
        return self.score

    async def a_measure(self, test_case: LLMTestCase, **kwargs):
        return self.measure(test_case)

    # Provide custom name attribute so DeepEval reports correctly
    @property
    def __name__(self):
        return self.name

    # BaseMetric requirement
    def is_successful(self) -> bool:
        if self.success is None:
            raise RuntimeError("measure() must be called before is_successful().")
        return bool(self.success)

def evaluate_batches_deep_eval(
    cases: list[LLMTestCase],
    batch_size: int,
    metrics_cfg: list,
    item_by_query: dict[str, dict],
    detailed_results: list[dict],
    output_path: Path):
    """Evaluate LLMTestCase batches with provided DeepEval metrics and append results.

    This standalone helper avoids defining functions inside other functions (per style preference).
    It mutates *detailed_results* in place and saves progress to *output_path* after each batch.
    """

    if not cases:
        return

    for batch_start in range(0, len(cases), batch_size):
        batch_cases = cases[batch_start : batch_start + batch_size]
        print(
            f"Evaluating batch {batch_start // batch_size + 1} / {((len(cases) - 1) // batch_size) + 1} (size={len(batch_cases)}) …"
        )

        batch_result = None
        if metrics_cfg:
            batch_result = evaluate(batch_cases, metrics=metrics_cfg)

        # Build mapping from query_text to test_result
        result_by_query: dict[str, any] = {}
        if batch_result is not None:
            for tr in batch_result.test_results:
                q = (
                    getattr(tr, "input", None)
                    or getattr(getattr(tr, "test_case", None), "input", None)
                )
                if q is not None:
                    result_by_query[q] = tr

        for test_case in batch_cases:
            test_result = result_by_query.get(test_case.input) if metrics_cfg else None
            if test_result is None and metrics_cfg:
                print(f"Warning: no result returned for query '{test_case.input}'. Skipping.")
                continue

            original_item = item_by_query[test_case.input]

            case_result = {
                "query_id": original_item["query_id"],
                "query_text": test_case.input,
                "generated_output": test_case.actual_output,
                "reference_answer": test_case.expected_output,
                "reference_answer_doc_name": original_item["reference_answer_doc_name"],
                "context_data": original_item["context_data"],
                "metrics": [],
                "success": test_result.success if test_result else None,
            }

            # Append DeepEval metric results if any
            if test_result and test_result.metrics_data:
                for metric_data in test_result.metrics_data:
                    case_result["metrics"].append(
                        {
                            "name": metric_data.name,
                            "score": metric_data.score,
                            "threshold": metric_data.threshold,
                            "success": metric_data.success,
                            "reason": metric_data.reason,
                            "error": metric_data.error,
                        }
                    )

            detailed_results.append(case_result)

        # Save progress
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(detailed_results, f, indent=4)
        print(f"Saved progress to {output_path} (total {len(detailed_results)} queries).")

async def generate_answers(
    retrieval_results_path: str | Path,
    output_dir: str | Path,
    output_file_name: str,
    async_client_completion_func: Awaitable[str],
    qa_prompt: str,
    ):
    """
    Generates answers using the queries and context data from the retrieval results.
    """

    retrieval_results_path = Path(retrieval_results_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(retrieval_results_path, "r", encoding="utf-8") as f:
        retrieval_data = json.load(f)
    
    print(f"Loaded {len(retrieval_data)} queries. Generating answers:")

    prompting_data = []
    for item in retrieval_data:
        query_id = item["query_id"]
        query_text = item["query_text"]
        context_data = item["results"]

        context_str = ""
        for i, doc in enumerate(context_data):
            doc_name = doc['meta']['doc_name']
            context_str += f"<Document, ID = {i}, name = {doc_name}>\n"
            context_str += f"Content: {doc['content']}\n"
            context_str += f"</Document, ID = {i}, name = {doc_name}>\n\n"

        prompt = qa_prompt.format(context_str=context_str.strip(), question_str=query_text.strip())
        print(f"Question id {query_id}, prompt tokens:", count_tokens(prompt))
        
        prompting_data.append({
            "query_id": query_id,
            "query_text": query_text,
            "reference_answer": item["reference_answer"] if "reference_answer" in item else None,
            "reference_answer_doc_name": item["reference_answer_doc_name"] if "reference_answer_doc_name" in item else None,
            "prompt": prompt,
            "context_data": context_data,
        })

    tasks = [async_client_completion_func(item["prompt"]) for item in prompting_data]
    generated_outputs = await asyncio.gather(*tasks)

    answers: list[dict] = []
    for idx, generated_output in enumerate(generated_outputs):
        qa_meta = prompting_data[idx]

        answers.append({
            "query_id": qa_meta["query_id"],
            "query_text": qa_meta["query_text"],
            "reference_answer": qa_meta.get("reference_answer"),
            "reference_answer_doc_name": qa_meta.get("reference_answer_doc_name"),
            "generated_output": generated_output,
            "context_data": qa_meta["context_data"],
        })

    print(f"Generated {len(answers)} answers.")

    with open(output_dir / output_file_name, "w", encoding="utf-8") as f:
        json.dump(answers, f, indent=4)

def evaluate_rag_results_generated_questions(
    generation_results_path: str | Path,
    gpt_model: GPTModel,
    output_dir: str | Path,
    output_file_name: str = "generated_questions_evaluation_results.json",
    continue_processing: bool = True,
    batch_size: int = 10,
    skip_correctness_with: str = "",
    ):
    """
    Evaluates generated answers for *generated questions*.

    Uses the following DeepEval metrics:
        1. ContextualRecallMetric - evaluates the adequacy of retrieved context.
        2. AnswerCorrectnessMetric - evaluates the factual correctness of the generated answer against a reference answer.

    Processing is done in batches to reduce the risk of long-running requests timing out. Results are appended to (or resumed from) the
    provided *output_file_name* inside *output_dir*.
    """

    generation_results_path = Path(generation_results_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / output_file_name

    # Load generation results
    with open(generation_results_path, "r", encoding="utf-8") as f:
        generation_data = json.load(f)

    # map query_text to generation_data item
    item_by_query: dict[str, dict] = {item["query_text"]: item for item in generation_data}

    print(f"Loaded {len(generation_data)} generated answers.")

    # Resume evaluation (if requested)
    detailed_results: list[dict]
    processed_queries: set[str] = set()

    if continue_processing and output_path.exists():
        with open(output_path, "r", encoding="utf-8") as f:
            detailed_results = json.load(f)
        processed_queries = {item["query_text"] for item in detailed_results}
        print(f"Resuming evaluation, {len(processed_queries)} queries already processed.")
    else:
        detailed_results = []

    # Define metrics configurations
    correctness_metric = GEval(
        name="Correctness",
        model=gpt_model,
        evaluation_params=[
            LLMTestCaseParams.EXPECTED_OUTPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
        ],
        evaluation_steps=[
            "Check whether the facts in 'actual output' contradict any facts in 'expected output'",
            "Lightly penalize omissions of detail, focusing on the main idea",
            "Vague language or contradicting opinions are permissible",
        ],
    )

    retrieval_comp_metric = RetrievalCompletenessMetric(model=gpt_model)

    full_metrics = [retrieval_comp_metric, correctness_metric]
    recall_only_metrics = [retrieval_comp_metric]

    # Build new test cases
    full_cases: list[LLMTestCase] = []
    recall_only_cases: list[LLMTestCase] = []
    remaining_items_full: list[str] = []  # track query_text order
    remaining_items_recall_only: list[str] = []

    for item in generation_data:
        query_text: str = item["query_text"]
        query_id: int = item["query_id"]

        if query_text in processed_queries:
            continue

        actual_output = item.get("generated_output")
        reference_answer = item.get("reference_answer")
        context_data = item["context_data"]

        # Decide which metrics to use based on conditions
        skip_correctness = (
            (skip_correctness_with and actual_output and skip_correctness_with in actual_output)
            or not reference_answer
        )

        test_case = LLMTestCase(
            input=query_text,
            actual_output=actual_output,
            expected_output=reference_answer,
            retrieval_context=[doc["content"] for doc in context_data],
        )

        if skip_correctness:
            recall_only_cases.append(test_case)
            remaining_items_recall_only.append(query_text)
        else:
            full_cases.append(test_case)
            remaining_items_full.append(query_text)

    total_remaining = len(full_cases) + len(recall_only_cases)
    print(f"{total_remaining} queries remain to be evaluated.")

    if total_remaining == 0:
        # Save skipped cases (if any) before exiting
        if detailed_results:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(detailed_results, f, indent=4)
            print(f"Saved skipped queries to {output_path} (total {len(detailed_results)} queries).")
        print("Nothing to do, all queries are already evaluated.")
        return

    # Evaluate recall-only cases first then full cases using the helper
    if recall_only_cases:
        print(f"Evaluating {len(recall_only_cases)} cases with retrieval completeness metric only …")
        evaluate_batches_deep_eval(
            recall_only_cases,
            batch_size,
            recall_only_metrics,
            item_by_query,
            detailed_results,
            output_path,
        )

    if full_cases:
        print(f"Evaluating {len(full_cases)} cases with both metrics …")
        evaluate_batches_deep_eval(
            full_cases,
            batch_size,
            full_metrics,
            item_by_query,
            detailed_results,
            output_path,
        )

    print("Evaluation finished for all remaining queries (generated questions).")
  
def evaluate_rag_results_real_questions(
    generation_results_path: str | Path,
    gpt_model: GPTModel,
    output_dir: str | Path,
    output_file_name: str = "real_questions_evaluation_results.json",
    continue_processing: bool = True,
    batch_size: int = 10,
    skip_answers_with: str = "",
    ):
    """
    Evaluates generated answers using DeepEval metrics in manageable batches to
    avoid long-running timeouts.
    """

    generation_results_path = Path(generation_results_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / output_file_name

    # Load generation results
    with open(generation_results_path, "r", encoding="utf-8") as f:
        generation_data = json.load(f)

    print(f"Loaded {len(generation_data)} generated answers.")

    # Resume evaluation
    detailed_results: list[dict]
    processed_queries: set[str] = set()

    if continue_processing and output_path.exists():
        with open(output_path, "r", encoding="utf-8") as f:
            detailed_results = json.load(f)
        processed_queries = {item["query_text"] for item in detailed_results}
        print(f"Resuming evaluation, {len(processed_queries)} queries already processed.")
    else:
        detailed_results = []

    # Prepare metrics (instantiated early so that we can reference metric metadata for skipped answers)
    metrics = [
        AnswerRelevancyMetric(model=gpt_model),
        FaithfulnessMetric(model=gpt_model),
        ContextualRelevancyMetric(model=gpt_model),
    ]

    # Build new test cases
    new_test_cases: list[LLMTestCase] = []
    remaining_items: list[dict] = []
    for item in generation_data:
        query_text = item["query_text"]
        if query_text in processed_queries:
            continue
        actual_output = item.get("generated_output")
        context_data = item["context_data"]

        # Skip evaluation for answers that contain the specified string
        if skip_answers_with and (actual_output and skip_answers_with in actual_output):
            skipped_case_result = {
                "query_text": query_text,
                "generated_output": actual_output,
                "retrieval_context": [doc["content"] for doc in context_data],
                "metrics": [
                    {
                        "name": getattr(m, "__name__", m.__class__.__name__),
                        "score": None,
                        "threshold": m.threshold,
                        "success": None,
                        "reason": "Skipped due to skip_answers_with filter",
                        "error": None,
                    }
                    for m in metrics
                ],
                "success": None,
            }
            detailed_results.append(skipped_case_result)
            continue

        test_case = LLMTestCase(
            input=query_text,
            actual_output=actual_output,
            retrieval_context=[doc["content"] for doc in context_data],
        )
        new_test_cases.append(test_case)
        remaining_items.append(item)

    print(f"{len(new_test_cases)} queries remain to be evaluated.")
    if not new_test_cases:
        # Save skipped cases (if any) before exiting
        if detailed_results:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(detailed_results, f, indent=4)
            print(f"Saved skipped queries to {output_path} (total {len(detailed_results)} queries).")
        print("Nothing to do, all queries are already evaluated.")
        return

    # Evaluate in batches
    for batch_start in range(0, len(new_test_cases), batch_size):
        batch_cases = new_test_cases[batch_start : batch_start + batch_size]
        print(f"Evaluating batch {batch_start // batch_size + 1} / {((len(new_test_cases)-1)//batch_size)+1} (size={len(batch_cases)}) …")
        batch_result = evaluate(batch_cases, metrics=metrics)

        # Convert batch results to detailed results structure
        for tc, tr in zip(batch_cases, batch_result.test_results):
            case_result = {
                "query_text": tc.input,
                "generated_output": tc.actual_output,
                "retrieval_context": tc.retrieval_context,
                "metrics": [],
                "success": tr.success,
            }
            if tr.metrics_data:
                for md in tr.metrics_data:
                    case_result["metrics"].append(
                        {
                            "name": md.name,
                            "score": md.score,
                            "threshold": md.threshold,
                            "success": md.success,
                            "reason": md.reason,
                            "error": md.error,
                        }
                    )
            detailed_results.append(case_result)

        # Save outputs after each batch
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(detailed_results, f, indent=4)
        print(f"Saved progress to {output_path} (total {len(detailed_results)} queries).")

    print("Evaluation finished for all remaining queries.")

def show_rag_results_union_only_real_questions(evaluation_results_paths: dict[str, str | Path], evaluation_llm: str = "gpt-4.1"):
    evaluation_results_paths = {k: Path(v) for k, v in evaluation_results_paths.items()}
    evaluation_results = {k: json.load(v.open("r", encoding="utf-8")) for k, v in evaluation_results_paths.items()}

    system_names: list[str] = list(evaluation_results.keys())

    # Track original total queries per system
    total_queries_per_system: dict[str, int] = {sys: len(cases) for sys, cases in evaluation_results.items()}

    # Build mapping system -> query_text -> case
    results_by_query: dict[str, dict[str, dict]] = {
        sys: {case["query_text"]: case for case in cases} for sys, cases in evaluation_results.items()
    }

    # Union of all queries across systems
    all_queries: set[str] = set().union(*[d.keys() for d in results_by_query.values()])

    # Determine which queries have at least one system with a numeric metric score
    included_queries: list[str] = []
    for q in all_queries:
        query_has_answer = False
        for sys in system_names:
            case = results_by_query.get(sys, {}).get(q)
            if case and any(m.get("score") is not None for m in case.get("metrics", [])):
                query_has_answer = True
                break
        if query_has_answer:
            included_queries.append(q)

    if not included_queries:
        print("No queries with answers were found across the provided systems.")
        return

    # Metric names to consider (order fixed)
    METRIC_NAMES = [
        "Answer Relevancy",
        "Contextual Relevancy",
        "Faithfulness",
    ]

    # Containers for metric scores and counts
    metrics_scores: dict[str, dict[str, list[float]]] = {sys: {m: [] for m in METRIC_NAMES} for sys in system_names}
    answered_counts: dict[str, int] = {sys: 0 for sys in system_names}

    # Populate scores according to metric-specific rules
    for q in included_queries:
        # Check if any system has a numeric score for Contextual Relevancy in this query
        contextual_has_value = any(
            (
                next(
                    (m for m in results_by_query.get(sys, {}).get(q, {}).get("metrics", []) if m.get("name", "").lower() == "contextual relevancy"),
                    {},
                ).get("score")
                is not None
            )
            for sys in system_names
        )

        for sys in system_names:
            case = results_by_query.get(sys, {}).get(q)
            system_answered = False
            for metric_name in METRIC_NAMES:
                # Default: do not append any value (ignored) unless conditions below are met
                score_to_append: list[float] | None = None
                if case:
                    metric_dict = next(
                        (m for m in case.get("metrics", []) if m.get("name", "").lower() == metric_name.lower()),
                        None,
                    )
                    metric_score = metric_dict.get("score") if metric_dict else None

                    if metric_score is not None:
                        score_to_append = metric_score  # always take numeric values
                        system_answered = True
                    else:
                        # Handle None scores depending on metric
                        if metric_name == "Contextual Relevancy" and contextual_has_value:
                            score_to_append = 0.0  # map to zero only when someone provided a value
                        # For Answer Relevancy and Faithfulness we ignore None (no append)

                # Append value if we decided to include it
                if score_to_append is not None:
                    metrics_scores[sys][metric_name].append(score_to_append)

            if system_answered:
                answered_counts[sys] += 1

    # Build results table
    header_row = ["Metric"] + system_names
    table_rows: list[list[str]] = []

    for metric_name in METRIC_NAMES:
        row = [metric_name]
        for sys in system_names:
            scores = metrics_scores[sys][metric_name]
            if not scores:
                row.append("-")
            else:
                m = mean(scores) * 100
                s = stdev(scores) * 100 if len(scores) > 1 else 0.0
                row.append(f"{m:.2f}% ± {s:.2f}%")
        table_rows.append(row)

    # Final Score row
    final_score_row = ["Final Score"]
    for sys in system_names:
        all_scores = [score for metric_scores in metrics_scores[sys].values() for score in metric_scores]
        if all_scores:
            final_score_row.append(f"{mean(all_scores)*100:.2f}%")
        else:
            final_score_row.append("-")
    table_rows.append(final_score_row)

    # Evaluated Queries row
    evaluated_row = ["Evaluated Queries"] + [str(len(metrics_scores[sys][METRIC_NAMES[0]])) for sys in system_names]
    table_rows.append(evaluated_row)

    # Total Queries row
    total_row = ["Total Queries"] + [str(total_queries_per_system[sys]) for sys in system_names]
    table_rows.append(total_row)

    print(f"RAG evaluation scores (mean ± std): evaluated with deepeval + {evaluation_llm}")
    print("Queries answered by at least one system are considered. For Answer Relevancy and Faithfulness, queries with None scores are ignored in aggregation. Contextual Relevancy None scores are treated as 0 only if at least one system produced a numeric value for that query.\n")
    print(tabulate(table_rows, headers=header_row, tablefmt="simple"))

def show_rag_results_union_only_skip_nones_real_questions(evaluation_results_paths: dict[str, str | Path], evaluation_llm: str = "gpt-4.1"):
    evaluation_results_paths = {k: Path(v) for k, v in evaluation_results_paths.items()}
    evaluation_results = {k: json.load(v.open("r", encoding="utf-8")) for k, v in evaluation_results_paths.items()}

    # compute the average and std of the evaluation results for each metric
    metrics_scores: dict[str, dict[str, list[float]]] = {sys: {} for sys in evaluation_results.keys()}

    for system_name, results in evaluation_results.items():
        for case in results:
            for metric in case.get("metrics", []):
                metric_name = metric["name"]
                score = metric.get("score")
                if score is None:
                    # Skip metrics without numeric score (skipped or errored)
                    continue
                metrics_scores[system_name].setdefault(metric_name, []).append(score)

    # Gather the union of all metric names across systems
    all_metric_names: list[str] = sorted({name for sys in metrics_scores.values() for name in sys.keys()})
    system_names: list[str] = list(evaluation_results.keys())

    # Build table rows (metric, system1, system2, ...)
    header_row = ["Metric"] + system_names
    table_rows: list[list[str]] = []

    for metric_name in all_metric_names:
        row: list[str] = [metric_name]
        for system_name in system_names:
            scores = metrics_scores[system_name].get(metric_name, [])
            if not scores:
                row.append("-")
            else:
                m = mean(scores) * 100
                s = stdev(scores) * 100 if len(scores) > 1 else 0.0
                row.append(f"{m:.2f}% ± {s:.2f}%")
        table_rows.append(row)

    # Final score row
    final_score_row = ["Final Score"]
    for system_name in system_names:
        all_scores: list[float] = [score for scores in metrics_scores[system_name].values() for score in scores]
        if all_scores:
            overall = mean(all_scores) * 100
            final_score_row.append(f"{overall:.2f}%")
        else:
            final_score_row.append("-")
    table_rows.append(final_score_row)

    # Answered queries row (at least one metric has numeric score)
    answered_queries_row = ["Evaluated Queries"]
    for system_name in system_names:
        evaluated = sum(
            1
            for case in evaluation_results[system_name]
            if any(m.get("score") is not None for m in case.get("metrics", []))
        )
        answered_queries_row.append(str(evaluated))
    table_rows.append(answered_queries_row)

    # Total queries row
    total_queries_row = ["Total Queries"]
    for system_name in system_names:
        total_queries_row.append(str(len(evaluation_results[system_name])))
    table_rows.append(total_queries_row)

    # Pretty print using tabulate
    print(f"RAG evaluation scores (mean ± std): evaluated with deepeval + {evaluation_llm}")
    print("Only 'union' queries that were evaluated by at least one system are considered.")
    print("Query scores that received a 'none' value in one of the metrics were ignored to the mean computation.\n")
    print(tabulate(table_rows, headers=header_row, tablefmt="simple"))

def show_rag_results_generated_questions(
    evaluation_results_paths: dict[str, str | Path],
    evaluation_llm: str = "gpt-4.1"):
    """
    Print a summary table of RAG evaluation results for generated questions.

    Args:
        evaluation_results_paths: dict mapping system names to result file paths.
        evaluation_llm: name of the LLM used for evaluation (for display only).

    The function loads evaluation results, aggregates scores for each metric,
    and prints a table comparing all systems on Contextual Recall and Correctness.
    Only queries with at least one non-None metric score are included.
    """

    evaluation_results_paths = {k: Path(v) for k, v in evaluation_results_paths.items()}
    evaluation_results = {k: json.load(p.open("r", encoding="utf-8")) for k, p in evaluation_results_paths.items()}

    system_names = list(evaluation_results.keys())
    METRIC_NAMES = ["Retrieval Completeness", "Correctness [GEval]"]

    metrics_scores = {sys: {m: [] for m in METRIC_NAMES} for sys in system_names}
    answered_counts = {sys: 0 for sys in system_names}

    for sys, cases in evaluation_results.items():
        for case in cases:
            correctness_present = False
            for metric in case.get("metrics", []):
                name = metric.get("name", "")
                score = metric.get("score")

                if name.lower().startswith("retrievalcompleteness") or name.lower().startswith("retrieval completeness"):
                    if score is not None:
                        metrics_scores[sys]["Retrieval Completeness"].append(score)
                elif name.lower().startswith("correctness") and score is not None:
                    metrics_scores[sys]["Correctness [GEval]"].append(score)
                    correctness_present = True

            if correctness_present:
                answered_counts[sys] += 1

    # table
    header = ["Metric"] + system_names
    rows = []
    for m in METRIC_NAMES:
        row = [m]
        for sys in system_names:
            scores = metrics_scores[sys][m]
            if scores:
                mu = mean(scores) * 100
                sd = stdev(scores) * 100 if len(scores) > 1 else 0.0
                row.append(f"{mu:.2f}% ± {sd:.2f}%")
            else:
                row.append("-")
        rows.append(row)

    # overall
    overall = ["Final Score"]
    for sys in system_names:
        flat = [s for metric_scores in metrics_scores[sys].values() for s in metric_scores]
        overall.append(f"{mean(flat)*100:.2f}%" if flat else "-")
    rows.append(overall)

    rows.append(["Answered Queries"] + [str(answered_counts[sys]) for sys in system_names])
    rows.append(["Total Queries"] + [str(len(evaluation_results[sys])) for sys in system_names])

    print(f"RAG evaluation scores (mean ± std): evaluated with deepeval + {evaluation_llm}")
    print("Correctness ignores None scores. Retrieval Completeness evaluated for all queries.\n")
    print(tabulate(rows, headers=header, tablefmt="simple"))