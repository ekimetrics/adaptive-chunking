import pandas as pd
from tabulate import tabulate
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import cast, Any, Tuple
import numbers
import json

# ANSI color codes
BLUE = "\033[94m"
GREEN = "\033[92m"
RED = "\033[91m"
RESET = "\033[0m"

def show_metrics_times(
    mentions_performance_path: str | Path,
    metrics_performance_path: str | Path,
    precision: int = 3) -> None:
    """Display timing summary for mention extraction and per-metric computations.

    Parameters
    ----------
    mentions_performance_path
        Path to ``mentions_performance.parquet`` produced by *extract_mentions*.
    metrics_performance_path
        Path to ``metrics_performance.parquet`` produced by *compute_metrics*.
    precision
        Number of decimal places to display for the times.
    """

    # ── Mention extraction total time ─────────────────────────────────────────
    try:
        mentions_df = pd.read_parquet(mentions_performance_path)
    except Exception as e:  # noqa: BLE001
        print(f"Could not read mentions performance parquet '{mentions_performance_path}': {e}")
        return

    total_mentions_time = mentions_df["time"].sum()
    print("Total mention extraction time (needed for references completeness): " + f"{total_mentions_time:.{precision}f}s")

    # ── Per-metric computation times ─────────────────────────────────────────
    try:
        metrics_df = pd.read_parquet(metrics_performance_path)
    except Exception as e:  # noqa: BLE001
        print(f"Could not read metrics performance parquet '{metrics_performance_path}': {e}")
        return

    if metrics_df.empty:
        print("No metric performance data available.")
        return

    # Aggregate total time per metric across all documents
    agg_times = metrics_df.groupby("metric")["time"].sum().reset_index()
    total_metrics_time = agg_times["time"].sum()

    # Sort for nicer display
    agg_times = agg_times.sort_values("time", ascending=False)

    # Append row with grand total
    total_row = pd.DataFrame({"metric": ["TOTAL"], "time": [total_metrics_time]})
    agg_times = pd.concat([agg_times, total_row], ignore_index=True)

    agg_times["time"] = agg_times["time"].apply(lambda x: f"{x:.{precision}f}s")

    print("\nTotal computation time per metric:\n")
    print(tabulate(agg_times, headers=["Metric", "Total time (s)"], tablefmt="simple", showindex=False))

def show_chunking_times(
    perf_path: str | Path,
    chunking_methods: list[str],
    apply_max_to_methods: list[str] | None = None,
    precision: int = 3) -> None:
    """Display total chunking time per chunking method.

    Parameters
    ----------
    perf_path
        Path to the ``performances.parquet`` file produced by :pyfunc:`split_documents`.
    chunking_methods
        List of chunking method identifiers to include in the report. Order is
        preserved.
    apply_max_to_methods
        List of methods for which the overall runtime should be taken as the
        *maximum* per-document time instead of the *sum*. Useful for asynchronous
        splitters where documents are processed in parallel. If *None*, an empty
        list is assumed.
    precision
        Number of decimal places to display for the total time.
    Notes
    -----
    The *performances* parquet contains one row per *(document, method)* with the
    time taken to split that document. When methods run *synchronously* we sum
    times across documents. For *asynchronous* methods we assume the documents
    are processed in parallel, therefore the overall runtime is approximated by
    the *maximum* per-document time.
    """

    try:
        df = pd.read_parquet(perf_path)
    except Exception as e:  # noqa: BLE001
        print(f"Could not read performances parquet '{perf_path}': {e}")
        return

    # Normalise param
    apply_max_to_methods = apply_max_to_methods or []

    # Keep only the requested methods
    df = df[df["method"].isin(chunking_methods)]

    if df.empty:
        print("No timing information found for the requested chunking methods.")
        return

    total_times: list[tuple[str, float]] = []

    for method in chunking_methods:
        sub = df[df["method"] == method]
        if sub.empty:
            continue

        use_max = method in apply_max_to_methods
        total_time = sub["time"].max() if use_max else sub["time"].sum()
        total_times.append((method, total_time))

    if not total_times:
        print("No timing information available for the specified methods.")
        return

    pretty = pd.DataFrame(total_times, columns=["Chunking method", "Total time (s)"])
    pretty["Total time (s)"] = pretty["Total time (s)"].apply(lambda x: f"{x:.{precision}f}s")

    print("Total chunking time per method:\n")
    print(tabulate(pretty, headers="keys", tablefmt="simple", showindex=False))

def _format_metric_value(value: Any, format_spec: str) -> str:
    """Safely formats a metric value, handling non-numeric types."""
    if not isinstance(value, numbers.Number) or not pd.notna(value):
        return "N/A"

    if format_spec == "s":
        return f"{value:.2f}s"
    if format_spec == "int":
        return str(int(round(value)))
    if format_spec == "%":
        return f"{(value * 100):.1f}%"
    return "N/A"

def _format_pm_value(vals: Tuple[Any, Any], format_spec: str) -> str:
    """Safely formats a value with plus/minus, e.g., mean ± std."""
    m, s = vals
    if not (
        isinstance(m, numbers.Number)
        and pd.notna(m)
        and isinstance(s, numbers.Number)
        and pd.notna(s)
    ):
        return "N/A"

    if format_spec == "int_pm_int":
        return f"{int(round(m))} ± {int(round(s))}"
    if format_spec == "%_pm_%":
        return f"{(m * 100):.1f}% ± {(s * 100):.1f}%"
    return "N/A"

def output_best_chunks(
    chunks_df_path: str | Path,
    metrics_df_path: str | Path,
    weights: dict[str, float],
    output_dir: Path | str,
    default_method: str = "page") -> dict[str, list[dict]]:
    """Selects the best chunking strategy for every document and writes the resulting
    chunks to *output_dir*.

    Parameters
    ----------
    chunks_df_path : str | Path
        A dataframe produced by *split_documents* / *postprocessing* that contains at
        least the following columns: ``doc_name``, ``method`` (chunking method
        identifier), ``chunk_index`` (integer order) and ``chunk_text`` (str).
    metrics_df_path : str | Path
        A dataframe produced by *compute_metrics* with the columns
        ``doc_name``, ``chunking_method`` (method identifier), ``metric_name`` and
        ``score``.
    weights : dict[str, float]
        Mapping from metric name to its weight. Only metrics with *weight > 0* are
        considered when computing the weighted score that determines the *best*
        method.
    output_dir : Path | str
        Directory where a ``*.json`` file per document will be written containing the
        ordered list of **best** chunks. The directory is created if it does not
        exist.
    default_method : str, default "page"
        Fallback chunking method to use for a document when no metric information is
        available.

    Returns
    -------
    dict[str, list[dict]]
        A mapping ``doc_name -> list_of_chunk_dicts`` using the best method per document.
        Each chunk dict contains:
            - ``chunk_text`` (str): the chunk content.
            - ``titles_context`` (str): concatenated title information for the chunk.
            - ``chunk_pages`` (list[int]): page numbers overlapped by the chunk.
    """

    try:
        chunks_df = pd.read_parquet(chunks_df_path)
    except Exception as e:
        print(f"Could not read chunks parquet '{chunks_df_path}': {e}")
        return

    try:
        metrics_df = pd.read_parquet(metrics_df_path)
    except Exception as e:
        print(f"Could not read metrics parquet '{metrics_df_path}': {e}")
        return

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    best_chunks_per_doc: dict[str, list[dict]] = {}

    scoring_metrics_keys = [m for m, w in weights.items() if w > 0]

    for doc_name, doc_scores in metrics_df.groupby("doc_name"):
        pivot = doc_scores.pivot(
            index="metric_name", columns="chunking_method", values="score"
        )

        scores_subset = pivot.loc[pivot.index.intersection(scoring_metrics_keys)]

        # if we have no usable metrics for this document, fall back to the first
        # available method recorded in *chunks_df* for that document
        if scores_subset.empty or scores_subset.isna().all().all():
            # no metrics available – default to the supplied *default_method* if the
            # document has been split with it. Otherwise, fall back to the first
            # available method we find in *chunks_df* for this document
            methods_for_doc = chunks_df.loc[chunks_df["doc_name"] == doc_name, "method"].values
            if default_method in methods_for_doc:
                best_method = default_method
            else:
                if len(methods_for_doc) == 0:
                    # nothing we can do – skip this doc
                    continue
                best_method = methods_for_doc[0]
        else:
            best_method, _ = find_best_method(scores_subset, weights)

        # retrieve the chunks for the chosen method
        doc_chunks_df = chunks_df[(chunks_df["doc_name"] == doc_name) & (chunks_df["method"] == best_method)]
        if doc_chunks_df.empty:
            continue

        # ensure correct ordering
        sorted_df = doc_chunks_df.sort_values("chunk_index")

        chunk_dicts: list[dict] = []
        for _, row in sorted_df.iterrows():
            chunk_text = str(row["chunk_text"]).strip()
            titles_context = row.get("titles_context", "")
            chunk_pages = row.get("chunk_pages", [])
            # ensure JSON-serializable plain Python types
            if isinstance(chunk_pages, (np.ndarray, pd.Series)):
                chunk_pages = chunk_pages.tolist()
            # coerce numpy scalar/ints to plain int
            chunk_pages = [int(x) for x in chunk_pages] if chunk_pages else []

            chunk_dicts.append({
                "chunk_text": chunk_text,
                "titles_context": str(titles_context),
                "chunk_pages": chunk_pages,
            })

        best_chunks_per_doc[doc_name] = chunk_dicts

        output_data = {
            "doc_name": doc_name,
            "method": best_method,
            "chunks": chunk_dicts,
        }

        out_file = output_path / f"{doc_name}.json"
        try:
            with out_file.open("w", encoding="utf-8") as f:
                json.dump(output_data, f, indent=4)
        except IOError as e:
            print(f"Warning: could not write chunks for '{doc_name}' to '{out_file}': {e}")

    print(f"Best chunks for all documents saved to: {output_path}")
    return best_chunks_per_doc

def find_best_method(scores_df: pd.DataFrame, weights: Any) -> tuple[str, list[float]]:
    """Return the best chunking *method* and the list of normalized weighted scores.

    The weighted score for a method is the weighted average of the provided
    metrics, *ignoring* metrics that are ``NaN`` for that method.
    """

    # Ensure weights is a Series aligned to metric rows
    if not isinstance(weights, pd.Series):
        weights = pd.Series(weights, index=scores_df.index)

    # Restrict to metrics present in *scores_df*
    weights = weights.reindex(scores_df.index)

    if weights.isna().all():
        raise ValueError("No weights overlap with the metrics in *scores_df*.")

    # Compute weighted average per column, skipping NaNs
    norm_scores: list[float] = []
    for method in scores_df.columns:
        col = scores_df[method]
        mask = col.notna() & weights.notna()
        if not mask.any():
            norm_scores.append(float("nan"))
            continue
        w_sum = weights[mask].sum()
        weighted_sum = (col[mask] * weights[mask]).sum()
        norm_scores.append(float(weighted_sum / w_sum))

    # Create a Series for easy idxmax (NaNs will be ignored by idxmax)
    norm_series = pd.Series(norm_scores, index=scores_df.columns)

    best_method = cast(str, norm_series.idxmax())
    return best_method, norm_scores

def serialize_chunks_for_docs(
    chunks_df: pd.DataFrame,
    method: str,
    output_dir: Path | str,
    ) -> dict[str, list[dict]]:
    """Serialize *chunks_df* (already filtered for *method*) into JSON files.

    Returns a mapping ``doc_name -> list_of_chunk_dicts`` similar to
    *output_best_chunks*.
    """

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    chunks_per_doc: dict[str, list[dict]] = {}

    for doc_name, doc_df in chunks_df.groupby("doc_name"):
        # Ensure deterministic ordering
        sorted_df = doc_df.sort_values("chunk_index")

        chunk_dicts: list[dict] = []
        for _, row in sorted_df.iterrows():
            chunk_text = str(row.get("chunk_text", "").strip())
            titles_context = str(row.get("titles_context", ""))
            chunk_pages = row.get("chunk_pages", [])

            # Ensure JSON-serialisable plain Python types
            if isinstance(chunk_pages, (np.ndarray, pd.Series)):
                chunk_pages = chunk_pages.tolist()
            chunk_pages = [int(x) for x in chunk_pages] if chunk_pages else []

            chunk_dicts.append(
                {
                    "chunk_text": chunk_text,
                    "titles_context": titles_context,
                    "chunk_pages": chunk_pages,
                }
            )

        # Store in-memory mapping
        chunks_per_doc[doc_name] = chunk_dicts

        # Write JSON file
        data_out = {
            "doc_name": doc_name,
            "method": method,
            "chunks": chunk_dicts,
        }

        json_file = out_path / f"{doc_name}.json"
        try:
            with json_file.open("w", encoding="utf-8") as f:
                json.dump(data_out, f, indent=4)
        except IOError as e:
            print(
                f"Warning: could not write chunks for '{doc_name}' to '{json_file}': {e}"
            )

    return chunks_per_doc

def output_selected_chunks(
    chunks_df_paths: dict[str, Path | str],
    selection: list[dict[str, str | Path]],
    ) -> dict[str, dict[str, list[dict]]]:
    """Export chunks for user-selected chunking methods.

    Parameters
    ----------
    chunks_df_paths
        Mapping *df_name -> parquet_path* with the chunk dataframes.
    selection
        List of selection dictionaries. Each dictionary must have:
            - ``chunks_df`` (or ``df_name``): name matching a key in ``chunks_df_paths``.
            - ``chunking_method``: identifier of the method to export.
            - ``output_dir``: directory to write the resulting JSON chunk files.

    Returns
    -------
    dict[str, dict[str, list[dict]]]
        Nested mapping ``df_name -> doc_name -> list_of_chunk_dicts`` with the
        exported chunks.
    """

    exported: dict[str, dict[str, list[dict]]] = {}

    for cfg in selection:
        df_name = cfg.get("chunks_df") or cfg.get("df_name") 
        chunking_method = cfg.get("chunking_method")
        output_dir = cfg.get("output_dir")

        if not df_name:
            print("Warning: selection entry missing 'df_name' – skipping.")
            continue

        if df_name not in chunks_df_paths:
            print(f"Warning: dataframe '{df_name}' not found in chunks_df_paths – skipping.")
            continue

        if not chunking_method or not output_dir:
            print(
                f"Warning: selection entry for '{df_name}' must include 'chunking_method' and 'output_dir' – skipping."
            )
            continue

        # Load dataframe
        parquet_path = Path(chunks_df_paths[df_name])
        if not parquet_path.exists():
            print(f"Warning: parquet path '{parquet_path}' does not exist – skipping '{df_name}'.")
            continue

        try:
            df = pd.read_parquet(parquet_path)
        except Exception as e:  # noqa: BLE001
            print(f"Warning: could not read parquet '{parquet_path}': {e}")
            continue

        # Filter for the requested method
        filtered_df = df[df["method"] == chunking_method]
        if filtered_df.empty:
            print(
                f"Warning: method '{chunking_method}' not found in dataframe '{df_name}' – skipping."
            )
            continue

        # Initialise nested dict if first time we encounter this df_name
        if df_name not in exported:
            exported[df_name] = {}

        exported[df_name].update(
            serialize_chunks_for_docs(filtered_df, chunking_method, output_dir)
        )
        print(f"Captured and saved chunks for {df_name} with method {chunking_method} to {output_dir}")

    return exported

def show_chunking_overall_metametrics(
    df_chunks_path: str | Path,
    df_metrics_path: str | Path,
    chunking_methods: list[str],
    weights: dict[str, float]) -> None:
    """Display overall meta metrics derived directly from *chunks* dataframe.

    Parameters
    ----------
    df_chunks_path
        Path to the parquet file produced by *split_documents* containing at
        least the columns ``method`` and ``chunk_len``.
    df_metrics_path
        Path to the metrics parquet file containing chunking metric scores. Used
        to determine the *best* method per document according to *weights*.
    chunking_methods
        List of chunking methods identifiers to include, order preserved.
    weights
        Mapping metric ➔ weight for selecting the best method per document.
    """

    try:
        chunks_df = pd.read_parquet(df_chunks_path)
    except Exception as e:  # noqa: BLE001
        print(f"Could not read chunks parquet '{df_chunks_path}': {e}")
        return

    chunks_df = chunks_df[chunks_df["method"].isin(chunking_methods)]

    if chunks_df.empty:
        print("No chunks data available for the requested methods.")
        return

    # Load metrics to find best method per doc
    try:
        metrics_df = pd.read_parquet(df_metrics_path)
    except Exception as e:  # noqa: BLE001
        print(f"Could not read metrics parquet '{df_metrics_path}': {e}")
        return

    metrics_df = metrics_df[metrics_df["chunking_method"].isin(chunking_methods)]

    # Determine best method per doc
    best_method_per_doc: dict[str, str] = {}
    scoring_metrics = [m for m, w in weights.items() if w > 0]
    for doc_name, g in metrics_df.groupby("doc_name"):
        pivot = g.pivot(index="metric_name", columns="chunking_method", values="score")
        scores = pivot.loc[pivot.index.intersection(scoring_metrics)]
        if scores.empty or scores.isna().all().all():
            continue
        best_method, _ = find_best_method(scores, weights)
        best_method_per_doc[doc_name] = best_method

    meta_rows = [
        "avg_chunk_tokens",
        "max_chunk_tokens",
        "min_chunk_tokens",
        "stddev_chunk_tokens",
        "total_num_chunks",
    ]

    pretty = pd.DataFrame(index=meta_rows)

    for method in chunking_methods:
        sub = chunks_df[chunks_df["method"] == method]
        lens = sub["chunk_len"].astype(float)
        if lens.empty:
            vals = [np.nan]*5
        else:
            vals = [
                lens.mean(),
                lens.max(),
                lens.min(),
                lens.std(ddof=0),
                len(lens),
            ]
        formatted = [
            _format_metric_value(vals[0], "int"),
            _format_metric_value(vals[1], "int"),
            _format_metric_value(vals[2], "int"),
            _format_metric_value(vals[3], "int"),
            _format_metric_value(vals[4], "int"),
        ]
        pretty[method] = formatted

    # Compute *best* column
    best_rows = []
    for doc_name, best_m in best_method_per_doc.items():
        rows = chunks_df[(chunks_df["doc_name"] == doc_name) & (chunks_df["method"] == best_m)]
        best_rows.append(rows)

    if best_rows:
        best_df = pd.concat(best_rows, ignore_index=True)
        lens_best = best_df["chunk_len"].astype(float)
        best_vals = [
            lens_best.mean(),
            lens_best.max(),
            lens_best.min(),
            lens_best.std(ddof=0),
            len(lens_best),
        ]
        pretty["best"] = [
            _format_metric_value(best_vals[0], "int"),
            _format_metric_value(best_vals[1], "int"),
            _format_metric_value(best_vals[2], "int"),
            _format_metric_value(best_vals[3], "int"),
            _format_metric_value(best_vals[4], "int"),
        ]

    print("Meta metrics (directly from chunks):")
    print(tabulate(pretty.T, headers="keys", tablefmt="simple", showindex=True))

def show_chunking_overall_report(
    df_path: str | Path,
    chunking_methods: list[str],
    metrics: list[str],
    weights: dict[str, float]) -> None:
    """Display an overall chunking report for a single metrics dataframe.

    Parameters
    ----------
    df_path
        Path to the parquet file containing the chunking metrics.
    chunking_methods
        List of chunking methods to include in the report. Order is preserved.
    metrics
        Ordered list of *chunking* metrics to display (e.g. completeness, cohesion …).
        ``final_weighted_score`` is **always** appended automatically as the last row.
    weights
        Mapping metric ➔ weight used to compute the *best* method per document.
    """

    # Load dataframe
    try:
        df = pd.read_parquet(df_path)
    except Exception as e:  # noqa: BLE001
        print(f"Could not read metrics parquet '{df_path}': {e}")
        return

    # Keep only the requested chunking methods
    df = df[df["chunking_method"].isin(chunking_methods)]

    # Compute *best* method per-document
    best_metrics_per_doc: list[pd.Series] = []
    for doc_name, g in df.groupby("doc_name"):
        pivot = g.pivot(index="metric_name", columns="chunking_method", values="score")
        scoring_metrics = [m for m, w in weights.items() if w > 0]
        scores = pivot.loc[pivot.index.intersection(scoring_metrics)]

        if scores.empty or scores.isna().all().all():
            continue

        best_method, _ = find_best_method(scores, weights)
        best_metrics_per_doc.append(pivot[best_method])

    best_df = pd.DataFrame(best_metrics_per_doc)
    best_mean = best_df.mean()
    best_std = best_df.std()

    # Aggregate mean/std for requested methods (chunking metrics only)
    agg = (
        df[df["metric_name"].isin(metrics)]
        .groupby(["metric_name", "chunking_method"])["score"]
        .agg(["mean", "std"])
        .reset_index()
    )
    wide = agg.pivot(index="metric_name", columns="chunking_method")

    # Helper to choose formatting based on metric name
    def _spec_for_metric(m_name: str) -> str:
        if "tokens" in m_name or "chunks" in m_name:
            return "int_pm_int"
        return "%_pm_%"

    # Build pretty tables
    pretty_chunking = pd.DataFrame(index=metrics + ["final_weighted_score"])

    for method in chunking_methods:
        mean_col = wide.get(("mean", method))
        std_col = wide.get(("std", method))

        # Chunking metrics + placeholder for final score
        chunking_formatted = []
        for m in metrics:
            m_val = mean_col.get(m, np.nan) if mean_col is not None else np.nan
            s_val = std_col.get(m, np.nan) if std_col is not None else np.nan
            chunking_formatted.append(_format_pm_value((m_val, s_val), _spec_for_metric(m)))
        # add placeholder for final_weighted_score (to be filled later)
        chunking_formatted.append("")
        pretty_chunking[method] = chunking_formatted

    # Insert *best* aggregated column
    best_chunking_vals = [
        _format_pm_value((best_mean.get(m, np.nan), best_std.get(m, np.nan)), _spec_for_metric(m))
        for m in metrics
    ]
    best_chunking_vals.append("")  # placeholder for final_weighted_score
    pretty_chunking["best"] = best_chunking_vals

    # Compute weighted score per method (mean of scores)
    scoring_metrics = [m for m, w in weights.items() if w > 0]
    scores_mean = (
        wide["mean"].loc[wide["mean"].index.intersection(scoring_metrics)]
        if "mean" in wide.columns else pd.DataFrame()
    )

    if not scores_mean.empty:
        best_method_overall, norm_scores = find_best_method(scores_mean, weights)
        final_scores_series = pd.Series(norm_scores, index=scores_mean.columns)
        pretty_chunking.loc["final_weighted_score", scores_mean.columns] = (
            final_scores_series.apply(lambda x: f"{x*100:.2f}%")
        )
        pretty_chunking.at["final_weighted_score", best_method_overall] = (
            f"{RED}{pretty_chunking.at['final_weighted_score', best_method_overall]}{RESET}"
        )

    # Compute score for *best* column
    weights_series = pd.Series(weights).reindex(scoring_metrics).fillna(0)
    if weights_series.sum() > 0:
        best_score = (best_mean.reindex(scoring_metrics).fillna(0) * weights_series).sum() / weights_series.sum()
    else:
        best_score = 0
    pretty_chunking.at["final_weighted_score", "best"] = f"{best_score*100:.2f}%"

    # Display results
    print("Chunking metrics (mean ± std dev):")
    print(tabulate(pretty_chunking.T, headers="keys", tablefmt="simple", showindex=True))

    print("\n" + "_" * 80)
    print("Scoring weights:")
    total_w = sum(weights.values())
    if total_w == 0:
        print("All weights are zero.")
    else:
        weighted_tbl = [[m, f"{(w / total_w) * 100:.2f}%"] for m, w in weights.items() if w > 0]
        print(tabulate(weighted_tbl, headers=["Metric", "Weight (%)"], tablefmt="simple"))

def show_chunking_metrics_per_doc(
    df_path: str | Path,
    chunking_methods: list[str],
    metrics: list[str],
    meta_metrics: list[str],
    weights: dict[str, float]) -> None:
    """
    Display per-document chunking metrics and meta metrics for each document, and highlight the best chunking method per document.

    This function reads a metrics parquet file, filters for the specified chunking methods, and for each document:
      - Displays a table of chunking metrics (e.g., accuracy, recall, etc.) for each method.
      - Computes and displays meta metrics (e.g., number of chunks, average chunk length) for each method.
      - Identifies and highlights the best chunking method for the document based on the provided metric weights.

    At the end, prints a summary table listing the best chunking method for each document.

    Args:
        df_path (str | Path): Path to the metrics parquet file.
        chunking_methods (list[str]): List of chunking method names to include.
        metrics (list[str]): List of metric names to display and use for scoring.
        meta_metrics (list[str]): List of meta metric names to display.
        weights (dict[str, float]): Dictionary mapping metric names to their weights for best-method selection.
    """

    try:
        df = pd.read_parquet(df_path)
    except Exception as e:  # noqa: BLE001
        print(f"Could not read metrics parquet '{df_path}': {e}")
        return

    df = df[df["chunking_method"].isin(chunking_methods)]

    summary: list[tuple[str, str]] = []

    for doc_name, g in df.groupby("doc_name"):
        pivot = g.pivot(index="metric_name", columns="chunking_method", values="score")

        # Determine best method
        score_rows = pivot.loc[pivot.index.intersection(metrics)]
        best_method, norm_scores = find_best_method(score_rows, weights)
        summary.append((doc_name, best_method))

        # --- Chunking metrics table ---
        chunk_tbl = pivot.reindex(metrics).applymap(lambda v: "N/A" if pd.isna(v) else f"{v*100:.1f}%")
        chunk_tbl.loc["final_weighted_score"] = [f"{s*100:.2f}%" for s in norm_scores]

        # (Per-metric highlighting skipped on request)
        chunk_tbl.at["final_weighted_score", best_method] = (
            f"{RED}{chunk_tbl.at['final_weighted_score', best_method]}{RESET}"
        )

        # --- Meta metrics table ---
        meta_tbl = pd.DataFrame(index=meta_metrics, columns=chunking_methods)
        for m in meta_metrics:
            source_metric = "num_chunks" if m == "total_num_chunks" else m
            for method in chunking_methods:
                val = pivot.at[source_metric, method] if source_metric in pivot.index else np.nan
                meta_tbl.at[m, method] = "N/A" if pd.isna(val) else int(round(cast(float, val)))

        # --- Print ---
        print(f"\n{doc_name}")
        print("Chunking metrics:")
        print(tabulate(chunk_tbl.T, headers="keys", tablefmt="simple", showindex=True))
        print("\nMeta metrics:")
        print(tabulate(meta_tbl.T, headers="keys", tablefmt="simple", showindex=True))
        print(f"\nBest method: {RED}{best_method}{RESET}")
        print("_" * 80)

    # Summary table
    print("\nSummary of best chunking method per document:\n")
    print(tabulate(summary, headers=["Doc name", "Best method"], tablefmt="simple"))

def plot_metric_correlations(
    df_path: str | Path,
    metrics: list[str],
    chunking_methods: list[str],
    figsize=(6, 12),
    cmap="seismic"):
    """
    Plot correlation heatmaps (Pearson, Kendall, Spearman) for selected chunking metrics.

    Args:
        df_path (str | Path): Path to the metrics parquet file.
        metrics (list[str]): List of metric names to include in the correlation analysis.
        chunking_methods (list[str]): List of chunking methods to filter the data.
        figsize (tuple, optional): Figure size for the plots. Default is (6, 12).
        cmap (str, optional): Colormap for the heatmaps. Default is "seismic".

    Displays:
        Three heatmaps showing the pairwise correlations between the selected metrics.
    """

    try:
        df = pd.read_parquet(df_path)
    except Exception as e:
        print(f"Could not read metrics parquet '{df_path}': {e}")
        return

    df = df[df["chunking_method"].isin(chunking_methods)]

    all_metrics_df = (
        df.pivot_table(
            values="score",
            index=["doc_name", "chunking_method"],
            columns="metric_name",
            aggfunc="mean",
        )
        .reset_index()
    )

    df_for_corr = (
        all_metrics_df[[c for c in metrics if c in all_metrics_df.columns]]
        .apply(pd.to_numeric, errors="coerce")
        .dropna()
    )

    pearson = df_for_corr.corr(method="pearson")
    kendall = df_for_corr.corr(method="kendall")
    spearman = df_for_corr.corr(method="spearman")

    for m in (pearson, kendall, spearman):
        np.fill_diagonal(m.values, np.nan)

    masks = {
        "pearson": np.triu(np.ones_like(pearson, dtype=bool), k=0),
        "kendall": np.triu(np.ones_like(kendall, dtype=bool), k=0),
        "spearman": np.triu(np.ones_like(spearman, dtype=bool), k=0),
    }

    fig, axes = plt.subplots(3, 1, figsize=figsize)

    sns.heatmap(
        pearson,
        ax=axes[0],
        annot=True,
        cmap=cmap,
        fmt=".2f",
        linewidths=0.5,
        vmin=-1,
        vmax=1,
        cbar=True,
        cbar_kws={"label": "Correlation"},
        mask=masks["pearson"],
    )
    axes[0].set_title("Pearson Correlation")
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45, ha="right")
    axes[0].set_yticklabels(axes[0].get_yticklabels(), rotation=0)

    sns.heatmap(
        kendall,
        ax=axes[1],
        annot=True,
        cmap=cmap,
        fmt=".2f",
        linewidths=0.5,
        vmin=-1,
        vmax=1,
        cbar=True,
        cbar_kws={"label": "Correlation"},
        mask=masks["kendall"],
    )
    axes[1].set_title("Kendall Tau Correlation")
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45, ha="right")
    axes[1].set_yticklabels(axes[1].get_yticklabels(), rotation=0)

    sns.heatmap(
        spearman,
        ax=axes[2],
        annot=True,
        cmap=cmap,
        fmt=".2f",
        linewidths=0.5,
        vmin=-1,
        vmax=1,
        cbar=True,
        cbar_kws={"label": "Correlation"},
        mask=masks["spearman"],
    )
    axes[2].set_title("Spearman Rank Correlation")
    axes[2].set_xticklabels(axes[2].get_xticklabels(), rotation=45, ha="right")
    axes[2].set_yticklabels(axes[2].get_yticklabels(), rotation=0)

    plt.tight_layout()
    plt.show()