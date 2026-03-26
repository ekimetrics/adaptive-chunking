import os
import json
from IPython.display import Markdown, HTML, display, clear_output
import markdown
import random
import ipywidgets as widgets
from ipywidgets.embed import embed_minimal_html
import pathlib
import matplotlib.pyplot as plt
import pandas as pd
import html
from html import escape
import glob
import functools
from pathlib import Path
from tabulate import tabulate
from typing import Callable

def display_html(html_string: str, height: int = 1000) -> None:
    html_scrollable = f"""
    <div style="height: {height}px; overflow-y: scroll; border: 1px solid #ccc; padding: 10px;">
      {html_string}
    </div>
    """
    display(HTML(html_scrollable))
  
def display_markdown(markdown_string: str) -> None:
    html_content = markdown.markdown(markdown_string, extensions=["extra"], output_format="html5")
    display_html(html_content)

def convert_to_html(text: str) -> str:
    html_text = text.replace("\n", "<br>")
    return html_text


def visualize_splits_txt(
    splits: list[str],
    whole_text: str,
    height: int = 200,
) -> None:
    """
    Visualize text splits by coloring each split with a different color.
    When two or more splits overlap on the same characters, those characters
    are rendered in an explicit overlap font color (no background shading).

    Args:
        splits:      List of text chunks / splits.
        whole_text:  The original complete text.
        height:      Height (in px) used by the display helper.
    """
    # Base colors for individual (non‑overlapping) splits
    split_colors = ["#FF5733", "#3357FF"]
    # Font color used when overlap occurs
    overlap_font_color = "#33FF57"

    # For every character in whole_text store the indices of splits covering it
    char_to_split_indices = [[] for _ in range(len(whole_text))]

    search_from = 0
    for i, split in enumerate(splits):
        try:
            start_pos = whole_text.index(split, search_from)
            for j in range(start_pos, start_pos + len(split)):
                if j < len(whole_text):
                    char_to_split_indices[j].append(i)
            # move the search window forward by 1 char to allow overlaps
            search_from = start_pos + 1
        except ValueError:
            # Split not found in the remaining text
            print(f"Warning: split {i} not found in text")

    html_content = ""
    current_spans: List[int] = []

    for idx, char in enumerate(whole_text):
        spans_now = char_to_split_indices[idx]

        # If we moved into/out‑of a different set of covering splits, close/open spans
        if spans_now != current_spans:
            if current_spans:
                html_content += "</span>"

            if spans_now:
                spans_now.sort()  # deterministic color assignment
                if len(spans_now) > 1:
                    # Overlap → use dedicated overlap font color
                    color = overlap_font_color
                else:
                    # Single split → use its assigned color
                    color = split_colors[spans_now[0] % len(split_colors)]

                html_content += f'<span style="color: {color};">'

            current_spans = spans_now

        # Preserve line‑breaks as <br>
        html_content += "<br>" if char == "\n" else html.escape(char)

    # Close any lingering open span
    if current_spans:
        html_content += "</span>"

    # Show the generated HTML
    display_html(html_content, height=height)

def visualize_splits_markdown(splits: list[str], whole_text: str) -> None:
    """
    Visualize markdown splits by inserting an HTML divider (<hr>) between chunks.
    This preserves the original markdown formatting.
    
    Args:
        splits: List of text chunks/splits (substrings of the whole markdown text)
        whole_text: The original complete markdown text
    """
    combined_md = ""
    remaining_text = whole_text
    first_split = True

    for split in splits:
        # Find the position of the split in the remaining text.
        start_pos = remaining_text.find(split)
        if start_pos == -1:
            continue

        # Append any text before the split (preserving markdown formatting)
        combined_md += remaining_text[:start_pos]

        # Insert a horizontal rule as a divider, if this isn't the first split
        if not first_split:
            # Ensure the divider is on its own line by surrounding it with newlines.
            combined_md += "\n\n<hr>\n\n"
        else:
            first_split = False

        # Insert the split itself
        combined_md += split

        # Update the remaining text to start after the current split
        remaining_text = remaining_text[start_pos + len(split):]

    # Append any remaining text from the original markdown
    if remaining_text:
        combined_md += remaining_text

    # Render the final markdown with dividers
    display_markdown(combined_md)

def visualize_text_clusters(text,
                             clusters_char_offsets,
                             html_height=1000,
                             dim_opacity=0.1):
    if len(clusters_char_offsets) < 1:
        raise ValueError("No clusters provided for visualization.")

    palette = [
        'hsl(0,90%,45%)','hsl(180,90%,45%)','hsl(72,90%,45%)','hsl(252,90%,45%)',
        'hsl(36,90%,45%)','hsl(216,90%,45%)','hsl(108,90%,45%)','hsl(288,90%,45%)',
        'hsl(144,90%,45%)','hsl(324,90%,45%)','hsl(54,90%,45%)','hsl(234,90%,45%)',
        'hsl(126,90%,45%)','hsl(306,90%,45%)','hsl(18,90%,45%)','hsl(198,90%,45%)',
        'hsl(90,90%,45%)','hsl(270,90%,45%)','hsl(162,90%,45%)','hsl(342,90%,45%)'
    ]
    colors = [palette[i % 20] for i in range(len(clusters_char_offsets))]

    # char index → (sorted) list of cluster-ids
    char2cids = {}
    for cid, spans in enumerate(clusters_char_offsets):
        for s, e in spans:
            for p in range(s, min(e + 1, len(text))):
                char2cids.setdefault(p, []).append(cid)

    segments, cur_ids, cur_txt = [], None, ""

    def flush():
        nonlocal cur_txt, cur_ids
        if not cur_txt:
            return
        if cur_ids:
            cids_str = ",".join(str(x) for x in cur_ids)
            bg = colors[cur_ids[0]]
            if len(cur_ids) == 1:
                seg = (
                    '<span class="mention" data-cids="{c}" '
                    'style="background:{bg};color:#fff;">{txt}</span>'
                ).format(c=cids_str, bg=bg, txt=html.escape(cur_txt))
            else:
                u = colors[cur_ids[1]]
                seg = (
                    '<span class="mention overlap" data-cids="{c}" '
                    'style="background:{bg};color:#fff;'
                    'border-bottom:2px solid {u};border-top:2px solid {u};'
                    'text-decoration:underline wavy {u};">'
                    '{txt}</span>'
                ).format(c=cids_str, bg=bg, u=u, txt=html.escape(cur_txt))
            segments.append(seg)
        else:
            segments.append(html.escape(cur_txt))
        cur_txt = ""

    for i, ch in enumerate(text):
        ids = sorted(char2cids.get(i, []))
        if ids != cur_ids:
            flush()
            cur_ids = ids
        cur_txt += ch
    flush()

    html_block = """
    <style>
      .mention {{
         transition:filter .15s,opacity .15s,border .15s;
         cursor:pointer;
      }}
      .mention.active {{
         filter:brightness(135%);
         border:2px solid currentColor;
      }}
      .mention.dim {{ opacity:{dim_opacity}; }}
      .cluster-box {{
         max-height:{html_height}px; overflow-y:auto; white-space:pre-wrap;
         line-height:1.4em; font-family:"IBM Plex Sans",sans-serif;
         font-size:.95rem;
      }}
    </style>

    <div class="cluster-box">{content}</div>

    <script>
      (function() {{
        const spans = document.querySelectorAll('.mention');
        let current = null;                    // id of cluster in focus

        function updateFocus(cid) {{
          spans.forEach(el => {{
            const ids = el.dataset.cids.split(',');
            const hit = ids.includes(cid);
            el.classList.toggle('active', hit);
            el.classList.toggle('dim', !hit);
          }});
        }}

        function clear() {{
          spans.forEach(el => el.classList.remove('active','dim'));
          current = null;
        }}

        spans.forEach(el => el.addEventListener('click', evt => {{
          const ids = el.dataset.cids.split(',');
          if (current === null || !ids.includes(current)) {{
            current = ids[0];                  // start with first id
          }} else {{
            const next = (ids.indexOf(current) + 1) % ids.length;
            current = ids[next];               // cycle to next id in that span
          }}
          updateFocus(current);
          evt.stopPropagation();
        }}));

        document.addEventListener('click', e => {{
          if (!e.target.closest('.mention')) clear();
        }});
      }})();
    </script>
    """.format(content="".join(segments),
               dim_opacity=dim_opacity,
               html_height=html_height)

    display(HTML(html_block))
    
def visualize_entity_pron_pairs(
    text: str,
    entity_pron_pairs,
    html_height: int = 1000,
    dim_opacity: float = 0.1):
    """
    Show entity–pronoun links in `text`.

    Parameters
    ----------
    text : str
        The source document.
    entity_pron_pairs : iterable
        Each element is [entity_span, pronoun_span], where a span is
        (start, end_inclusive). Elements may be lists, tuples or numpy arrays.
    """

    if entity_pron_pairs is None or len(entity_pron_pairs) == 0:
        raise ValueError("No entity–pronoun pairs supplied.")

    # normalise → list[(ent_start, ent_end), (pro_start, pro_end)]
    norm_pairs = []
    for pair in entity_pron_pairs:
        if len(pair) != 2:
            continue
        ent_span = tuple(int(x) for x in pair[0])   # ensure hashable
        pro_span = tuple(int(x) for x in pair[1])
        norm_pairs.append((ent_span, pro_span))

    entity2id = {}
    next_id = 0
    char2tag = {}            # char_idx → (role, entity_id, pair_id | None)

    # First, collect all character indices that are part of a pronoun.
    pronoun_chars = set()
    for _, pro_span in norm_pairs:
        for i in range(pro_span[0], pro_span[1] + 1):
            if 0 <= i < len(text):
                pronoun_chars.add(i)

    for idx, (ent_span, pro_span) in enumerate(norm_pairs):
        ent_id = entity2id.setdefault(ent_span, f"e{next_id}")
        if ent_id == f"e{next_id}":
            next_id += 1
        pair_id = f"p{idx}"

        # Tag entities, but only if the character is not part of a pronoun.
        for i in range(ent_span[0], ent_span[1] + 1):
            if 0 <= i < len(text) and i not in pronoun_chars:
                char2tag[i] = ("entity", ent_id, None)

        # Tag pronouns.
        for i in range(pro_span[0], pro_span[1] + 1):
            if 0 <= i < len(text):
                char2tag[i] = ("pronoun", ent_id, pair_id)

    ENTITY_COL = "hsl(140,70%,40%)"   # green
    PRONOUN_COL = "hsl(210,80%,45%)"  # blue

    segments, cur_tag, cur_txt = [], None, ""

    def flush():
        nonlocal cur_txt, cur_tag
        if not cur_txt:
            return
        if cur_tag is None:
            segments.append(escape(cur_txt))
        else:
            role, e_id, p_id = cur_tag
            style = (
                f"background:{ENTITY_COL};color:#fff;"
                if role == "entity"
                else f"background:{PRONOUN_COL};color:#fff;"
            )
            attrs = f'data-entity="{e_id}"'
            if p_id:
                attrs += f' data-pair="{p_id}"'
            segments.append(
                f'<span class="mention role-{role}" {attrs} style="{style}">'
                f"{escape(cur_txt)}</span>"
            )
        cur_txt = ""

    for pos, ch in enumerate(text):
        tag = char2tag.get(pos)
        if tag != cur_tag:
            flush()
            cur_tag = tag
        cur_txt += ch
    flush()

    html_block = f"""
    <style>
    .mention {{
        cursor:pointer; transition:filter .15s,opacity .15s;
        border-radius:3px; padding:0 2px;
    }}
    .mention.active {{
        filter:brightness(135%); border:2px solid currentColor;
    }}
    .mention.dim {{ opacity:{dim_opacity}; }}
    .pair-box {{
        max-height:{html_height}px; overflow-y:auto; white-space:pre-wrap;
        line-height:1.4em; font-family:"IBM Plex Sans",sans-serif;
        font-size:.95rem;
    }}
    </style>

    <div class="pair-box">{''.join(segments)}</div>

    <script>
    (function() {{
    const spans = document.querySelectorAll('.mention');
    let mode = null, v = null;            // mode 'entity' | 'pair'

    function clear() {{
        spans.forEach(el => el.classList.remove('active','dim'));
        mode = v = null;
    }}

    function apply() {{
        spans.forEach(el => {{
            let hit = false;
            if (mode === 'entity') {{
                hit = el.dataset.entity === v;
            }} else if (mode === 'pair') {{
                if (el.classList.contains('role-entity')) {{
                    hit = el.dataset.entity === v.entity;
                }} else {{
                    hit = el.dataset.pair === v.pair;
                }}
            }}
            el.classList.toggle('active', hit);
            el.classList.toggle('dim', !hit && mode !== null);
        }});
    }}

    spans.forEach(el => el.addEventListener('click', e => {{
        if (el.classList.contains('role-entity')) {{
            const ent = el.dataset.entity;
            if (mode === 'entity' && v === ent) {{
                clear();
            }} else {{
                mode = 'entity';
                v = ent;
                apply();
            }}
        }} else {{
            const ent = el.dataset.entity;
            const pair = el.dataset.pair;
            if (mode === 'pair' && v && v.pair === pair) {{
                clear();
            }} else {{
                mode = 'pair';
                v = {{entity: ent, pair: pair}};
                apply();
            }}
        }}
        e.stopPropagation();
    }}));

    document.addEventListener('click', e => {{
        if (!e.target.closest('.mention')) clear();
    }});
    }})();
    </script>
    """
    display(HTML(html_block))

def interactive_parsed_docs_view(load_path: Path | str, height: int = 500) -> None:
    load_path = Path(load_path)

    # gather JSON files (single file or all in dir)
    json_files = sorted(load_path.glob("*.json")) if load_path.is_dir() else [load_path]
    if not json_files:
        raise ValueError("No JSON files found")

    # {name: parsed_json}
    parsed_dict = {
        f.stem: json.load(open(f, "r", encoding="utf-8"))
        for f in json_files
    }

    doc_dd = widgets.Dropdown(
        options=sorted(parsed_dict),
        description="Document:",
        style={"description_width": "6em"},
        layout=widgets.Layout(width="450px"),
    )
    out = widgets.Output()

    def _render(_=None):
        with out:
            clear_output(wait=True)
            parsed = parsed_dict[doc_dd.value]
            full_text = parsed["full_text"]
            split_points = parsed.get("split_points", [])

            # build chunks according to split_points
            chunks = []
            if split_points:
                # first chunk
                chunks.append(full_text[:split_points[0]])
                # middle chunks
                for i, sp in enumerate(split_points[:-1]):
                    chunks.append(full_text[sp:split_points[i + 1]])
                # last chunk
                chunks.append(full_text[split_points[-1]:])
            else:
                chunks = [full_text]

            visualize_splits_txt(chunks, full_text, height)

    doc_dd.observe(_render, names="value")
    _render()
    display(widgets.VBox([doc_dd, out]))

def interactive_chunks_view(chunks_path: str, parsed_docs_dir: str | Path, height: int = 500) -> None:
    parsed_docs_dir = Path(parsed_docs_dir)
    df = pd.read_parquet(chunks_path)
    df = df.sort_values(by=["doc_name", "method", "chunk_index"]).reset_index(drop=True)

    doc_options = sorted(df["doc_name"].unique())
    doc_dd = widgets.Dropdown(
        options=doc_options,
        description="Document:",
        style={"description_width": "6em"},
        layout=widgets.Layout(width="450px"),
    )
    method_dd = widgets.Dropdown(
        description="Method:",
        style={"description_width": "6em"},
        layout=widgets.Layout(width="450px"),
    )
    out = widgets.Output()

    def _get_full_text(doc_name: str) -> str | None:
        file_name = f"{Path(doc_name)}.json"
        doc_path = parsed_docs_dir / file_name
        if not doc_path.exists():
            print(f"Warning: JSON file not found for document '{doc_name}' at {doc_path}")
            return None
        try:
            with open(doc_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data.get("full_text")
        except (json.JSONDecodeError, OSError) as e:
            print(f"Error reading or parsing {doc_path}: {e}")
            return None

    def _refresh_methods(change=None):
        if not doc_dd.value:
            method_dd.options = []
            return
        methods = df.loc[df["doc_name"] == doc_dd.value, "method"].unique()
        method_dd.options = sorted(methods)

    def _render(_=None):
        with out:
            clear_output(wait=True)
            if not doc_dd.value or not method_dd.value:
                print("Please select a document and a method.")
                return

            subset = df[(df["doc_name"] == doc_dd.value) & (df["method"] == method_dd.value)]
            if subset.empty:
                print("No splits.")
                return
            
            full_text = _get_full_text(doc_dd.value)
            if full_text is None:
                print("Could not retrieve full text. Visualization may be incorrect.")
                # Fallback to joining splits, though it may be inaccurate
                full_text = "".join(subset["chunk_text"].tolist())

            splits = subset["chunk_text"].tolist()

            visualize_splits_txt(splits, full_text, height=height)

    doc_dd.observe(_refresh_methods, names="value")
    method_dd.observe(_render, names="value")

    _refresh_methods()
    if doc_options:
        _render()
    display(widgets.VBox([doc_dd, method_dd, out]))

def interactive_text_clusters_view(
    parsed_docs_dir: str | Path,
    mentions_dir: str | Path,
    height: int = 200):

    parsed_docs_dir = Path(parsed_docs_dir)
    mentions_dir = Path(mentions_dir)
    if not parsed_docs_dir.is_dir():
        raise FileNotFoundError(f"{parsed_docs_dir} is not a directory")
    if not mentions_dir.is_dir():
        raise FileNotFoundError(f"{mentions_dir} is not a directory")

    docs_text = {}
    for doc_file in parsed_docs_dir.glob("*.json"):
        try:
            with doc_file.open(encoding="utf-8") as f:
                data = json.load(f)
            docs_text[doc_file.name] = data["full_text"]
        except (KeyError, json.JSONDecodeError, OSError) as exc:
            print(f"Skipping {doc_file.name}: {exc}")

    if not docs_text:
        raise ValueError(
            f"No JSON files with a 'full_text' key found in {parsed_docs_dir}"
        )

    def _parquet_for(doc_name: str):
        matches = list(mentions_dir.glob(f"*{Path(doc_name).stem}.parquet"))
        return matches[0] if len(matches) == 1 else None

    available_docs = sorted(
        [d for d in docs_text if _parquet_for(d) is not None]
    )

    if not available_docs:
        raise ValueError(
            "No overlapping documents between JSON files and Parquet files."
        )

    doc_dd = widgets.Dropdown(
        options=available_docs,
        description="Document:",
        style={"description_width": "6em"},
        layout=widgets.Layout(width="450px"),
    )
    view_tb = widgets.ToggleButtons(
        options=[("Clusters", "clusters"), ("Entity-Pronoun pairs", "pairs")],
        description="View:",
        style={"description_width": "4em"},
    )
    out = widgets.Output()

    _mentions_cache = {}

    def _load_mentions_for(doc_name: str):
        if doc_name in _mentions_cache:
            return _mentions_cache[doc_name]

        pq_path = _parquet_for(doc_name)
        if pq_path is None:
            raise FileNotFoundError(f"No Parquet file found for {doc_name}")

        df = pd.read_parquet(pq_path)
        data = {
            "mentions": df.iloc[0]["mentions"] if "mentions" in df.columns else None,
            "entity_pron_mentions": df.iloc[0]["entity_pron_mentions"]
            if "entity_pron_mentions" in df.columns
            else None,
        }
        if data["mentions"] is None and data["entity_pron_mentions"] is None:
            raise ValueError(
                f"{pq_path.name} lacks both 'mentions' and 'entity_pron_mentions' columns"
            )

        _mentions_cache[doc_name] = data
        return data

    def _render(_=None):
        with out:
            clear_output(wait=True)
            doc = doc_dd.value
            if not doc:
                return

            original_text = docs_text[doc]
            try:
                data = _load_mentions_for(doc)
            except Exception as exc:
                print(exc)
                return

            if view_tb.value == "clusters":
                mentions_list = (
                    data["mentions"]
                    if data["mentions"] is not None
                    else data["entity_pron_mentions"]
                )
                visualize_text_clusters(original_text, mentions_list, height)
            else:
                if data["entity_pron_mentions"] is None:
                    print("No entity-pronoun pairs available for this document.")
                    return
                visualize_entity_pron_pairs(
                    original_text, data["entity_pron_mentions"], height
                )

    doc_dd.observe(_render, names="value")
    view_tb.observe(_render, names="value")
    _render()
    display(widgets.VBox([widgets.HBox([doc_dd, view_tb]), out]))

def interactive_length_histograms(input_path: str, bins: int = 50) -> None:
    df = pd.read_parquet(input_path)

    methods = sorted(df["method"].unique())
    method_dd = widgets.Dropdown(
        options=methods,
        description="Method:",
        layout=widgets.Layout(width="450px"),
    )
    out = widgets.Output()

    def _update(_=None):
        with out:
            clear_output(wait=True)
            method = method_dd.value
            if not method:
                return
            lengths = df.loc[df["method"] == method, "chunk_len"].to_list()
            plt.figure()
            plt.hist(lengths, bins=bins)
            plt.title(f"Chunking length distribution – {method}")
            plt.xlabel("Tokens per chunk")
            plt.ylabel("Frequency")
            plt.tight_layout()
            plt.show()

    method_dd.observe(_update, names="value")
    _update()
    display(widgets.VBox([method_dd, out]))

def infer_domain_from_filename(filename: str) -> str:
    """Infer the *domain* of a document from its *filename*.

    The current convention is that parsed JSON files are named
    ``<domain>_<anything_else>.json`` – we therefore take the substring
    preceding the first underscore (``_``). If no underscore is found we take
    the whole stem.
    """

    stem = Path(filename).with_suffix("").name
    return stem.split("_", 1)[0]

def show_corpus_statistics(
    parsed_docs_dir: str | Path,
    count_tokens_func: Callable[[str], int]):
    """Display corpus-level statistics for a directory of *parsed* JSON docs.

    The function scans *parsed_docs_dir* for ``*.json`` files produced by the
    parsing pipeline, infers the *domain* from the filename, and computes - per
    domain - the following statistics:

    • total number of documents
    • mean ± std dev tokens per *document*
    • mean ± std dev tokens per *page*

    The statistics are printed in a pretty tabular form and also returned as a
    :class:`pandas.DataFrame` with columns
    ``["domain", "num_docs", "mean_doc_tokens", "std_doc_tokens",
    "mean_page_tokens", "std_page_tokens"]``.
    """

    parsed_docs_dir = Path(parsed_docs_dir)
    if not parsed_docs_dir.exists():
        raise FileNotFoundError(f"Directory not found: {parsed_docs_dir}")

    records: list[dict] = []

    for json_path in parsed_docs_dir.glob("*.json"):
        try:
            with json_path.open("r", encoding="utf-8") as fp:
                doc = json.load(fp)
        except Exception as exc:  # noqa: BLE001
            print(f"Warning: could not read '{json_path.name}': {exc}")
            continue

        domain = infer_domain_from_filename(json_path.name)

        full_text = doc.get("full_text", "")
        doc_tokens = count_tokens_func(full_text)

        pages_dict = doc.get("pages") or {}
        # compute tokens per *page*; ensure we treat missing pages as 0 tokens
        page_tokens = [count_tokens_func(str(text)) for text in pages_dict.values()]
        if not page_tokens:
            # fallback: treat whole doc as single page
            page_tokens = [doc_tokens]

        records.append(
            {
                "domain": domain,
                "doc_tokens": doc_tokens,
                "page_tokens": page_tokens,  # list - will explode later
                "num_pages": len(pages_dict) if pages_dict else 1,
            }
        )

    if not records:
        print("No parsed JSON documents found.")
        return pd.DataFrame()

    # Build dataframe: one row per document
    df_docs = pd.DataFrame(records)

    # explode page token lists so that each row becomes a page – useful for per-page stats
    df_pages = df_docs.explode("page_tokens")

    # Aggregate per domain
    agg_doc_tokens = (
        df_docs.groupby("domain")["doc_tokens"]
        .agg(["count", "mean", "std", "max", "min"])
        .rename(
            columns={
                "count": "num_docs",
                "mean": "mean_doc_tokens",
                "std": "std_doc_tokens",
                "max": "max_doc_tokens",
                "min": "min_doc_tokens",
            }
        )
    )

    agg_doc_pages = (
        df_docs.groupby("domain")["num_pages"]
        .agg(["mean", "std"])
        .rename(columns={"mean": "mean_pages_per_doc", "std": "std_pages_per_doc"})
    )

    agg_page_tokens = (
        df_pages.groupby("domain")["page_tokens"]
        .agg(["mean", "std"])
        .rename(columns={"mean": "mean_page_tokens", "std": "std_page_tokens"})
    )

    summary = (
        agg_doc_tokens
        .join(agg_doc_pages, how="left")
        .join(agg_page_tokens, how="left")
        .reset_index()
    )

    print("Corpus statistics per domain:\n")
    # Select and order columns for clearer presentation
    columns_to_show = [
        "domain",
        "num_docs",
        "mean_doc_tokens",
        "std_doc_tokens",
        "max_doc_tokens",
        "min_doc_tokens",
        "mean_pages_per_doc",
        "std_pages_per_doc",
        "mean_page_tokens",
        "std_page_tokens",
    ]

    # Some columns may be missing (e.g., if there's only one document per domain -> std == NaN),
    # so intersect with existing columns to avoid KeyError.
    columns_present = [col for col in columns_to_show if col in summary.columns]
    summary_reordered = summary[columns_present]

    # Transpose so that metrics become rows and domains become columns
    summary_transposed = summary_reordered.set_index("domain").T.reset_index()
    summary_transposed = summary_transposed.rename(columns={"index": "metric"})

    print(
        tabulate(
            summary_transposed,
            headers="keys",
            tablefmt="simple",
            showindex=False,
            floatfmt=".0f",
        )
    )