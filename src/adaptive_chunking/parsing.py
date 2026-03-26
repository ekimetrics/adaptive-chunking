import json
from abc import ABC, abstractmethod
from typing import Any, Callable, ClassVar, Dict, List, Set
import pandas as pd
import os
from pathlib import Path
import re
import numpy as np
from .chunking_utils import count_tokens


class BaseParser(ABC):
    """Abstract base class for document parsers.

    All parsers must produce JSON output with this structure:
    {
        "document_name": str,
        "pages": {page_num: "markdown content"},
        "full_text": str,
        "split_points": [int],
        "titles": [{"title": str, "start": int, "end": int, "level": int}]
    }
    """

    @abstractmethod
    def parse_docs_in_dir(
        self,
        input_dir: Path | str,
        output_dir: Path | str,
        overwrite_outputs: bool = False,
    ) -> list[dict]:
        """Parse raw documents and save intermediate results."""

    @abstractmethod
    def convert_raw_results_to_markdown(
        self,
        raw_input_dir: Path | str,
        output_dir: Path | str,
    ) -> list[dict]:
        """Convert parsed results to standard JSON format."""


class AzureDIParser(BaseParser):
    def __init__(self,
                 endpoint: str,
                 key: str,
                 count_tokens_func: Callable[[str], int]= count_tokens,
                 max_tokens_per_block: int = 1000):
        try:
            from azure.core.credentials import AzureKeyCredential
            from azure.ai.documentintelligence import DocumentIntelligenceClient
        except ImportError:
            raise ImportError(
                "Azure Document Intelligence SDK is required for AzureDIParser. "
                "Install with: pip install adaptive-chunking[parsing]"
            )
        self.client = DocumentIntelligenceClient(
            endpoint=endpoint,
            credential=AzureKeyCredential(key)
        )
        self.count_tokens_func = count_tokens_func
        self.max_tokens_per_block = max_tokens_per_block

    def _df_to_markdown(self, df: pd.DataFrame) -> str:
        """Convert a pandas DataFrame to html string and then to markdown."""
        from markdownify import markdownify
        html = df.to_html(index=False, na_rep="")
        return markdownify(html, heading_style="ATX")

    def _table_to_markdown(self, table) -> list[str]:
        """Convert an Azure DocumentTable into one or more Markdown strings that each
        respect `self.max_tokens_per_block`."""

        df = pd.DataFrame(index=range(table.row_count), columns=range(table.column_count))
        for cell in table.cells:
            df.iat[cell.row_index, cell.column_index] = cell.content

        full_md = self._df_to_markdown(df)
        if self.count_tokens_func(full_md) <= self.max_tokens_per_block:
            return [full_md]

        sub_mds: list[str] = []
        rows: list[tuple] = []
        for row in df.itertuples(index=False, name=None):
            candidate_rows = rows + [row]
            candidate_md = self._df_to_markdown(pd.DataFrame(candidate_rows, columns=df.columns))
            if self.count_tokens_func(candidate_md) > self.max_tokens_per_block:
                if rows:
                    part_df = pd.DataFrame(rows, columns=df.columns)
                    sub_mds.append(self._df_to_markdown(part_df))
                rows = [row]
                single_md = self._df_to_markdown(pd.DataFrame(rows, columns=df.columns))
                if self.count_tokens_func(single_md) > self.max_tokens_per_block:
                    sub_mds.append(single_md)
                    rows = []
            else:
                rows.append(row)
        if rows:
            part_df = pd.DataFrame(rows, columns=df.columns)
            sub_mds.append(self._df_to_markdown(part_df))
        return sub_mds

    def _resolve_section_ref(self, result, ref: str) -> Any:
        """Given a string reference like "/sections/0" from `result.sections`,
        return the corresponding object from `result`."""

        pattern = re.compile(r"^/(\w+)/(\d+)$") # match strings like "/<letters_or_digits>/<digits>"
        match = pattern.match(ref)
        if not match:
            return None
        collection, idx = match.groups() # returns ("sections", "0") for "/sections/0"
        try:
            return getattr(result, collection, [])[int(idx)]
        except (IndexError, ValueError):
            return None

    def _emit_block(self, blocks: list[dict], seen: set[int], order_state: dict, obj_id: int, payload: dict) -> None:
        """Append payload dict to blocks ensuring uniqueness and global document order."""

        if obj_id in seen:
            return
        seen.add(obj_id)
        payload["_order"] = order_state["order"]
        order_state["order"] += 1
        blocks.append(payload)

    def _walk_section(self, section, depth: int, result, blocks: list[dict], seen: set[int], order_state: dict) -> None:
        """Depth-first traversal of the document section tree.
        Recursive function that emits blocks for each paragraph, table, figure, and section."""

        sid = id(section)
        if sid in seen:
            return
        seen.add(sid)

        from azure.ai.documentintelligence.models import (
            DocumentParagraph, DocumentTable, DocumentFigure, DocumentSection,
        )
        for ref in section.elements:
            obj = self._resolve_section_ref(result, ref)
            if obj is None:
                continue
            if isinstance(obj, DocumentParagraph):
                role = obj.role.name if obj.role else "TEXT" # default to TEXT if the paragraph has no role
                page_number = obj.bounding_regions[0].page_number
                self._emit_block(blocks, seen, order_state, id(obj), {
                    "type": "TEXT",
                    "role": role,
                    "content": obj.content,
                    "page_number": int(page_number),
                    "depth": depth,
                })
            elif isinstance(obj, DocumentTable):
                md_list = self._table_to_markdown(obj)
                caption = obj.caption.content if obj.caption else ""
                if caption:
                    md_list = [md + ("\n" + caption) for md in md_list]
                page_number = obj.bounding_regions[0].page_number
                self._emit_block(blocks, seen, order_state, id(obj), {
                    "type": "TABLE",
                    "role": "TABLE",
                    "content": md_list,
                    "page_number": int(page_number),
                    "depth": depth,
                })
            elif isinstance(obj, DocumentFigure):
                caption = obj.caption.content if obj.caption else None
                if caption:
                    page_number = obj.bounding_regions[0].page_number
                    self._emit_block(blocks, seen, order_state, id(obj), {
                        "type": "FIGURE",
                        "role": "FIGURE",
                        "content": caption,
                        "page_number": int(page_number),
                        "depth": depth,
                    })
            elif isinstance(obj, DocumentSection):
                self._walk_section(obj, depth + 1, result, blocks, seen, order_state)

    @staticmethod
    def _get_special_sort_order(role: str) -> int:
        """Get numeric order used for header/body/footer sorting."""
        if role == "PAGE_HEADER":
            return 0
        if role == "PAGE_FOOTER":
            return 2
        if role == "PAGE_NUMBER":
            return 3
        return 1

    def _extract_blocks(self, result) -> list[dict]:
        """Extract a flat list of blocks from AnalyzeResult."""
        
        blocks: list[dict] = []
        seen: set[int] = set()
        order_state = {"order": 0}
        referenced_ids = {
            id(self._resolve_section_ref(result, ref))
            for section in (result.sections or [])
            for ref in section.elements
            if ref.startswith("/sections/") and self._resolve_section_ref(result, ref) is not None
        }
        root_sections = [section for section in (result.sections or []) if id(section) not in referenced_ids]

        # emit blocks for each section in result.sections
        for root in root_sections:
            self._walk_section(root, 1, result, blocks, seen, order_state)
        
        # emit page header, page footer, and page number blocks that are not in sections
        for para in (result.paragraphs or []):
            if not para.role:
                continue
            role_name = para.role.name
            if role_name not in ("PAGE_HEADER", "PAGE_FOOTER", "PAGE_NUMBER"):
                continue
            page_number = para.bounding_regions[0].page_number
            self._emit_block(blocks, seen, order_state, id(para), {
                "type": "TEXT",
                "role": role_name,
                "content": para.content,
                "page_number": int(page_number),
                "depth": 0,
            })

        # sort blocks by page number, role, and order to ensure global document order
        blocks.sort(key=lambda blk: (blk["page_number"], self._get_special_sort_order(blk["role"]), blk["_order"]))
        for blk in blocks:
            blk.pop("_order", None)  # remove the order field

        # add an explicit PAGE_BREAK block after the last element of each page
        blocks_with_pagebreaks: list[dict] = []
        for idx, blk in enumerate(blocks):
            blocks_with_pagebreaks.append(blk)
            is_last = idx == len(blocks) - 1
            next_page = None if is_last else blocks[idx + 1]["page_number"]
            if is_last or next_page != blk["page_number"]:
                blocks_with_pagebreaks.append(
                    {
                        "type": "PAGE_BREAK",
                        "role": "PAGE_BREAK",
                        "content": None,
                        "page_number": blk["page_number"],
                        "depth": 0,
                    }
                )

        return blocks_with_pagebreaks

    def parse_docs_in_dir(
        self,
        input_dir: Path | str,
        raw_output_dir: Path | str,
        overwrite_outputs: bool = False) -> list[dict]:

        input_dir, raw_output_dir = Path(input_dir), Path(raw_output_dir)
        raw_output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Parsing docs from {input_dir}...")

        # create doc_paths
        current_parsed_file_paths = set(raw_output_dir.rglob("*.json"))

        doc_paths = {}
        for doc_path in input_dir.rglob("*.pdf"): # filter to parse only pdfs
            if not doc_path.is_file():
                continue
            doc_name = doc_path.with_suffix('').name
            save_path = raw_output_dir / f"adi_output_{doc_path.parent.name}_{doc_name + '.json'}"

            if not overwrite_outputs and save_path in current_parsed_file_paths:
                print(f"Skipping {doc_name} parsing, overwrite_outputs is False")
                continue

            doc_paths[doc_name] = doc_path
        
        # main loop: parse each doc and save the result to raw_output_dir
        for doc_name, doc_path in doc_paths.items():
            print(f"Parsing {doc_name}")

            try:
                from azure.ai.documentintelligence.models import (
                    AnalyzeDocumentRequest, ContentFormat,
                )
                with open(doc_path, "rb") as fp:
                    poller = self.client.begin_analyze_document(
                        "prebuilt-layout",
                        AnalyzeDocumentRequest(bytes_source=fp.read()),
                        output_content_format=ContentFormat.MARKDOWN
                        ) # support for DocumentFormulas can be added, but it should raise costs

                result = poller.result()
            except Exception as e:
                print(f"Failed to parse {doc_name}: {e}")
                continue

            save_name = f"adi_output_{doc_path.parent.name}_{doc_name + '.json'}"
            save_path = raw_output_dir / f"{save_name}"

            with open(save_path, "w", encoding="utf-8") as fp:
                json.dump(result.as_dict(), fp, ensure_ascii=False, indent=2)

        print(f"\nRaw ADI outputs saved to {raw_output_dir}")

    def convert_raw_results_to_markdown(
        self,
        raw_input_dir: Path | str,
        output_dir: Path | str) -> list[dict]:
        
        raw_input_dir, output_dir = Path(raw_input_dir), Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Reading ADI outputs from {raw_input_dir}...")

        extracted_texts: list[dict] = []

        for doc_path in raw_input_dir.rglob("*.json"):
            if not doc_path.is_file():
                continue
            
            doc_name = doc_path.with_suffix('').name.replace("adi_output_", "")

            print(f"Converting {doc_name}")

            from azure.ai.documentintelligence.models import AnalyzeResult
            with open(doc_path, encoding="utf-8") as fp:
                result: AnalyzeResult = AnalyzeResult(json.load(fp))

            # perform a DFS over the sections attribute to get the document structure
            # append extra blocks for page header, page footer, and page number
            blocks = self._extract_blocks(result)

            if not blocks:
                print("No parsable content.")
                continue

            # Build pages_content , full_text, split_points
            pages_content: dict[int, str] = {}
            split_points: list[int] = []
            current_split = 0
            titles: list[dict] = []  # accumulate heading metadata
            title_level = 0

            # main loop for markdown generation
            i = 0
            main_title_found = False

            while i < len(blocks):
                blk = blocks[i]
                role = blk["role"]
                page_number = blk["page_number"]
                content = blk["content"]

                # merge PAGE_HEADER sequence
                if role == "PAGE_HEADER":
                    parts = [content]
                    j = i + 1
                    while (
                        j < len(blocks)
                        and blocks[j]["role"] == "PAGE_HEADER"
                        and blocks[j]["page_number"] == page_number
                    ):
                        parts.append(blocks[j]["content"].strip())
                        j += 1
                    content = " ".join(parts)
                    block_text = f"<!-- PageHeader: {content} -->\n\n"
                    i = j  # jump past the entire run

                # merge PAGE_FOOTER + PAGE_NUMBER sequence
                elif role in ("PAGE_FOOTER", "PAGE_NUMBER"):
                    footer_parts, number_parts = [], []
                    j = i
                    while (
                        j < len(blocks)
                        and blocks[j]["role"] in ("PAGE_FOOTER", "PAGE_NUMBER")
                        and blocks[j]["page_number"] == page_number
                    ):
                        if blocks[j]["role"] == "PAGE_FOOTER":
                            footer_parts.append(blocks[j]["content"])
                        else:
                            number_parts.append("page " + blocks[j]["content"])
                        j += 1
                    content = " ".join(footer_parts + number_parts)
                    block_text = f"<!-- PageFooter: {content} -->\n"
                    i = j  # jump past the entire run

                # default cases
                else:
                    if role == "TABLE":
                        table_markdowns = content if isinstance(content, list) else [content]
                        for k, tbl_md in enumerate(table_markdowns):
                            tbl_block = f"<Table>\n{tbl_md}\n</Table>\n\n"
                            pages_content.setdefault(page_number, "")
                            pages_content[page_number] += tbl_block
                            current_split += len(tbl_block)
                            # add split point after each table chunk except the very last one
                            is_last_tbl = (k == len(table_markdowns) - 1)
                            is_last_block = (i + 1 >= len(blocks)) and is_last_tbl
                            if not is_last_block:
                                split_points.append(current_split)
                        i += 1  # advance to next block after processing the table
                        continue  # move to "while" top; skip generic handling below
                    elif role == "FIGURE":
                        block_text = f"<Figure>\n{content}\n</Figure>\n\n"
                    elif role == "FORMULA_BLOCK":
                        block_text = f"<Formula>\n{content}\n</Formula>\n\n"
                    elif role in ("TITLE", "SECTION_HEADING"):
                        if blk["depth"] == 1:
                            main_title_found = True
                        title_level = blk["depth"] if main_title_found else max(1, blk["depth"] - 1)
                        start_offset = current_split  # record where this title starts
                        block_text = "#" * title_level + " " + content + "\n\n"
                        # store title metadata (span end to be filled later)
                        titles.append({
                            "title": block_text.strip(),
                            "start": start_offset,
                            "level": title_level,
                        })
                    elif role == "FOOTNOTE":
                        block_text = r"\* " + content + "\n\n"
                    elif role == "PAGE_BREAK":
                        block_text = "<!-- PageBreak -->\n\n"
                    else:  # TEXT and anything else
                        block_text = content + "\n\n"
                    i += 1  # advance one position for the non‑merged case

                # write out and maintain state 
                pages_content.setdefault(page_number, "")
                pages_content[page_number] += block_text
                current_split += len(block_text)

                # split‑point rules (unchanged except for using i after update)
                is_last = (i >= len(blocks))
                add_split = True
                if not is_last:
                    nxt_role = blocks[i]["role"]
                    if role in ("TITLE", "SECTION_HEADING"):
                        add_split = False
                    elif role == "TEXT" and self.count_tokens_func(content) < 100:
                        if nxt_role == "TEXT" and self.count_tokens_func(blocks[i]["content"]) < 100:
                            add_split = False
                    if nxt_role == "FOOTNOTE":
                        add_split = False
                if add_split and not is_last:
                    split_points.append(current_split)

            # ensure we add entries for any missing (empty) pages – including empty
            total_pages_detected = 0
            if hasattr(result, "pages") and result.pages is not None:
                try:
                    total_pages_detected = len(result.pages)
                except TypeError:
                    total_pages_detected = 0

            total_pages_observed = max(pages_content.keys(), default=0)
            total_pages = max(total_pages_detected, total_pages_observed)

            for pn in range(1, total_pages + 1):
                pages_content.setdefault(pn, "")

            # sort page contents and create full_text string
            pages_content = dict(sorted(pages_content.items()))
            full_text = "".join(pages_content.values())

            # compute end offsets for titles
            for idx, t in enumerate(titles):
                level = t["level"]
                start = t["start"]
                end = len(full_text)
                for jdx in range(idx + 1, len(titles)):
                    if titles[jdx]["level"] <= level:
                        end = titles[jdx]["start"]
                        break
                t["end"] = end

            # Write output parsed doc as json
            out_data = {
                "document_name": doc_name,
                "pages": pages_content,
                "full_text": full_text,
                "split_points": split_points,
                "titles": titles
            }
            extracted_texts.append(out_data)

            save_name = f"{doc_name + '.json'}"
            save_path = output_dir / f"{save_name}"
            
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(out_data, f, indent=2)

        print(f"\nOutputs saved to {output_dir}")
        
        return extracted_texts

class ExcelParser(BaseParser):
    def __init__(self,
                 max_tokens_per_block: int = 1100,
                 count_tokens_func: Callable[[str], int] = count_tokens):
        self.max_tokens_per_block = max_tokens_per_block
        self.count_tokens_func = count_tokens_func

    def _df_to_markdown(self, df: pd.DataFrame, title: str) -> str:
        """Converts a DataFrame into GitHub-flavoured Markdown with a level-1 heading."""
        from markdownify import markdownify
        html = df.to_html(index=False, na_rep="")
        return f"# {title}\n" + markdownify(html, heading_style="ATX")

    def _clean_headers(self, raw_headers: list[str]):
        """Replace blanks/dupes with safe names (col_1, col_2, …)."""
        safe: list[str] = []
        seen: set[str] = set()
        for i, h in enumerate(raw_headers, start=1):
            # Treat any NaN / blank-like value as missing
            if pd.isna(h) or str(h).strip() == "" or str(h).strip().lower() in {"nan", "none"}:
                name = f"col_{i}"
            else:
                name = str(h).strip()

            # ensure uniqueness
            if name in seen:
                j = 1
                while f"{name}_{j}" in seen:
                    j += 1
                name = f"{name}_{j}"
            safe.append(name)
            seen.add(name)
        return safe

    def _split_by_rows(self, data_df: pd.DataFrame, sheet_name: str, part_start: int = 1) -> list[str]:
        """
        Split *data_df* into markdown chunks whose token length is <= *max_tokens_per_block*.
        Each returned chunk already contains the header row and a trailing blank line.  
        The chunk titles follow the pattern "{sheet_name} - part {n}".
        """
        sub_mds: list[str] = []
        rows: list[tuple] = []
        part = part_start

        def rows_to_md(_rows: list[tuple]) -> str:
            df_part = pd.DataFrame(_rows, columns=data_df.columns)
            return self._df_to_markdown(df_part, f"{sheet_name} - part {part}") + "\n\n"

        for row in data_df.itertuples(index=False, name=None):
            candidate_rows = rows + [row]  # tentatively add this row
            candidate_md = self._df_to_markdown(pd.DataFrame(candidate_rows, columns=data_df.columns),
                                               f"{sheet_name} - part {part}") + "\n\n"
            if self.count_tokens_func(candidate_md) > self.max_tokens_per_block:
                if rows:  # flush the chunk built so far
                    md = rows_to_md(rows)
                    sub_mds.append(md)
                    part += 1
                # start new chunk with current row
                rows = [row]
                single_md = rows_to_md(rows)
                if self.count_tokens_func(single_md) > self.max_tokens_per_block:
                    # emit the single row even if it alone exceeds the limit
                    sub_mds.append(single_md)
                    part += 1
                    rows = []
            else:
                rows.append(row)

        if rows:  # flush remaining rows
            md = rows_to_md(rows)
            sub_mds.append(md)

        return sub_mds

    def _parse_xlsx_to_markdown(self, doc_path: str):
        """
        Convert every sheet in `doc_path` to Markdown.
        """
        with open(doc_path, "rb") as fp:
            sheets = pd.read_excel(fp, sheet_name=None, engine="openpyxl")

        pages_content: dict[str, str] = {}
        split_points:  list[int] = []
        offset = 0
        page_num = 1
        titles: list[dict] = []

        for sheet_name, df in sheets.items():
            # split on blank rows
            df = df.replace(r"^\s*$", np.nan, regex=True)
            blank_rows = df.index[df.isna().all(axis=1)].tolist()
            split_rows = [-1] + blank_rows + [len(df)]

            sheet_chunks: list[str] = []
            part = 1

            for start, end in zip(split_rows, split_rows[1:]):
                block = df.iloc[start + 1 : end]
                if block.empty:
                    continue

                # promote first block row to header
                raw_header = block.iloc[0]
                headers = self._clean_headers(raw_header)
                data_block = block.iloc[1:].copy()
                data_block = data_block.iloc[:, : len(headers)]
                data_block.columns = headers
                if data_block.empty:
                    continue

                # build markdown, split block further if > max_tokens
                first_md = self._df_to_markdown(data_block, f"{sheet_name} - part {part}") + "\n\n"
                if self.count_tokens_func(first_md) <= self.max_tokens_per_block:
                    # record heading metadata before mutating offset
                    heading_end = first_md.find("\n") + 1
                    title_str = first_md[:heading_end]
                    md_lvl = title_str.count("#", 0, heading_end - 1)
                    titles.append({"title": title_str.strip(), "start": offset, "level": md_lvl})

                    sheet_chunks.append(first_md)
                    part += 1
                    offset += len(first_md)
                    split_points.append(offset)
                else:
                    for sub_md in self._split_by_rows(data_block, sheet_name, part_start=part):
                        heading_end = sub_md.find("\n") + 1
                        title_str = sub_md[:heading_end]
                        md_lvl = title_str.count("#", 0, heading_end - 1)
                        titles.append({"title": title_str.strip(), "start": offset, "level": md_lvl})
                        sheet_chunks.append(sub_md)
                        part += 1
                        offset += len(sub_md)
                        split_points.append(offset)

            # store per-sheet result
            if sheet_chunks:
                pagebreak = "<!-- PageBreak -->\n\n"
                sheet_md = "".join(sheet_chunks) + pagebreak

                # update offset to include the pagebreak characters
                offset += len(pagebreak)
                # treat the page break itself as a split point
                split_points.append(offset)

                pages_content[page_num] = sheet_md
                page_num += 1

        # Join pages in order of their page number to preserve original layout
        full_text = "".join(pages_content[k] for k in sorted(pages_content))

        # compute end offsets for titles
        for idx, t in enumerate(titles):
            level = t["level"]
            start_pos = t["start"]
            end_pos = len(full_text)
            for j in range(idx + 1, len(titles)):
                if titles[j]["level"] <= level:
                    end_pos = titles[j]["start"]
                    break
            t["end"] = end_pos

        split_points.pop()  # drop last point == len(full_text)

        out_data = {
            "pages": pages_content,
            "full_text": full_text,
            "split_points": split_points,
            "titles": titles,
        }
        return out_data

    def parse_docs_in_dir(
        self,
        input_dir: Path | str,
        output_dir: Path | str):
        
        input_dir, output_dir = Path(input_dir), Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Parsing XLSX docs from {input_dir}...")

        # find doc paths
        doc_paths = {}
        for doc_path in input_dir.rglob("*.xlsx"): # filter to parse only xlsx
            if not doc_path.is_file():
                continue
            doc_name = doc_path.with_suffix('').name
            doc_paths[doc_name] = doc_path
        
        # parse and save to filesystem
        for doc_name, doc_path in doc_paths.items():
            print(f"Parsing {doc_name}")
            try:
                doc_data = self._parse_xlsx_to_markdown(doc_path = doc_path)
            except Exception as e:
                print(f"Failed to parse {doc_name}: {e}")
                continue
            doc_data["document_name"] = doc_name

            save_name = f"{doc_path.parent.name}_{doc_name + '.json'}"
            save_path = output_dir / f"{save_name}"

            with open(save_path, "w", encoding="utf-8") as fp:
                json.dump(doc_data, fp, indent=2)

    def convert_raw_results_to_markdown(
        self,
        raw_input_dir: Path | str,
        output_dir: Path | str,
    ) -> list[dict]:
        """ExcelParser produces final output directly in parse_docs_in_dir.
        This method is a no-op that returns existing parsed JSON files."""
        output_dir = Path(output_dir)
        results = []
        for json_path in output_dir.rglob("*.json"):
            with open(json_path, encoding="utf-8") as fp:
                results.append(json.load(fp))
        return results


class DoclingParser(BaseParser):
    """Local open-source PDF parser using IBM Docling."""

    def __init__(self,
                 count_tokens_func: Callable[[str], int] = count_tokens,
                 max_tokens_per_block: int = 1000):
        try:
            from docling.document_converter import DocumentConverter
        except ImportError:
            raise ImportError(
                "docling is required for DoclingParser. "
                "Install with: pip install adaptive-chunking[parsing]"
            )
        self.converter = DocumentConverter()
        self.count_tokens_func = count_tokens_func
        self.max_tokens_per_block = max_tokens_per_block

    def parse_docs_in_dir(
        self,
        input_dir: Path | str,
        output_dir: Path | str,
        overwrite_outputs: bool = False,
    ) -> list[dict]:
        input_dir, output_dir = Path(input_dir), Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Parsing docs with Docling from {input_dir}...")

        current_output_files = {p.stem for p in output_dir.rglob("*.json")}

        for doc_path in sorted(input_dir.rglob("*.pdf")):
            if not doc_path.is_file():
                continue
            doc_name = doc_path.with_suffix("").name
            save_name = f"docling_output_{doc_path.parent.name}_{doc_name}"

            if not overwrite_outputs and save_name in current_output_files:
                print(f"Skipping {doc_name} parsing, overwrite_outputs is False")
                continue

            print(f"Parsing {doc_name}")
            try:
                conv_result = self.converter.convert(str(doc_path))
            except Exception as e:
                print(f"Failed to parse {doc_name}: {e}")
                continue

            # Save raw docling result as JSON
            raw_dict = conv_result.document.export_to_dict()
            save_path = output_dir / f"{save_name}.json"
            with open(save_path, "w", encoding="utf-8") as fp:
                json.dump(raw_dict, fp, ensure_ascii=False, indent=2)

        print(f"\nRaw Docling outputs saved to {output_dir}")

    def convert_raw_results_to_markdown(
        self,
        raw_input_dir: Path | str,
        output_dir: Path | str,
    ) -> list[dict]:
        from docling_core.types.doc import DoclingDocument as DLDocument
        from docling_core.types.doc import TextItem, TableItem

        raw_input_dir, output_dir = Path(raw_input_dir), Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Reading Docling outputs from {raw_input_dir}...")

        extracted_texts: list[dict] = []

        for doc_path in sorted(raw_input_dir.rglob("*.json")):
            if not doc_path.is_file():
                continue

            doc_name = doc_path.with_suffix("").name.replace("docling_output_", "")
            print(f"Converting {doc_name}")

            with open(doc_path, encoding="utf-8") as fp:
                raw_dict = json.load(fp)

            dl_doc = DLDocument.model_validate(raw_dict)

            # Build pages_content, split_points, titles by iterating items
            pages_content: dict[int, str] = {}
            split_points: list[int] = []
            titles: list[dict] = []
            current_offset = 0

            prev_item_type = None
            for item, level in dl_doc.iterate_items():
                # Determine page number from provenance
                page_number = 1
                if hasattr(item, "prov") and item.prov:
                    page_number = item.prov[0].page_no

                if isinstance(item, TableItem):
                    # Export table as markdown
                    try:
                        table_df = item.export_to_dataframe(doc=dl_doc)
                        table_md = table_df.to_markdown(index=False)
                    except Exception:
                        table_md = item.text if hasattr(item, "text") else ""

                    # Split large tables
                    table_blocks = self._split_table_markdown(table_md)
                    for k, tbl_md in enumerate(table_blocks):
                        tbl_block = f"<Table>\n{tbl_md}\n</Table>\n\n"
                        pages_content.setdefault(page_number, "")
                        pages_content[page_number] += tbl_block
                        current_offset += len(tbl_block)
                        is_last_tbl = k == len(table_blocks) - 1
                        if is_last_tbl:
                            split_points.append(current_offset)
                    prev_item_type = "TABLE"
                    continue

                elif isinstance(item, TextItem):
                    text = item.text
                    label = item.label.value if hasattr(item.label, "value") else str(item.label)

                    if label in ("section_header", "title"):
                        heading_level = max(1, level)
                        block_text = "#" * heading_level + " " + text + "\n\n"
                        titles.append({
                            "title": block_text.strip(),
                            "start": current_offset,
                            "level": heading_level,
                        })
                        prev_item_type = "HEADING"
                    elif label == "caption":
                        block_text = f"<Figure>\n{text}\n</Figure>\n\n"
                        prev_item_type = "FIGURE"
                    elif label == "page_header":
                        block_text = f"<!-- PageHeader: {text} -->\n\n"
                        prev_item_type = "PAGE_HEADER"
                    elif label == "page_footer":
                        block_text = f"<!-- PageFooter: {text} -->\n\n"
                        prev_item_type = "PAGE_FOOTER"
                    elif label == "footnote":
                        block_text = r"\* " + text + "\n\n"
                        prev_item_type = "FOOTNOTE"
                    elif label == "formula":
                        block_text = f"<Formula>\n{text}\n</Formula>\n\n"
                        prev_item_type = "FORMULA"
                    else:
                        block_text = text + "\n\n"
                        prev_item_type = "TEXT"
                else:
                    # PictureItem or other - skip unless it has a caption
                    if hasattr(item, "caption") and item.caption:
                        cap_text = item.caption if isinstance(item.caption, str) else str(item.caption)
                        block_text = f"<Figure>\n{cap_text}\n</Figure>\n\n"
                        prev_item_type = "FIGURE"
                    else:
                        continue

                pages_content.setdefault(page_number, "")
                pages_content[page_number] += block_text
                current_offset += len(block_text)

                # Split-point rules (mirroring AzureDIParser logic)
                add_split = True
                if prev_item_type == "HEADING":
                    add_split = False
                elif prev_item_type == "FOOTNOTE":
                    add_split = False
                elif prev_item_type == "TEXT" and self.count_tokens_func(text) < 100:
                    add_split = False

                if add_split:
                    split_points.append(current_offset)

            # Add page breaks between pages
            if pages_content:
                sorted_pages = sorted(pages_content.keys())
                for pn in sorted_pages[:-1]:
                    pages_content[pn] += "<!-- PageBreak -->\n\n"

                # Rebuild with page breaks included in offsets
                pages_content_final: dict[int, str] = {}
                final_offset = 0
                for pn in sorted_pages:
                    pages_content_final[pn] = pages_content[pn]
                    final_offset += len(pages_content[pn])

                pages_content = pages_content_final

            # Build full_text
            full_text = "".join(pages_content[k] for k in sorted(pages_content))

            # Compute end offsets for titles
            for idx, t in enumerate(titles):
                level_val = t["level"]
                end = len(full_text)
                for jdx in range(idx + 1, len(titles)):
                    if titles[jdx]["level"] <= level_val:
                        end = titles[jdx]["start"]
                        break
                t["end"] = end

            out_data = {
                "document_name": doc_name,
                "pages": pages_content,
                "full_text": full_text,
                "split_points": split_points,
                "titles": titles,
            }
            extracted_texts.append(out_data)

            save_path = output_dir / f"{doc_name}.json"
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(out_data, f, indent=2)

        print(f"\nOutputs saved to {output_dir}")
        return extracted_texts

    def _split_table_markdown(self, table_md: str) -> list[str]:
        """Split a large table markdown into chunks respecting max_tokens_per_block."""
        if self.count_tokens_func(table_md) <= self.max_tokens_per_block:
            return [table_md]

        lines = table_md.split("\n")
        if len(lines) < 3:
            return [table_md]

        # First two lines are header + separator
        header = lines[0] + "\n" + lines[1]
        data_lines = lines[2:]

        sub_mds: list[str] = []
        current_lines: list[str] = []

        for line in data_lines:
            candidate = header + "\n" + "\n".join(current_lines + [line])
            if self.count_tokens_func(candidate) > self.max_tokens_per_block and current_lines:
                sub_mds.append(header + "\n" + "\n".join(current_lines))
                current_lines = [line]
            else:
                current_lines.append(line)

        if current_lines:
            sub_mds.append(header + "\n" + "\n".join(current_lines))

        return sub_mds


class PyMuPDFParser(BaseParser):
    """Lightweight local PDF parser using PyMuPDF4LLM."""

    def __init__(self,
                 count_tokens_func: Callable[[str], int] = count_tokens,
                 max_tokens_per_block: int = 1000):
        try:
            import pymupdf4llm  # noqa: F401
        except ImportError:
            raise ImportError(
                "pymupdf4llm is required for PyMuPDFParser. "
                "Install with: pip install adaptive-chunking[parsing]"
            )
        self.count_tokens_func = count_tokens_func
        self.max_tokens_per_block = max_tokens_per_block

    def parse_docs_in_dir(
        self,
        input_dir: Path | str,
        output_dir: Path | str,
        overwrite_outputs: bool = False,
    ) -> list[dict]:
        import pymupdf4llm

        input_dir, output_dir = Path(input_dir), Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Parsing docs with PyMuPDF from {input_dir}...")

        current_output_files = {p.stem for p in output_dir.rglob("*.json")}

        for doc_path in sorted(input_dir.rglob("*.pdf")):
            if not doc_path.is_file():
                continue
            doc_name = doc_path.with_suffix("").name
            save_name = f"pymupdf_output_{doc_path.parent.name}_{doc_name}"

            if not overwrite_outputs and save_name in current_output_files:
                print(f"Skipping {doc_name} parsing, overwrite_outputs is False")
                continue

            print(f"Parsing {doc_name}")
            try:
                page_chunks = pymupdf4llm.to_markdown(str(doc_path), page_chunks=True)
            except Exception as e:
                print(f"Failed to parse {doc_name}: {e}")
                continue

            # Save raw page chunks as JSON
            save_path = output_dir / f"{save_name}.json"
            # page_chunks is a list of dicts with 'metadata' and 'text' keys
            serializable = []
            for chunk in page_chunks:
                entry = {"text": chunk.get("text", ""), "metadata": {}}
                meta = chunk.get("metadata", {})
                if isinstance(meta, dict):
                    entry["metadata"] = {k: v for k, v in meta.items()
                                         if isinstance(v, (str, int, float, bool, list))}
                serializable.append(entry)

            with open(save_path, "w", encoding="utf-8") as fp:
                json.dump(serializable, fp, ensure_ascii=False, indent=2)

        print(f"\nRaw PyMuPDF outputs saved to {output_dir}")

    def convert_raw_results_to_markdown(
        self,
        raw_input_dir: Path | str,
        output_dir: Path | str,
    ) -> list[dict]:
        raw_input_dir, output_dir = Path(raw_input_dir), Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Reading PyMuPDF outputs from {raw_input_dir}...")

        extracted_texts: list[dict] = []

        for doc_path in sorted(raw_input_dir.rglob("*.json")):
            if not doc_path.is_file():
                continue

            doc_name = doc_path.with_suffix("").name.replace("pymupdf_output_", "")
            print(f"Converting {doc_name}")

            with open(doc_path, encoding="utf-8") as fp:
                page_chunks = json.load(fp)

            pages_content: dict[int, str] = {}
            split_points: list[int] = []
            titles: list[dict] = []
            current_offset = 0

            for page_idx, chunk in enumerate(page_chunks):
                page_number = page_idx + 1
                page_text = chunk.get("text", "")

                if not page_text.strip():
                    pages_content[page_number] = ""
                    continue

                # Extract headings from markdown (lines starting with #)
                lines = page_text.split("\n")
                page_md = ""

                for line in lines:
                    heading_match = re.match(r"^(#{1,6})\s+(.+)", line)
                    if heading_match:
                        level = len(heading_match.group(1))
                        title_text = line.strip()
                        titles.append({
                            "title": title_text,
                            "start": current_offset + len(page_md),
                            "level": level,
                        })

                    page_md += line + "\n"

                # Add page break (except for the last page)
                if page_idx < len(page_chunks) - 1:
                    page_md += "<!-- PageBreak -->\n\n"

                pages_content[page_number] = page_md

                # Add split point at the end of each page
                current_offset += len(page_md)
                split_points.append(current_offset)

            # Remove last split point (end of document)
            if split_points:
                split_points.pop()

            # Build full_text
            full_text = "".join(pages_content[k] for k in sorted(pages_content))

            # Compute end offsets for titles
            for idx, t in enumerate(titles):
                level_val = t["level"]
                end = len(full_text)
                for jdx in range(idx + 1, len(titles)):
                    if titles[jdx]["level"] <= level_val:
                        end = titles[jdx]["start"]
                        break
                t["end"] = end

            out_data = {
                "document_name": doc_name,
                "pages": pages_content,
                "full_text": full_text,
                "split_points": split_points,
                "titles": titles,
            }
            extracted_texts.append(out_data)

            save_path = output_dir / f"{doc_name}.json"
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(out_data, f, indent=2)

        print(f"\nOutputs saved to {output_dir}")
        return extracted_texts