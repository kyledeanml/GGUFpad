"""
GGUFpad Multi-Format Context File Loader
Extracts structured text, code, tables, document contents, and multimodal image payloads
from various file types to inject into LLM conversation context.
"""

import os
import csv
import json
import base64
import zipfile
import subprocess
import io
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple


SUPPORTED_CONTEXT_CATEGORIES = [
    (
        "All Supported Context Files",
        (
            "*.txt", "*.md", "*.markdown", "*.rst", "*.log", "*.pdf",
            "*.docx", "*.odt", "*.rtf", "*.epub",
            "*.py", "*.js", "*.ts", "*.jsx", "*.tsx", "*.c", "*.cpp", "*.h", "*.hpp",
            "*.rs", "*.go", "*.java", "*.cs", "*.sh", "*.bash", "*.sql", "*.html", "*.htm", "*.css",
            "*.json", "*.jsonl", "*.yaml", "*.yml", "*.toml", "*.ini", "*.cfg", "*.conf", "*.xml",
            "*.csv", "*.tsv", "*.xlsx", "*.ipynb", "*.srt", "*.vtt", "*.tex",
            "*.png", "*.jpg", "*.jpeg", "*.webp", "*.bmp", "*.gif", "*.svg",
        ),
    ),
    ("Documents (*.pdf, *.docx, *.odt, *.rtf, *.epub, *.txt, *.md)", ("*.pdf", "*.docx", "*.odt", "*.rtf", "*.epub", "*.txt", "*.md", "*.rst", "*.tex")),
    ("Code & Notebooks (*.py, *.js, *.ts, *.c, *.cpp, *.rs, *.go, *.sh, *.sql, *.ipynb)", ("*.py", "*.js", "*.ts", "*.jsx", "*.tsx", "*.c", "*.cpp", "*.h", "*.hpp", "*.rs", "*.go", "*.java", "*.cs", "*.sh", "*.bash", "*.sql", "*.ipynb")),
    ("Data, Configs & Tables (*.csv, *.tsv, *.xlsx, *.json, *.jsonl, *.yaml, *.toml, *.xml, *.log)", ("*.csv", "*.tsv", "*.xlsx", "*.json", "*.jsonl", "*.yaml", "*.yml", "*.toml", "*.ini", "*.cfg", "*.conf", "*.xml", "*.log")),
    ("Web & Subtitles (*.html, *.htm, *.css, *.srt, *.vtt)", ("*.html", "*.htm", "*.css", "*.srt", "*.vtt")),
    ("Images / Vision (*.png, *.jpg, *.jpeg, *.webp, *.bmp, *.gif, *.svg)", ("*.png", "*.jpg", "*.jpeg", "*.webp", "*.bmp", "*.gif", "*.svg")),
]


@dataclass
class ContextAttachment:
    """Represents a file loaded into GGUFpad's active conversation context."""
    file_path: str
    filename: str
    file_type: str  # e.g. 'PDF', 'PYTHON', 'CSV', 'DOCX', 'IMAGE', 'TEXT'
    size_bytes: int
    extracted_text: str
    summary_label: str  # e.g. '142 lines' or '3.4 KB' or '1024x768 PNG'
    image_data_url: Optional[str] = None
    truncated: bool = False

    def format_for_prompt(self) -> str:
        """Formats this attachment as a clear context block for the LLM prompt."""
        header = f"=== ATTACHED FILE: {self.filename} ({self.file_type}, {self.summary_label}) ==="
        footer = f"=== END OF FILE: {self.filename} ==="
        return f"{header}\n{self.extracted_text}\n{footer}"


class ContextFileLoader:
    """Loads and extracts context from documents, code, data files, and images."""

    DEFAULT_MAX_CHARS = 24000

    @classmethod
    def load_file(cls, file_path: str, max_chars: int = DEFAULT_MAX_CHARS) -> ContextAttachment:
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        filename = os.path.basename(file_path)
        ext = os.path.splitext(filename)[1].lower()
        size_bytes = os.path.getsize(file_path)

        # Dispatch by extension
        if ext == ".pdf":
            text, file_type, summary = cls._extract_pdf(file_path)
        elif ext == ".docx":
            text, file_type, summary = cls._extract_docx(file_path)
        elif ext == ".odt":
            text, file_type, summary = cls._extract_odt(file_path)
        elif ext == ".epub":
            text, file_type, summary = cls._extract_epub(file_path)
        elif ext == ".rtf":
            text, file_type, summary = cls._extract_rtf(file_path)
        elif ext in (".csv", ".tsv"):
            text, file_type, summary = cls._extract_csv(file_path, delimiter="\t" if ext == ".tsv" else ",")
        elif ext == ".xlsx":
            text, file_type, summary = cls._extract_xlsx(file_path)
        elif ext == ".ipynb":
            text, file_type, summary = cls._extract_ipynb(file_path)
        elif ext in (".html", ".htm"):
            text, file_type, summary = cls._extract_html(file_path)
        elif ext in (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"):
            return cls._extract_image(file_path, filename, ext, size_bytes)
        else:
            text, file_type, summary = cls._extract_text_or_code(file_path, ext, size_bytes)

        truncated = False
        if len(text) > max_chars:
            text = (
                text[:max_chars]
                + f"\n\n[... Content truncated at {max_chars:,} characters to fit model context window ...]"
            )
            truncated = True

        return ContextAttachment(
            file_path=file_path,
            filename=filename,
            file_type=file_type,
            size_bytes=size_bytes,
            extracted_text=text,
            summary_label=summary + (" · trunc" if truncated else ""),
            image_data_url=None,
            truncated=truncated,
        )

    @classmethod
    def _extract_pdf(cls, file_path: str) -> Tuple[str, str, str]:
        """Extracts text from PDF using pdftotext (Poppler) with pure-Python fallback."""
        try:
            res = subprocess.run(
                ["pdftotext", "-layout", file_path, "-"],
                capture_output=True,
                text=True,
                timeout=15,
            )
            if res.returncode == 0 and res.stdout.strip():
                text = res.stdout.strip()
                pages = max(1, res.stdout.count("\f"))
                return text, "PDF", f"{pages} page{'s' if pages != 1 else ''}"
        except Exception:
            pass

        # Fallback: extract printable text strings from PDF stream
        with open(file_path, "rb") as f:
            raw = f.read(512 * 1024)
        chunks = re.findall(rb"\(([^\(\)\\]{3,})\)", raw)
        extracted = " ".join(c.decode("latin-1", errors="ignore") for c in chunks)
        if not extracted.strip():
            extracted = "[PDF file contains scanned images or compressed streams without extractable plain text.]"
        return extracted, "PDF", f"{len(raw) / 1024:.1f} KB"

    @classmethod
    def _extract_docx(cls, file_path: str) -> Tuple[str, str, str]:
        """Extracts paragraphs and tables from a Word .docx archive."""
        paragraphs = []
        with zipfile.ZipFile(file_path, "r") as zf:
            if "word/document.xml" in zf.namelist():
                xml_bytes = zf.read("word/document.xml")
                root = ET.fromstring(xml_bytes)
                for elem in root.iter():
                    if elem.tag.endswith("}p"):
                        p_text = "".join(
                            node.text for node in elem.iter() if node.tag.endswith("}t") and node.text
                        )
                        if p_text.strip():
                            paragraphs.append(p_text.strip())
        full_text = "\n\n".join(paragraphs) if paragraphs else "[Empty .docx document]"
        return full_text, "DOCX", f"{len(paragraphs)} paragraphs"

    @classmethod
    def _extract_odt(cls, file_path: str) -> Tuple[str, str, str]:
        """Extracts text from an OpenDocument .odt archive."""
        paragraphs = []
        with zipfile.ZipFile(file_path, "r") as zf:
            if "content.xml" in zf.namelist():
                xml_bytes = zf.read("content.xml")
                root = ET.fromstring(xml_bytes)
                for elem in root.iter():
                    if elem.tag.endswith("}p") or elem.tag.endswith("}h"):
                        p_text = "".join(elem.itertext()).strip()
                        if p_text:
                            paragraphs.append(p_text)
        full_text = "\n\n".join(paragraphs) if paragraphs else "[Empty .odt document]"
        return full_text, "ODT", f"{len(paragraphs)} paragraphs"

    @classmethod
    def _extract_epub(cls, file_path: str) -> Tuple[str, str, str]:
        """Extracts readable text from EPUB XHTML chapters."""
        sections = []
        with zipfile.ZipFile(file_path, "r") as zf:
            for name in sorted(zf.namelist()):
                if name.lower().endswith((".xhtml", ".html", ".htm")):
                    raw_html = zf.read(name).decode("utf-8", errors="ignore")
                    clean = cls._strip_html(raw_html)
                    if clean.strip():
                        sections.append(clean.strip())
        full_text = "\n\n---\n\n".join(sections) if sections else "[Empty .epub file]"
        return full_text, "EPUB", f"{len(sections)} chapters"

    @classmethod
    def _extract_rtf(cls, file_path: str) -> Tuple[str, str, str]:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            raw = f.read()
        clean = re.sub(r"\\[a-z]+(-?\d+)?[ ]?", "", raw)
        clean = re.sub(r"[{}]", "", clean).strip()
        lines = [ln.strip() for ln in clean.splitlines() if ln.strip()]
        return "\n".join(lines), "RTF", f"{len(lines)} lines"

    @classmethod
    def _extract_csv(cls, file_path: str, delimiter: str = ",") -> Tuple[str, str, str]:
        rows = []
        with open(file_path, "r", encoding="utf-8", errors="replace", newline="") as f:
            reader = csv.reader(f, delimiter=delimiter)
            for idx, row in enumerate(reader):
                rows.append(row)
                if idx >= 250:
                    break

        if not rows:
            return "[Empty tabular file]", "CSV", "0 rows"

        header = rows[0]
        lines = [
            f"Columns ({len(header)}): {', '.join(header)}",
            "",
            "| " + " | ".join(header) + " |",
            "| " + " | ".join(["---"] * len(header)) + " |",
        ]
        for r in rows[1:]:
            padded = (r + [""] * len(header))[: len(header)]
            lines.append("| " + " | ".join(str(c).replace("\n", " ") for c in padded) + " |")

        return "\n".join(lines), "CSV/TSV", f"{len(rows) - 1} rows, {len(header)} cols"

    @classmethod
    def _extract_xlsx(cls, file_path: str) -> Tuple[str, str, str]:
        """Extracts shared strings and worksheet cells from an Excel .xlsx archive."""
        shared_strings = []
        rows_out = []
        with zipfile.ZipFile(file_path, "r") as zf:
            if "xl/sharedStrings.xml" in zf.namelist():
                ss_root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
                for si in ss_root.iter():
                    if si.tag.endswith("}si"):
                        shared_strings.append("".join(si.itertext()))

            sheet_files = sorted(
                n for n in zf.namelist() if n.startswith("xl/worksheets/sheet") and n.endswith(".xml")
            )
            for sf in sheet_files:
                ws_root = ET.fromstring(zf.read(sf))
                for row_el in ws_root.iter():
                    if row_el.tag.endswith("}row"):
                        vals = []
                        for c_el in row_el:
                            if c_el.tag.endswith("}c"):
                                t_attr = c_el.attrib.get("t", "")
                                v_text = ""
                                for child in c_el:
                                    if child.tag.endswith("}v") and child.text:
                                        v_text = child.text
                                    elif child.tag.endswith("}is"):
                                        v_text = "".join(child.itertext())
                                if t_attr == "s" and v_text.isdigit():
                                    idx = int(v_text)
                                    if 0 <= idx < len(shared_strings):
                                        v_text = shared_strings[idx]
                                vals.append(v_text)
                        if any(vals):
                            rows_out.append(" | ".join(vals))
                        if len(rows_out) >= 250:
                            break
        text = "\n".join(rows_out) if rows_out else "[Empty .xlsx workbook]"
        return text, "XLSX", f"{len(rows_out)} rows"

    @classmethod
    def _extract_ipynb(cls, file_path: str) -> Tuple[str, str, str]:
        """Extracts Markdown and Code cells from a Jupyter Notebook."""
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            nb = json.load(f)
        cells = nb.get("cells", [])
        blocks = []
        for idx, cell in enumerate(cells, 1):
            ctype = cell.get("cell_type", "code")
            src = "".join(cell.get("source", []))
            if not src.strip():
                continue
            if ctype == "markdown":
                blocks.append(f"### [Cell {idx} - Markdown]\n{src.strip()}")
            else:
                blocks.append(f"### [Cell {idx} - Code]\n```python\n{src.strip()}\n```")
        return "\n\n".join(blocks), "NOTEBOOK", f"{len(cells)} cells"

    @classmethod
    def _strip_html(cls, raw_html: str) -> str:
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(raw_html, "html.parser")
            for tag in soup(["script", "style", "noscript"]):
                tag.decompose()
            return soup.get_text(separator="\n", strip=True)
        except Exception:
            text = re.sub(r"<[^>]+>", " ", raw_html)
            return re.sub(r"\s+", " ", text).strip()

    @classmethod
    def _extract_html(cls, file_path: str) -> Tuple[str, str, str]:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            raw = f.read()
        clean = cls._strip_html(raw)
        lines = [ln for ln in clean.splitlines() if ln.strip()]
        return "\n".join(lines), "HTML", f"{len(lines)} lines"

    @classmethod
    def _extract_image(
        cls, file_path: str, filename: str, ext: str, size_bytes: int
    ) -> ContextAttachment:
        """Extracts image metadata and builds a base64 data URL for multimodal vision models."""
        width, height, mode, fmt = 0, 0, "RGB", ext.lstrip(".").upper()
        data_url = None
        try:
            from PIL import Image
            with Image.open(file_path) as img:
                width, height = img.size
                mode = img.mode
                fmt = img.format or fmt
                # Resize if larger than 1280px on longest side for fast vision encoding
                work_img = img.copy()
                if work_img.mode not in ("RGB", "L"):
                    work_img = work_img.convert("RGB")
                work_img.thumbnail((1280, 1280))
                buf = io.BytesIO()
                work_img.save(buf, format="JPEG", quality=85)
                b64 = base64.b64encode(buf.getvalue()).decode("ascii")
                data_url = f"data:image/jpeg;base64,{b64}"
        except Exception:
            try:
                with open(file_path, "rb") as f:
                    b64 = base64.b64encode(f.read(2 * 1024 * 1024)).decode("ascii")
                mime = "image/png" if ext == ".png" else "image/jpeg"
                data_url = f"data:{mime};base64,{b64}"
            except Exception:
                pass

        summary = f"{width}x{height} {fmt}" if width and height else f"{size_bytes / 1024:.1f} KB {fmt}"
        desc = (
            f"Image File: {filename}\n"
            f"Format: {fmt} ({mode})\n"
            f"Dimensions: {width}x{height} px\n"
            f"File Size: {size_bytes / 1024:.1f} KB\n"
            f"(Attached as multimodal vision input for VLM / GGUF inspection)"
        )
        return ContextAttachment(
            file_path=file_path,
            filename=filename,
            file_type="IMAGE",
            size_bytes=size_bytes,
            extracted_text=desc,
            summary_label=summary,
            image_data_url=data_url,
            truncated=False,
        )

    @classmethod
    def _extract_text_or_code(cls, file_path: str, ext: str, size_bytes: int) -> Tuple[str, str, str]:
        code_langs = {
            ".py": ("PYTHON", "python"),
            ".js": ("JAVASCRIPT", "javascript"),
            ".ts": ("TYPESCRIPT", "typescript"),
            ".jsx": ("JSX", "jsx"),
            ".tsx": ("TSX", "tsx"),
            ".c": ("C", "c"),
            ".cpp": ("C++", "cpp"),
            ".h": ("C-HEADER", "c"),
            ".hpp": ("CPP-HEADER", "cpp"),
            ".rs": ("RUST", "rust"),
            ".go": ("GO", "go"),
            ".java": ("JAVA", "java"),
            ".cs": ("C#", "csharp"),
            ".sh": ("SHELL", "bash"),
            ".bash": ("BASH", "bash"),
            ".sql": ("SQL", "sql"),
            ".json": ("JSON", "json"),
            ".jsonl": ("JSONL", "json"),
            ".yaml": ("YAML", "yaml"),
            ".yml": ("YAML", "yaml"),
            ".toml": ("TOML", "toml"),
            ".xml": ("XML", "xml"),
            ".svg": ("SVG", "xml"),
            ".css": ("CSS", "css"),
            ".md": ("MARKDOWN", "markdown"),
            ".markdown": ("MARKDOWN", "markdown"),
            ".tex": ("LATEX", "latex"),
        }

        with open(file_path, "rb") as f:
            raw = f.read(512 * 1024)

        # Detect if binary
        if b"\x00" in raw[:4096]:
            return (
                f"[Binary file '{os.path.basename(file_path)}' ({size_bytes:,} bytes)]",
                ext.lstrip(".").upper() or "BINARY",
                f"{size_bytes / 1024:.1f} KB",
            )

        text = raw.decode("utf-8", errors="replace")
        line_count = len(text.splitlines())
        if ext in code_langs:
            ftype, fence = code_langs[ext]
            formatted = f"```{fence}\n{text}\n```" if fence != "markdown" else text
            return formatted, ftype, f"{line_count} lines"

        ftype = ext.lstrip(".").upper() if ext else "TEXT"
        return text, ftype, f"{line_count} lines"
