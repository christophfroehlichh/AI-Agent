from pathlib import Path
from typing import List, Tuple
from pypdf import PdfReader


def extract_sections_from_pdf(pdf_path: Path) -> Tuple[str, str, str]:
    reader = PdfReader(str(pdf_path))
    texts: List[str] = [page.extract_text() or "" for page in reader.pages]
    full_text = "\n".join(texts)

    lower = full_text.lower()
    invoices_idx = lower.find("invoices")
    summary_idx = lower.find("summary")

    if invoices_idx == -1 or summary_idx == -1:
        header_text = full_text.strip()
        invoices_text = ""
        summary_text = ""
    else:
        header_text = full_text[:invoices_idx].strip()
        invoices_text = full_text[invoices_idx:summary_idx].strip()
        summary_text = full_text[summary_idx:].strip()

    return header_text, invoices_text, summary_text

