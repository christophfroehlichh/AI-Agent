"""
PDF text extraction utilities.
Responsible for loading PDFs and splitting them into logical sections.
"""

from pathlib import Path
from typing import List, Tuple
import logging

from pypdf import PdfReader

logger = logging.getLogger(__name__)
logging.getLogger("pypdf").setLevel(logging.ERROR)


def extract_sections_from_pdf(pdf_path: Path) -> Tuple[str, str, str]:
    """Splits a PDF into header, invoices and summary text sections."""
    reader = PdfReader(str(pdf_path))
    texts: List[str] = [page.extract_text() or "" for page in reader.pages]
    full_text = "\n".join(texts)

    lower = full_text.lower()
    invoices_idx = lower.find("invoices")
    summary_idx = lower.find("summary")

    if invoices_idx == -1 or summary_idx == -1:
        logger.warning("Could not find section headers in PDF: %s", pdf_path)
        logger.info("PDF extraction finished (fallback used).")
        return full_text.strip(), "", ""

    logger.info("PDF extraction finished successfully.")
    return (
        full_text[:invoices_idx].strip(),
        full_text[invoices_idx:summary_idx].strip(),
        full_text[summary_idx:].strip(),
    )
