from __future__ import annotations

from pathlib import Path
from typing import TypedDict

from pydantic import BaseModel
from langgraph.graph import StateGraph, END

from tools.pdf_tools import extract_sections_from_pdf
from tools.llm_tools import extract_header_with_llm, extract_invoices_with_llm, extract_summary_with_llm
from models.expense import PdfSections, HeaderExtraction, InvoicesExtraction, SummaryExtraction

class GraphState(TypedDict, total=False):
    """
    State:
    - pdf_path: Input
    - pdf_sections: Output des PDF-Tools (Header/Invoices/Summary-Text)
    - header_extraction: strukturierte Header-Daten aus dem LLM
    """
    pdf_path: Path
    pdf_sections: PdfSections
    header_extraction: HeaderExtraction
    invoices_extraction: InvoicesExtraction
    summary_extraction: SummaryExtraction


def extract_pdf_node(state: GraphState) -> GraphState:
    """
    Node 1:
    - nimmt pdf_path aus dem State
    - ruft dein vorhandenes PDF-Tool auf
    - schreibt die drei Text-Blöcke zurück in den State
    """
    pdf_path = state["pdf_path"]
    header_text, invoices_text, summary_text = extract_sections_from_pdf(pdf_path)

    sections = PdfSections(
        header=header_text,
        invoices=invoices_text,
        summary=summary_text,
    )

    return {
        **state,  # pdf_path bleibt drin
        "pdf_sections": sections,
    }

def extract_header_node(state: GraphState) -> GraphState:
    """
    Node 2:
    - Nimmt den Header-Text aus pdf_sections im State
    - Ruft das LLM auf, um destination, time_period_header und ticket_id zu extrahieren
    - Schreibt das Ergebnis als HeaderExtraction in den State
    """
    pdf_sections = state["pdf_sections"]
    header_text = pdf_sections.header
    invoices_text = pdf_sections.invoices
    summary_text = pdf_sections.summary

    header_result: HeaderExtraction = extract_header_with_llm(header_text)
    invoices_result: InvoicesExtraction = extract_invoices_with_llm(invoices_text)
    summary_result: SummaryExtraction = extract_summary_with_llm(summary_text)

    return {
        **state,
        "header_extraction": header_result,
        "invoices_extraction": invoices_result,
        "summary_extraction": summary_result
    }

def build_app():
    """
    Baut den LangGraph-Workflow:
    - ein Node: extract_pdf
    - danach direkt END
    """
    graph = StateGraph(GraphState)

    graph.add_node("extract_pdf", extract_pdf_node)
    graph.add_node("extract_header", extract_header_node)

    graph.set_entry_point("extract_pdf")
    graph.add_edge("extract_pdf", "extract_header")
    graph.add_edge("extract_header", END)

    return graph.compile()


def run_workflow(pdf_path: Path) -> None:
    app = build_app()

    initial_state: GraphState = {
        "pdf_path": pdf_path,
    }

    final_state: GraphState = app.invoke(initial_state)

    print("=== Finaler GraphState ===")
    print("pdf_path:", final_state)
