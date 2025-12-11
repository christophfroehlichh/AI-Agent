from __future__ import annotations

from pathlib import Path
from typing import TypedDict, Optional, Dict

from langgraph.graph import StateGraph, START, END

from tools.pdf_tools import extract_sections_from_pdf
from tools.llm_tools import extract_header_with_llm, extract_invoices_with_llm, extract_summary_with_llm
from tools.backend_tools import get_allowances, check_ticket_exists
from tools.checks import check_total
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
    allowances: dict
    total_ok: bool
    ticket_exists: bool
    allowances_ok: bool
    ticket_data: Optional[Dict]


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
        "pdf_sections": sections
    }

def extract_data_node(state: GraphState) -> GraphState:
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
        "header_extraction": header_result,
        "invoices_extraction": invoices_result,
        "summary_extraction": summary_result
    }

def get_allowances_node(state: GraphState) -> GraphState:
    allowances = get_allowances()

    return {
        "allowances": allowances
    }

def check_ticket_exists_node(state: GraphState) -> GraphState:
    ticket_id = state["header_extraction"].ticket_id
    ticket_exists, ticket_data = check_ticket_exists(ticket_id)
    return {
        "ticket_exists": ticket_exists,
        "ticket_data": ticket_data
    }

def check_total_node(state: GraphState) -> GraphState:
    invoices_extraction = state.get("invoices_extraction")
    summary_extraction = state.get("summary_extraction")

    if invoices_extraction is None or summary_extraction is None:
        return {}  

    total_ok = check_total(
        invoices_extraction,
        summary_extraction,
    )

    return {
        "total_ok": total_ok
    }



def build_app():
    """
    Baut den LangGraph-Workflow:
    - ein Node: extract_pdf
    - danach direkt END
    """
    graph = StateGraph(GraphState)

    graph.add_node("extract_pdf", extract_pdf_node)
    graph.add_node("extract_data", extract_data_node)
    graph.add_node("get_allowances", get_allowances_node)
    graph.add_node("check_ticket_exists", check_ticket_exists_node)
    graph.add_node("check_total", check_total_node)

    # Start fächert auf:
    graph.add_edge(START, "extract_pdf")
    graph.add_edge(START, "get_allowances")

    # PDF-Flow
    graph.add_edge("extract_pdf", "extract_data")
    graph.add_edge("extract_data", "check_ticket_exists")
    graph.add_edge("extract_data", "check_total")
    graph.add_edge("check_ticket_exists", END)
    graph.add_edge("check_total", END)


    # Allowances-Flow endet direkt
    graph.add_edge("get_allowances", END)

    return graph.compile()


def run_workflow(pdf_path: Path) -> None:
    app = build_app()

    initial_state: GraphState = {
        "pdf_path": pdf_path,
    }

    final_state: GraphState = app.invoke(initial_state)

    print("=== Finaler GraphState ===")
    print("pdf_path:", final_state)
