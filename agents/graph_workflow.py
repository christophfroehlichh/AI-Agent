"""
LangGraph workflow definition for processing travel expense PDFs.
Defines the full agent pipeline from PDF extraction to approval
decision and backend ticket update using a shared graph state.
"""

from __future__ import annotations
import logging
from pathlib import Path
from typing import TypedDict, Optional, Dict
from langgraph.graph import StateGraph, START, END


from models.expense import (
    AllowanceCalculation,
    ApprovalDecision,
    DateComparsion,
    HeaderExtraction,
    InvoicesExtraction,
    PdfSections,
    RateSelection,
    SummaryExtraction,
)
from tools.backend_tools import (
    check_ticket_exists, 
    get_allowances, 
    update_ticket_status
)
from tools.checks import (
    calculate_allowance, 
    check_total, 
    compare_time_periods
)
from tools.llm_tools import (
    build_approval_decision_with_llm,
    extract_header_with_llm,
    extract_invoices_with_llm,
    extract_summary_with_llm,
    select_daily_rate_with_llm,
)
from tools.pdf_tools import extract_sections_from_pdf

logger = logging.getLogger(__name__)

class GraphState(TypedDict, total=False):
    """
    Shared state passed between LangGraph nodes.
    Each node reads required keys and appends its own results.
    """
    pdf_path: Path
    pdf_sections: PdfSections
    header_extraction: HeaderExtraction
    invoices_extraction: InvoicesExtraction
    summary_extraction: SummaryExtraction
    allowances: dict
    total_ok: bool
    ticket_exists: bool
    allowance_calculation: AllowanceCalculation
    ticket_data: Optional[Dict]
    rate_selection: RateSelection
    date_comparsion: DateComparsion
    approval_decision: ApprovalDecision


def extract_pdf_node(state: GraphState) -> GraphState:
    """Extracts header, invoice and summary text sections from the input PDF."""
    pdf_path = state.get("pdf_path")
    if pdf_path is None:
        return {}

    header_text, invoices_text, summary_text = extract_sections_from_pdf(pdf_path)

    return {
        "pdf_sections": PdfSections(
            header=header_text,
            invoices=invoices_text,
            summary=summary_text,
        )
    }


def extract_data_node(state: GraphState) -> GraphState:
    """Extracts structured header, invoices and summary data from PDF sections using LLMs."""
    pdf_sections = state.get("pdf_sections")
    if pdf_sections is None:
        return {}

    return {
        "header_extraction": extract_header_with_llm(pdf_sections.header),
        "invoices_extraction": extract_invoices_with_llm(pdf_sections.invoices),
        "summary_extraction": extract_summary_with_llm(pdf_sections.summary),
    }


def get_allowances_node(state: GraphState) -> GraphState:
    """Loads allowance rates from the backend service."""
    return {"allowances": get_allowances()}


def check_ticket_exists_node(state: GraphState) -> GraphState:
    """Checks whether the extracted ticket_id exists in the backend and returns ticket data."""
    header = state.get("header_extraction")
    ticket_id = header.ticket_id if header else None
    if not ticket_id:
        return {"ticket_exists": False, "ticket_data": None}

    ticket_exists, ticket_data = check_ticket_exists(ticket_id)
    return {
        "ticket_exists": ticket_exists,
        "ticket_data": ticket_data,
    }


def check_total_node(state: GraphState) -> GraphState:
    """Checks whether the sum of invoice amounts matches the summary total."""
    invoices = state.get("invoices_extraction")
    summary = state.get("summary_extraction")
    if invoices is None or summary is None:
        return {}

    return {"total_ok": check_total(invoices, summary)}


def select_daily_rate_node(state: GraphState) -> GraphState:
    """Selects the applicable daily allowance rate based on destination."""
    header = state.get("header_extraction")
    allowances = state.get("allowances")
    if header is None or allowances is None:
        return {}

    return {
        "rate_selection": select_daily_rate_with_llm(
            header.destination,
            allowances,
        )
    }


def compare_dates_node(state: GraphState) -> GraphState:
    """Compares header and summary travel time periods."""
    header = state.get("header_extraction")
    summary = state.get("summary_extraction")
    if header is None or summary is None:
        return {}

    return {
        "date_comparsion": compare_time_periods(
            header.time_period_header,
            summary.time_period_summary,
        )
    }


def allowance_check_node(state: GraphState) -> GraphState:
    """Calculates expected allowances and compares them with the summary."""
    date_cmp = state.get("date_comparsion")
    rate = state.get("rate_selection")
    summary = state.get("summary_extraction")
    if date_cmp is None or rate is None or summary is None:
        return {}

    return {
        "allowance_calculation": calculate_allowance(
            date_cmp,
            rate.daily_rate,
            summary.allowance,
        )
    }


def approval_decision_node(state: GraphState) -> GraphState:
    """Builds the final approval decision based on all validation results."""
    total_ok = state.get("total_ok")
    ticket_exists = state.get("ticket_exists")
    allowance_calc = state.get("allowance_calculation")
    date_cmp = state.get("date_comparsion")

    if total_ok is None or ticket_exists is None or allowance_calc is None or date_cmp is None:
        return {}

    decision = build_approval_decision_with_llm(
        total_ok,
        ticket_exists,
        allowance_calc,
        date_cmp.periods_match,
    )

    return {"approval_decision": decision}


def update_ticket_status_node(state: GraphState) -> GraphState:
    """Updates the ticket status in the backend based on the approval decision."""
    ticket_data = state.get("ticket_data")
    decision = state.get("approval_decision")
    header = state.get("header_extraction")

    ticket_id = header.ticket_id if header else None
    if ticket_id is None or decision is None or ticket_data is None:
        return {}

    update_ticket_status(ticket_id, decision, ticket_data)
    return {}


def build_app():
    """Builds and compiles the LangGraph workflow."""
    graph = StateGraph(GraphState)

    # Nodes
    graph.add_node("extract_pdf", extract_pdf_node)
    graph.add_node("extract_data", extract_data_node)
    graph.add_node("get_allowances", get_allowances_node)
    graph.add_node("check_ticket_exists", check_ticket_exists_node)
    graph.add_node("check_total", check_total_node)
    graph.add_node("select_daily_rate", select_daily_rate_node)
    graph.add_node("compare_dates", compare_dates_node)
    graph.add_node("allowance_check", allowance_check_node)
    graph.add_node("approval_decision", approval_decision_node)
    graph.add_node("update_ticket_status", update_ticket_status_node)

    # Start
    graph.add_edge(START, "extract_pdf")
    graph.add_edge(START, "get_allowances")

    # Extraction flow
    graph.add_edge("extract_pdf", "extract_data")

    # Validation + enrichment
    graph.add_edge("extract_data", "check_ticket_exists")
    graph.add_edge("extract_data", "check_total")
    graph.add_edge("extract_data", "compare_dates")
    graph.add_edge("extract_data", "select_daily_rate")
    graph.add_edge("get_allowances", "select_daily_rate")

    # Allowance check depends on dates + daily rate
    graph.add_edge("compare_dates", "allowance_check")
    graph.add_edge("select_daily_rate", "allowance_check")

    # Final decision and backend update
    graph.add_edge("check_ticket_exists", "approval_decision")
    graph.add_edge("check_total", "approval_decision")
    graph.add_edge("allowance_check", "approval_decision")
    graph.add_edge("approval_decision", "update_ticket_status")
    graph.add_edge("update_ticket_status", END)

    return graph.compile()


def run_workflow(pdf_path: Path) -> None:
    """Runs the compiled LangGraph workflow for the given PDF."""
    logger.info("Workflow started (pdf=%s).", pdf_path)
    app = build_app()
    app.invoke({"pdf_path": pdf_path})
    logger.info("Workflow finished.")

