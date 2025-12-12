"""
LangGraph workflow definition for processing travel expense PDFs.
Defines the full agent pipeline from PDF extraction to approval
decision and backend ticket update using a shared graph state.
"""

from __future__ import annotations
from pathlib import Path
from typing import TypedDict, Optional, Dict
from langgraph.graph import StateGraph, START, END

from tools.pdf_tools import extract_sections_from_pdf
from tools.llm_tools import extract_header_with_llm, extract_invoices_with_llm, extract_summary_with_llm, select_daily_rate_with_llm, build_approval_decision_with_llm
from tools.backend_tools import get_allowances, check_ticket_exists, update_ticket_status
from tools.checks import check_total, compare_time_periods_with_llm, calculate_allowance
from models.expense import PdfSections, HeaderExtraction, InvoicesExtraction, SummaryExtraction, RateSelection, DateComparsion, AllowanceCalculation, ApprovalDecision


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


def get_allowances_node(_: GraphState) -> GraphState:
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
            destination=header.destination,
            allowances=allowances,
        )
    }


def compare_dates_node(state: GraphState) -> GraphState:
    """
    Vergleicht die Zeiträume aus Header und Summary
    und schreibt ein DateComparsion-Objekt in den State.
    """
    header_extraction = state.get("header_extraction")
    summary_extraction = state.get("summary_extraction")

    # Guard: läuft erst, wenn beide vorhanden sind
    if header_extraction is None or summary_extraction is None:
        return {}

    header_time_period = header_extraction.time_period_header
    summary_time_period = summary_extraction.time_period_summary

    date_comparsion: DateComparsion = compare_time_periods_with_llm(
        header_time_period=header_time_period,
        summary_time_period=summary_time_period,
    )

    return {
        "date_comparsion": date_comparsion,
    }


def allowance_check_node(state: GraphState) -> GraphState:
    date_comparsion = state.get("date_comparsion")
    rate_selection = state.get("rate_selection")
    summary_extraction = state.get("summary_extraction")
    
    # Guard: Node nur „wirklich“ ausführen, wenn ALLES da ist
    if date_comparsion is None or rate_selection is None or summary_extraction is None:
        # Nichts ändern, nur State durchreichen
        return state

    daily_rate = rate_selection.daily_rate
    allowance_summary = summary_extraction.allowance  # Feldname aus SummaryExtraction

    result = calculate_allowance(date_comparsion, daily_rate, allowance_summary)

    return {
        "allowance_calculation": result
    }

def approval_decision_node(state: GraphState) -> GraphState:
    total_ok = state.get("total_ok")
    ticket_exists = state.get("ticket_exists")
    allowance_calc = state.get("allowance_calculation")
    date_comparsion = state.get("date_comparsion")

    # Guard: nur ausführen, wenn alles da ist
    if total_ok is None or ticket_exists is None or allowance_calc is None or date_comparsion is None:
        return state
    
    dates_ok = date_comparsion.periods_match

    decision = build_approval_decision_with_llm(
        total_ok=total_ok,
        ticket_exists=ticket_exists,
        allowance_calc=allowance_calc,
        dates_ok=dates_ok,
    )

    return {
        "approval_decision": decision,
    }

def update_ticket_status_node(state: GraphState) -> GraphState:
    print("\n🔥 [update_ticket_status_node] Aufgerufen!")
    print("State Keys:", list(state.keys()))

    ticket_data = state.get("ticket_data")
    decision = state.get("approval_decision")
    header = state.get("header_extraction")

    # ticket_id erst NACHDEM wir header geladen haben holen
    ticket_id = header.ticket_id if header else None

    print("  ➜ ticket_id:", ticket_id)
    print("  ➜ decision:", decision)
    print("  ➜ ticket_data:", ticket_data)

    # Guard prüfen
    missing = []
    if ticket_id is None:
        missing.append("ticket_id")
    if decision is None:
        missing.append("approval_decision")
    if ticket_data is None:
        missing.append("ticket_data")

    if missing:
        print(f"⛔ Guard aktiv – folgende Werte fehlen noch: {missing}")
        print("⛔ update_ticket_status_node beendet – State unverändert.")
        return state

    print("✅ Alle Werte vorhanden – führe Backend-Update aus...")

    update_ticket_status(
        ticket_id=ticket_id,
        decision=decision,
        ticket_data=ticket_data,
    )

    print(f"🎉 Ticket {ticket_id} erfolgreich im Backend aktualisiert.")
    return state





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
    graph.add_node("select_daily_rate", select_daily_rate_node)
    graph.add_node("compare_dates", compare_dates_node)
    graph.add_node("allowance_check", allowance_check_node)
    graph.add_node("approval_decision", approval_decision_node)
    graph.add_node("update_ticket_status", update_ticket_status_node)

    # Start: zwei Äste parallel
    graph.add_edge(START, "extract_pdf")
    graph.add_edge(START, "get_allowances")

    # PDF-Flow
    graph.add_edge("extract_pdf", "extract_data")

    # Checks, die nur die Extraktion brauchen
    graph.add_edge("extract_data", "check_ticket_exists")
    graph.add_edge("extract_data", "check_total")

    # select_daily_rate braucht BEIDES:
    # - destination aus header_extraction (kommt aus extract_data)
    # - allowances aus get_allowances
    graph.add_edge("extract_data", "select_daily_rate")
    graph.add_edge("extract_data", "compare_dates")
    graph.add_edge("get_allowances", "select_daily_rate")

    graph.add_edge("compare_dates", "allowance_check")
    graph.add_edge("select_daily_rate", "allowance_check")
    # vorläufige Enden
    graph.add_edge("check_ticket_exists", "approval_decision")
    graph.add_edge("check_total", "approval_decision")
    graph.add_edge("allowance_check", "approval_decision")
    graph.add_edge("approval_decision", "update_ticket_status")
    graph.add_edge("update_ticket_status", END)

    return graph.compile()



def run_workflow(pdf_path: Path) -> None:
    app = build_app()

    initial_state: GraphState = {
        "pdf_path": pdf_path,
    }

    final_state: GraphState = app.invoke(initial_state)

    print("=== Finaler GraphState ===")
    print("pdf_path:", final_state)
