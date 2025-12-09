import sys
import time
from pathlib import Path
from typing import Any, Dict

from models.expense import RateSelection
from tools.pdf_tools import extract_sections_from_pdf
from tools.backend_tools import get_allowances, check_ticket_exists, update_ticket_status
from tools.checks import check_total
from tools.llm_tools import (
    get_llm,
    analyze_expenses_with_llm,
    select_daily_rate_with_llm,
    calculate_allowance_with_llm,
    build_approval_decision_with_llm,
)


# ---------- main ----------

def main(pdf_path_str: str) -> None:
    program_start = time.time()

    pdf_path = Path(pdf_path_str)
    if not pdf_path.exists():
        print(f"PDF not found: {pdf_path}")
        sys.exit(1)

    print("📄 Extracting sections from PDF...")
    header_text, invoices_text, summary_text = extract_sections_from_pdf(pdf_path)

    print("🤖 Calling LLM for extraction...")
    llm = get_llm()
    expense = analyze_expenses_with_llm(header_text, invoices_text, summary_text, llm)

    print("\n🔎 Extrahierter ExpenseReport:")
    print(expense)

    print("\n🌐 Fetching allowances mapping from backend...")
    allowances = get_allowances()
    print(f"Allowances: {allowances}")

    # --- TOOL EXECUTION ---
    # 1) Total prüfen
    total_ok = check_total(expense.model_dump())

    # 2) Ticket im Backend prüfen 
    ticket_exists, cached_ticket_data = check_ticket_exists(expense.model_dump())

    # 3) Allowance-Rate wählen
    selection: RateSelection = select_daily_rate_with_llm(
        llm, expense.destination, allowances
    )
    print("\n🔎 Auswahl durch LLM (RateSelection):")
    print(selection)

    # 4) Allowances gegen Summary prüfen
    allowance_calc = calculate_allowance_with_llm(
        llm,
        expense.summary.time_period_summary,
        selection.daily_rate,
        expense.summary.allowances,
    )
    print("\n🔎 Allowance Calculation via LLM:")
    print(allowance_calc)
    print(
        f"days={allowance_calc.days}, "
        f"expected={allowance_calc.expected_allowance}, "
        f"matches={allowance_calc.matches_summary}"
    )

    # 5) Approval-Entscheidung durch LLM
    decision = build_approval_decision_with_llm(
        llm=llm,
        expense=expense,
        total_ok=total_ok,
        ticket_exists=ticket_exists,
        allowance_calc=allowance_calc,
    )
    print("\n✅ Approval Decision from LLM:")
    print(decision)

    # 6) Ticket-Status im Backend updaten (mit gecachten Daten)
    update_ticket_status(expense.ticket_id, decision, cached_ticket_data)

    print(f"\n⏱ Gesamtzeit ab Programmstart: {time.time() - program_start:.2f} Sekunden")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python main.py <pfad_zum_pdf>")
        sys.exit(1)

    main(sys.argv[1])
