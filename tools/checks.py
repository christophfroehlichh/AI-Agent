from typing import Any, Dict
from models.expense import InvoicesExtraction, SummaryExtraction


def check_total(
    invoices_extraction: InvoicesExtraction,
    summary_extraction: SummaryExtraction,
) -> bool:
    """
    Check: Summe aller invoice.amounts soll summary.total entsprechen.
    Kleine Rundungsfehler werden toleriert.
    """
    invoice_sum = sum(inv.amount for inv in invoices_extraction.invoices)
    total = summary_extraction.total

    print(f"invoice_sum={invoice_sum}, summary_total={total}")

    toleranz = 0.01  
    return abs(invoice_sum - total) <= toleranz

