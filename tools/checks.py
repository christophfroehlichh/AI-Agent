from typing import Any, Dict


def check_total(
    expense_report_dict: Dict[str, Any],
) -> bool:
    """
    Addiert alle invoice amounts und vergleicht sie mit summary.total.
    Gibt True zurück, wenn OK, sonst False. Fehlermeldungen werden per print ausgegeben.
    """
    try:
        invoices = expense_report_dict.get("invoices", [])
        summary = expense_report_dict.get("summary", {}) or {}

        invoice_sum = 0.0
        for inv in invoices:
            amount = inv.get("amount", 0.0)
            try:
                invoice_sum += float(amount)
            except (TypeError, ValueError):
                continue

        try:
            total = float(summary.get("total", 0.0))
        except (TypeError, ValueError):
            total = 0.0

        print(f"total: {total}, invoice_sum: {invoice_sum}")

        if abs(invoice_sum - total) < 0.01:
            print("Total amount is equal to the sum of items.")
            return True
        else:
            print(
                "Total amount is NOT equal to the sum of items "
                f"(sum={invoice_sum:.2f}, total={total:.2f})."
            )
            return False

    except Exception as e:
        print(f"Error while checking total: {e}")
        return False

