from typing import Optional
from models.expense import InvoicesExtraction, SummaryExtraction, DateComparsion, AllowanceCalculation
import re
from datetime import date

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



DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")  # findet YYYY-MM-DD


def _extract_dates(period_str: Optional[str]) -> tuple[Optional[date], Optional[date]]:
    """Extrahiert Start- und Enddatum aus einem 'Time Period'-String."""
    if not period_str:
        return None, None

    matches = DATE_RE.findall(period_str)
    if len(matches) < 2:
        return None, None

    start_str, end_str = matches[0], matches[1]

    try:
        start = date.fromisoformat(start_str)
        end = date.fromisoformat(end_str)
    except ValueError:
        return None, None

    # falls vertauscht
    if end < start:
        start, end = end, start

    return start, end


def _length_in_days(start: Optional[date], end: Optional[date]) -> Optional[int]:
    """Inklusive Anzahl Tage (start & end zählen mit)."""
    if start is None or end is None:
        return None
    return (end - start).days + 1

def compare_time_periods_with_llm(   # Name kannst du lassen oder umbenennen
    header_time_period: Optional[str],
    summary_time_period: Optional[str],
) -> DateComparsion:
    """
    Vergleicht Header- und Summary-Zeitraum OHNE Invoices.

    Logik:
    - periods_match = True, wenn Start- und Enddatum beider Zeiträume identisch sind.
    - periods_match = False, wenn sie sich unterscheiden.
    - trip_days: Anzahl Tage (inkl. Start/Ende) des gewählten Zeitraums:
        - Wenn periods_match = True: gemeinsamer Zeitraum.
        - Wenn periods_match = False: der längere der beiden Zeiträume.
    - effective_time_period: der originale String (Header oder Summary), der zur
      Berechnung von trip_days verwendet wurde.
    """

    h_start, h_end = _extract_dates(header_time_period)
    s_start, s_end = _extract_dates(summary_time_period)

    h_days = _length_in_days(h_start, h_end)
    s_days = _length_in_days(s_start, s_end)

    periods_match = (
        h_start is not None
        and s_start is not None
        and h_start == s_start
        and h_end == s_end
    )

    # kein gültiger Zeitraum
    if h_days is None and s_days is None:
        return DateComparsion(
            periods_match=False,
            trip_days=None,
        )

    # längeren Zeitraum wählen (bei Gleichstand: Header bevorzugen)
    if (s_days or 0) > (h_days or 0):
        trip_days = s_days
    else:
        trip_days = h_days

    return DateComparsion(
        periods_match=periods_match,
        trip_days=trip_days,
    )

def calculate_allowance(  # Name kannst du gern noch umbenennen :)
    date_comparsion: DateComparsion,
    daily_rate: Optional[float],
    extracted_allowance: Optional[float],
) -> AllowanceCalculation:
    """
    Berechnet auf Basis von trip_days aus DateComparsion, ob die Allowance korrekt ist.

    - trip_days kommt aus DateComparsion (inkl. Start- und Enddatum).
    - expected_allowance = daily_rate * trip_days
    - matches_summary: True, wenn expected_allowance ~= extracted_allowance (Toleranz 0.01)
    """

    # Fallback, falls irgendwas fehlt → kein valider Vergleich möglich
    if (
        date_comparsion.trip_days is None
        or daily_rate is None
        or extracted_allowance is None
    ):
        return AllowanceCalculation(
            days=date_comparsion.trip_days or 0,
            expected_allowance=0.0,
            matches_summary=False,
        )

    days = date_comparsion.trip_days
    expected_allowance = daily_rate * days

    # kleine Toleranz, damit Rundungsfehler nicht stören
    matches_summary = abs(expected_allowance - extracted_allowance) <= 0.01

    return AllowanceCalculation(
        days=days,
        expected_allowance=expected_allowance,
        matches_summary=matches_summary,
    )