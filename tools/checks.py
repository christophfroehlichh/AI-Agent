"""
Deterministic validation and calculation logic.
Contains non-LLM checks used within the workflow.
"""

import re
from datetime import date
from typing import Optional

from models.expense import (
    AllowanceCalculation,
    DateComparsion,
    InvoicesExtraction,
    SummaryExtraction,
)

DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")
MONEY_TOLERANCE = 0.01


def check_total(invoices: InvoicesExtraction, summary: SummaryExtraction) -> bool:
    """Checks whether invoice sum matches the summary total (tolerance applied)."""
    invoice_sum = sum(inv.amount for inv in invoices.invoices)
    return abs(invoice_sum - summary.total) <= MONEY_TOLERANCE


def _extract_dates(period_str: Optional[str]) -> tuple[Optional[date], Optional[date]]:
    """Extracts (start, end) dates from a time period string (YYYY-MM-DD ... YYYY-MM-DD)."""
    if not period_str:
        return None, None

    matches = DATE_RE.findall(period_str)
    if len(matches) < 2:
        return None, None

    try:
        start = date.fromisoformat(matches[0])
        end = date.fromisoformat(matches[1])
    except ValueError:
        return None, None

    if end < start:
        start, end = end, start

    return start, end


def _length_in_days(start: Optional[date], end: Optional[date]) -> Optional[int]:
    """Inclusive day count for a start/end date range."""
    if start is None or end is None:
        return None
    return (end - start).days + 1


def compare_time_periods(
    header_time_period: Optional[str],
    summary_time_period: Optional[str],
) -> DateComparsion:
    """Compares header vs. summary periods and returns whether they match and the effective trip length."""
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

    if h_days is None and s_days is None:
        return DateComparsion(periods_match=False, trip_days=None)

    trip_days = s_days if (s_days or 0) > (h_days or 0) else h_days
    return DateComparsion(periods_match=periods_match, trip_days=trip_days)


def calculate_allowance(
    date_cmp: DateComparsion,
    daily_rate: Optional[float],
    extracted_allowance: Optional[float],
) -> AllowanceCalculation:
    """Computes expected allowance and checks it against the extracted summary allowance."""
    if date_cmp.trip_days is None or daily_rate is None or extracted_allowance is None:
        return AllowanceCalculation(days=date_cmp.trip_days or 0, expected_allowance=0.0, matches_summary=False)

    expected = daily_rate * date_cmp.trip_days
    matches = abs(expected - extracted_allowance) <= MONEY_TOLERANCE

    return AllowanceCalculation(days=date_cmp.trip_days, expected_allowance=expected, matches_summary=matches)
