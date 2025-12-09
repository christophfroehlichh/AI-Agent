from typing import List, Optional

from pydantic import BaseModel, Field


class Invoice(BaseModel):
    amount: float = Field(..., description="Betrag einer Invoice")


class Summary(BaseModel):
    total: float = Field(..., description="Gesamtbetrag Summary")
    allowances: float = Field(..., description="Allowances Summary")
    transportation_total: float = Field(..., description="Transportkosten Summary")
    accommodation_total: float = Field(
        ...,
        description="Unterkunftskosten Summary",
    )
    time_period_summary: Optional[str] = Field(
        None, description="Zeile aus der Summary, die die Zeitspanne enthält"
    )


class ExpenseReport(BaseModel):
    destination: Optional[str] = Field(
        None, description="Destination aus dem Header"
    )
    time_period_header: Optional[str] = Field(
        None, description=" Time-Period-Zeile aus dem Header"
    )
    ticket_id: Optional[str] = Field(
        None, description="Ticket-ID aus dem Header"
    )
    invoices: List[Invoice] = Field(
        default_factory=list, description="Liste der einzelnen Beträge"
    )
    summary: Summary


class RateSelection(BaseModel):
    matched_city: Optional[str] = Field(
        None, description="Name der gematchten Stadt"
    )
    daily_rate: Optional[float] = Field(
        None, description="Tagesatz für die Stadt"
    )


class AllowanceCalculation(BaseModel):
    days: int
    expected_allowance: float
    matches_summary: bool


class ApprovalDecision(BaseModel):
    approve: bool = Field(..., description="Ob der Report freigegeben werden soll")
    comment: str = Field(..., description="Kommentar zur Entscheidung")


__all__ = [
    "Invoice",
    "Summary",
    "ExpenseReport",
    "RateSelection",
    "AllowanceCalculation",
    "ApprovalDecision",
]
