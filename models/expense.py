"""
Pydantic models used across the expense agent workflow.
"""

from typing import List, Optional

from pydantic import BaseModel, Field


# -------------------------
# Raw PDF Sections
# -------------------------

class PdfSections(BaseModel):
    header: str
    invoices: str
    summary: str


# -------------------------
# LLM Extraction Models
# -------------------------

class HeaderExtraction(BaseModel):
    destination: Optional[str] = Field(
        None, description="Travel destination extracted from the header"
    )
    time_period_header: Optional[str] = Field(
        None, description="Original time period line from the header"
    )
    ticket_id: Optional[str] = Field(
        None, description="Ticket or booking ID from the header"
    )


class Invoice(BaseModel):
    amount: float = Field(..., description="Invoice amount")
    date: Optional[str] = Field(None, description="Invoice date")


class InvoicesExtraction(BaseModel):
    invoices: List[Invoice] = Field(
        default_factory=list,
        description="Invoices extracted from the INVOICES section",
    )


class SummaryExtraction(BaseModel):
    total: float = Field(..., description="Total amount from summary")
    allowance: float = Field(..., description="Allowance total from summary")
    transportation_total: float = Field(..., description="Transportation costs")
    accommodation_total: float = Field(..., description="Accommodation costs")
    time_period_summary: Optional[str] = Field(
        None, description="Time period line from the summary"
    )


# -------------------------
# Business Logic Models
# -------------------------

class RateSelection(BaseModel):
    matched_city: Optional[str] = Field(
        None, description="Matched city for allowance lookup"
    )
    daily_rate: Optional[float] = Field(
        None, description="Daily allowance rate"
    )


class DateComparsion(BaseModel):  # known typo, kept for compatibility
    periods_match: bool = Field(
        ..., description="Whether header and summary periods match"
    )
    trip_days: Optional[int] = Field(
        None, description="Number of travel days (inclusive)"
    )


class AllowanceCalculation(BaseModel):
    days: int
    expected_allowance: float
    matches_summary: bool


class ApprovalDecision(BaseModel):
    approve: bool = Field(..., description="Whether the report is approved")
    comment: str = Field(..., description="Decision rationale")


__all__ = [
    "PdfSections",
    "HeaderExtraction",
    "Invoice",
    "InvoicesExtraction",
    "SummaryExtraction",
    "RateSelection",
    "DateComparsion",
    "AllowanceCalculation",
    "ApprovalDecision",
]
