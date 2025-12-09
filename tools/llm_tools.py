import time
import json
from typing import Dict, Optional

from langchain_ollama import ChatOllama
from config.settings import OLLAMA_MODEL
from models.expense import (
    AllowanceCalculation,
    ApprovalDecision,
    ExpenseReport,
    RateSelection,
)


def get_llm() -> ChatOllama:
    """
    Erzeugt eine ChatOllama-Instanz, die für JSON/structured output taugt.
    Das `format="json"` hilft, dass das Modell gültiges JSON ausspuckt.
    """
    return ChatOllama(
        model=OLLAMA_MODEL,
        temperature=0.0,
        format="json",
    )


def build_prompt(header_text: str, invoices_text: str, summary_text: str) -> str:
    return f"""
Lies die folgenden drei Textbereiche (HEADER, INVOICES, SUMMARY)
und extrahiere daraus ein JSON-Objekt mit GENAU der folgenden Struktur:

{{
  "destination": "string oder null",
  "time_period_header": "string oder null",
  "ticket_id": "string oder null",
  "invoices": [
    {{
      "amount": 0.00
    }}
  ],
  "summary": {{
    "total": 0.00,
    "allowances": 0.00,
    "transportation_total": 0.00,
    "accommodation_total": 0.00,
    "time_period_summary": "string oder null"
  }}
}}

Wichtige Regeln:
- Gib AUSSCHLIESSLICH ein JSON in genau dieser Struktur zurück, keine Erklärungen.
- Wenn eine Information nicht gefunden wird → null (Strings) oder 0.0 (Zahlen).
- Zahlen ohne Währungssymbole oder Kommas, z.B. "1,121.00 USD" → 1121.00.

---------------- HEADER ----------------
Extrahiere NUR aus dem HEADER:
- destination
- ticket_id
- time_period_header → die komplette Zeile, in der die Time Period im Header steht.

HEADER_TEXT:
{header_text}
Ende HEADER_Text

---------------- INVOICES ----------------
Extrahiere NUR aus dem INVOICES-Text:

- ALLE Beträge der Form 000.00 
- JEDER gefundene Betrag erzeugt genau ein Objekt im Array "invoices".
- WICHTIG: Verwende AUSSCHLIESSLICH Beträge aus diesem Abschnitt.
  NIEMALS Werte aus der SUMMARY oder anderen Bereichen.
- Wenn keine Beträge im INVOICES-Text enthalten sind, gib eine leere Liste zurück.

INVOICES_TEXT:
{invoices_text}
Ende INVOICES_TEXT

---------------- SUMMARY ----------------
Extrahiere NUR aus dem SUMMARY-Text:
- summary.total
- summary.allowances
- summary.transportation_total
- summary.accommodation_total
- time_period_summary → die komplette Zeile, die die Summary-Zeitspanne enthält.

SUMMARY_TEXT:
{summary_text}
Ende SUMMARY_TEXT
"""


def analyze_expenses_with_llm(
    header_text: str,
    invoices_text: str,
    summary_text: str,
    llm: Optional[ChatOllama] = None,
) -> ExpenseReport:
    if llm is None:
        llm = get_llm()

    prompt = build_prompt(header_text, invoices_text, summary_text)
    structured_llm = llm.with_structured_output(ExpenseReport)

    start = time.time()
    result: ExpenseReport = structured_llm.invoke(prompt)
    end = time.time()
    print(f"⏱ LLM-Antwortzeit (Extraktion): {end - start:.2f} Sekunden")

    return result


def select_daily_rate_with_llm(
    llm: ChatOllama,
    destination: Optional[str],
    allowances: Dict[str, float],
) -> RateSelection:
    """
    Wählt anhand der Destination eine passende Stadt und Rate aus dem allowances-Mapping.
    Gibt ein RateSelection-Objekt zurück.
    """
    prompt = f"""
    Du bekommst eine Destination als String und ein Mapping von Städten zu Tagesätzen (Allowances).

    Deine Aufgabe:
    - Finde die am besten passende Stadt im Allowances-Mapping zur Destination.
    - Gib ausschließlich dieses JSON zurück:
    {{"matched_city": "string oder null", "daily_rate": 0.0}}

    Destination:
    {destination}

    Allowances (JSON):
    {json.dumps(allowances, ensure_ascii=False)}
    """

    structured_llm = llm.with_structured_output(RateSelection)
    start = time.time()
    result: RateSelection = structured_llm.invoke(prompt)
    end = time.time()
    print(f"⏱ LLM-Antwortzeit (RateSelection): {end - start:.2f} Sekunden")

    return result


def calculate_allowance_with_llm(
    llm: ChatOllama,
    time_period_summary: Optional[str],
    daily_rate: Optional[float],
    extracted_allowance: Optional[float],
) -> AllowanceCalculation:
    """
    Lässt das LLM die Dauer der Reise berechnen und anschließend wird geprüft ob die Allowance korrekt berechnet wurde.
    """
    prompt = f"""
    Du bist ein Sachbearbeiter für Reisekostenabrechnungen und du musst berechnen wie lange die Reise ging und
    ob die Allowance korrekt berechnet wurde.

    Du bekommst:
    - Eine Zeitspanne (time_period_summary) im Format "YYYY-MM-DD – YYYY-MM-DD"
    - Einen Tagessatz (daily_rate)
    - Eine Allowance (extracted_allowance)

    Aufgaben:
    1. Berechne die Anzahl der Tage INKLUSIVE Start- und Enddatum.
    2. Berechne expected_allowance = daily_rate * days.
    3. Prüfe, ob expected_allowance und extracted_allowance übereinstimmen (Toleranz 0.01).

    Gib ausschließlich folgendes JSON zurück:
    {{"days": 0, "expected_allowance": 0.0, "matches_summary": true/false}}

    time_period_summary: {time_period_summary}
    daily_rate: {daily_rate}
    extracted_allowance: {extracted_allowance}
    """

    structured_llm = llm.with_structured_output(AllowanceCalculation)
    start = time.time()
    result: AllowanceCalculation = structured_llm.invoke(prompt)
    end = time.time()
    print(f"⏱ LLM-Antwortzeit (AllowanceCalculation): {end - start:.2f} Sekunden")

    return result


def build_approval_decision_with_llm(
    llm: ChatOllama,
    expense: ExpenseReport,
    total_ok: bool,
    ticket_exists: bool,
    allowance_calc: AllowanceCalculation,
) -> ApprovalDecision:
    """
    Kurzer Prompt: nur Booleans und Ergebnisentscheidung.
    """
    prompt = f"""
    Du bist ein Sachbearbeiter für Reisekostenabrechnungen und du musst anhand der folgenden
    Werte entscheiden ob die Reisenkostenabrechnung approved oeder rejected werden soll.
    Anschließend schreibst du einen kurzen Kommentar warum so entschieden wurde.

    Gesamtkosten wurden richtig berechnet: {total_ok}
    Das Ticket existiert im System: {ticket_exists}
    Die Allowance wurde korrekt berechnet: {allowance_calc.matches_summary}

    Regeln:
    - Wenn ein Wert False -> approve = false
    - Wenn alle Werte True -> approve = true
    - Kommentar: max. 2 kurze Sätze, kurz begründen.

    Antworte NUR mit JSON: {{"approve": true/false, "comment": "string"}}
    """

    structured_llm = llm.with_structured_output(ApprovalDecision)
    start = time.time()
    decision: ApprovalDecision = structured_llm.invoke(prompt)
    end = time.time()
    print(f"⏱ LLM-Antwortzeit (ApprovalDecision): {end - start:.2f} Sekunden")

    return decision

