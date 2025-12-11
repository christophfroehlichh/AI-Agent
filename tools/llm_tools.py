import time
import json
from typing import Dict, Optional

from langchain_ollama import ChatOllama
from config.settings import OLLAMA_MODEL
from models.expense import (
    AllowanceCalculation,
    ApprovalDecision,
    RateSelection,
    HeaderExtraction,
    InvoicesExtraction,
    SummaryExtraction,
    DateComparsion
)

# LLM Instanz zum weiterverwenden in diesem Modul
_LLM: ChatOllama | None = None

def get_llm() -> ChatOllama:
    """
    Erzeugt eine ChatOllama-Instanz, die für JSON/structured output taugt.
    Das `format="json"` hilft, dass das Modell gültiges JSON ausspuckt.
    """
    global _LLM
    if _LLM is None:
        _LLM = ChatOllama(
            model=OLLAMA_MODEL,
            temperature=0.0,
        )
    return _LLM

# -------------------------------------------
# 1) HEADER mit LLM extrahieren
# -------------------------------------------

def extract_header_with_llm(header_text: str) -> HeaderExtraction:
    llm = get_llm()
    print("\n-------headertext-------")
    print(header_text)
    prompt = f"""
    Lies den folgenden HEADER-Text und extrahiere exakt diese Felder:

    - destination: Reiseziel / Adresse / Firma.
    - ticket_id: TicketID.
    - time_period_header: Das Datum oder die Zeitspanne.

    Gib AUSSCHLIESSLICH ein JSON im folgenden Format zurück:

    {{
    "destination": "string oder null",
    "time_period_header": "string oder null",
    "ticket_id": "string oder null"
    }}

    Beispiel zur Orientierung:

    HEADER_TEXT_BEISPIEL:
    "2024-03-12   Maria Henderson (Employee 7721) Department: 445200 Destination: Microsoft HQ, One Microsoft Way, Redmond, WA Time Period: 2024-03-01 – 2024-03-05 Ticket ID: 992211"

    Beispiel-Antwort:
    {{
    "destination": "Microsoft HQ, One Microsoft Way, Redmond, WA",
    "time_period_header": "2024-03-01 – 2024-03-05",
    "ticket_id": "992211"
    }}

    Jetzt verarbeite diesen HEADER:

    HEADER_TEXT:
    {header_text}
    """


    structured_llm = llm.with_structured_output(HeaderExtraction)

    start = time.time()
    result: HeaderExtraction = structured_llm.invoke(prompt)
    end = time.time()
    print(f"⏱ LLM-Antwortzeit (HEADER): {end - start:.2f} Sekunden")

    return result


# -------------------------------------------
# 2) INVOICES mit LLM extrahieren
# -------------------------------------------

def extract_invoices_with_llm(invoices_text: str) -> InvoicesExtraction:
    llm = get_llm()
    print("\n-------invoicestext-------")
    print(invoices_text)
    prompt = f"""
    Lies den folgenden INVOICES-Text und extrahiere eine Liste von Einträgen.

    Extraktion:
    - date = das erste erkannte Datum im Format YYYY-MM-DD vor dem Betrag.
    - amount = der Betrag (000.00), der zu diesem Eintrag gehört.
    - Jeder Betrag erzeugt genau ein Objekt im Array.

    Gib ausschließlich folgendes JSON zurück:

    {{
    "invoices": [
        {{
        "date": "string oder null",
        "amount": 0.00
        }}
    ]
    }}

    Beispiel Input:
    "Invoices Date Type Details Amount (USD) 2024-05-02 Transport Taxi 42.50 2024-05-03 Other Lunch 18.20 2024-05-05 – 2024-05-07 Accommodation Hotel 420.00 2024-05-08 Transport Train 67.00"

    Beispiel-Antwort:
    {{
    "invoices": [
        {{ "date": "2024-05-02", "amount": 42.50 }},
        {{ "date": "2024-05-03", "amount": 18.20 }},
        {{ "date": "2024-05-05 – 2024-05-07", "amount": 420.00 }},
        {{ "date": "2024-05-08", "amount": 67.00 }}
    ]
    }}

    Hier der echte Text:

    INVOICES_TEXT:
    {invoices_text}
    """


    structured_llm = llm.with_structured_output(InvoicesExtraction)

    start = time.time()
    result: InvoicesExtraction = structured_llm.invoke(prompt)
    end = time.time()
    print(f"⏱ LLM-Antwortzeit (INVOICES): {end - start:.2f} Sekunden")

    return result


# -------------------------------------------
# 3) SUMMARY mit LLM extrahieren
# -------------------------------------------

def extract_summary_with_llm(summary_text: str) -> SummaryExtraction:
    llm = get_llm()
    print("\n-------summarytext-------")
    print(summary_text)
    prompt = f"""
    Lies den folgenden Text und extrahiere:

    - allowance Ist der Wert nachdem Wort allowance
    - transportation_total Ist der Wert nachdem Wort transportation Total
    - accommodation_total Ist der Wert nachdem Wort accommodation Total
    - time_period_summary: der Textteil mit "Time Period" und der Datums-Range.
    - total Ist der Wert nachdem Wort Total

    Regeln:
    - Beträge wie "1,121.00 USD" → 1121.00
    - Wenn ein Wert fehlt: Zahlen = 0.0, Strings = null

    Gib ausschließlich dieses JSON zurück:

    {{
        "allowance": 0.00,
        "transportation_total": 0.00,
        "accommodation_total": 0.00,
        "time_period_summary": "string oder null",
        "total": 0.00,
    }}

    Beispiel SUMMARY_TEXT:
    "Summary Time Period 2024-04-01 – 2024-04-03 Allowances 15.00 USD Transportation Details 300.00 USD Accommodation 450.00 USD TOTAL 765.00 USD"

    Beispiel-Antwort:
    {{
        "allowance": 15.00,
        "transportation_total": 300.00,
        "accommodation_total": 450.00,
        "time_period_summary": "2024-04-01 – 2024-04-03"
        "total": 765.00,
    }}

    Hier der echte Text:

    SUMMARY_TEXT:
    {summary_text}
    """


    structured_llm = llm.with_structured_output(SummaryExtraction)

    start = time.time()
    result: SummaryExtraction = structured_llm.invoke(prompt)
    end = time.time()
    print(f"⏱ LLM-Antwortzeit (SUMMARY): {end - start:.2f} Sekunden")

    return result


def select_daily_rate_with_llm(
    destination: Optional[str],
    allowances: Dict[str, float],
) -> RateSelection:
    """
    Wählt anhand der Destination eine passende Stadt und Rate aus dem allowances-Mapping.
    Gibt ein RateSelection-Objekt zurück.
    """
    llm = get_llm()
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
    total_ok: bool,
    ticket_exists: bool,
    allowance_calc: AllowanceCalculation,
    dates_ok: bool,   
) -> ApprovalDecision:
    """
    Lässt das LLM anhand vierer bools entscheiden, ob ein Report approved oder rejected wird.
    """
    llm = get_llm()

    prompt = f"""
    Du bist ein Sachbearbeiter für Reisekostenabrechnungen und musst anhand der folgenden
    Werte entscheiden, ob die Reisekostenabrechnung approved oder rejected wird.
    Anschließend schreibst du einen kurzen Kommentar, warum so entschieden wurde.

    Werte:
    - Gesamtkosten korrekt berechnet: {total_ok}
    - Ticket existiert im System: {ticket_exists}
    - Allowance korrekt berechnet: {allowance_calc.matches_summary}
    - Datumsabgleich korrekt (Header vs Summary): {dates_ok}

    Regeln:
    - Wenn EIN Wert False ist -> approve = false
    - Nur wenn ALLE Werte True sind -> approve = true
    - Kommentar: maximal 2 kurze Sätze, klare Begründung.

    Antworte NUR mit JSON: {{"approve": true/false, "comment": "string"}}
    """

    structured_llm = llm.with_structured_output(ApprovalDecision)

    start = time.time()
    decision: ApprovalDecision = structured_llm.invoke(prompt)
    end = time.time()
    print(f"⏱ LLM-Antwortzeit (ApprovalDecision): {end - start:.2f} Sekunden")

    return decision


