import sys
import time
import json
from pathlib import Path
from typing import Any, Dict, Optional, List, Tuple

import requests
from pypdf import PdfReader
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field


# ---------- Pydantic-Modelle für strukturierten Output ----------

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


# ---------- STEP 1: PDF → Header / Invoices / Summary ----------

def extract_sections_from_pdf(pdf_path: Path) -> tuple[str, str, str]:
    """Liest das PDF ein und splittet den Text in Header, Invoices und Summary."""
    reader = PdfReader(str(pdf_path))
    texts: List[str] = []
    for page in reader.pages:
        texts.append(page.extract_text() or "")
    full_text = "\n".join(texts)

    lower = full_text.lower()
    invoices_idx = lower.find("invoices")
    summary_idx = lower.find("summary")

    if invoices_idx == -1 or summary_idx == -1:
        header_text = full_text.strip()
        invoices_text = ""
        summary_text = ""
    else:
        header_text = full_text[:invoices_idx].strip()
        invoices_text = full_text[invoices_idx:summary_idx].strip()
        summary_text = full_text[summary_idx:].strip()

    return header_text, invoices_text, summary_text


# ---------- STEP 2: Prompt für Extraktion ----------

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


# ---------- STEP 3: LLM für Extraktion ----------

def get_llm() -> ChatOllama:
    """
    Erzeugt eine ChatOllama-Instanz, die für JSON/structured output taugt.
    Das `format="json"` hilft, dass das Modell gültiges JSON ausspuckt.
    """
    return ChatOllama(
        model="llama3.2",
        temperature=0.0,
        format="json",
    )


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


# ---------- STEP 4: Backend-API Helper ----------

BASE_URL = "https://agents-workshop-backend.cfapps.eu10-004.hana.ondemand.com"
USERNAME = "agentsworkshopbackend"
PASSWORD = "bowseS-caqne6-satmus"

AUTH = (USERNAME, PASSWORD)


def _backend_request(
    method: str,
    path: str,
    *,
    params: Optional[Dict[str, Any]] = None,
    json: Optional[Dict[str, Any]] = None,
    timeout: float = 10.0,
) -> Tuple[Optional[requests.Response], Optional[Exception]]:
    """
    Zentrale Helper-Funktion für alle Backend-Calls.
    Kümmert sich um Base URL, Auth und Timeout.

    Gibt (response, error) zurück. Genau einer von beiden ist None.
    """
    url = f"{BASE_URL}{path}"
    try:
        resp = requests.request(
            method=method.upper(),
            url=url,
            params=params,
            json=json,
            auth=AUTH,
            timeout=timeout,
        )
        return resp, None
    except Exception as e:
        return None, e


def get_allowances() -> Dict[str, float]:
    """
    Holt das Allowances-Mapping vom Backend: GET /allowances
    """
    resp, error = _backend_request("GET", "/allowances")
    if error is not None:
        print(f"Error while fetching allowances: {error}")
        return {}
    if resp is None:
        print("No response when fetching allowances.")
        return {}
    if resp.status_code == 200:
        try:
            data = resp.json()
            return {k: float(v) for k, v in data.items()}
        except Exception as e:
            print(f"Error parsing allowances JSON: {e}")
            return {}
    else:
        print(f"Unexpected HTTP {resp.status_code} while fetching allowances.")
        return {}
    
    
def update_ticket_status(
    ticket_id: Optional[str],
    decision: ApprovalDecision,
    ticket_data: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Setzt ticketStatus + comment auf bereits gehohlte Ticket-Daten (falls vorhanden)
    und schreibt es dann mit PUT /travelTicket zurück.
    Gibt keine Log-Zeichenkette zurück, sondern druckt bei Fehlern/Erfolg.
    """
    if not ticket_id:
        print("Cannot update ticket status: no ticket_id available.")
        return

    # Status + Kommentar setzen
    ticket_data["ticketStatus"] = "APPROVED" if decision.approve else "REJECTED"
    ticket_data["comment"] = decision.comment

    # PUT /travelTicket
    resp, error = _backend_request(
        "PUT",
        "/travelTicket",
        json=ticket_data,
    )
    if error is not None:
        print(f"Error while updating ticket {ticket_id} in backend: {error}")
        return
    if resp is None:
        print(f"No response while updating ticket {ticket_id} in backend.")
        return
    if resp.status_code not in (200, 204):
        print(
            f"Unexpected HTTP {resp.status_code} while updating ticket {ticket_id}: "
            f"{resp.text}"
        )
        return

    print(f"Ticket {ticket_id} updated in backend (status={ticket_data['ticketStatus']}).")


# ---------- STEP 5: Checks & LLM-gestützte Tools ----------

def check_ticket_exists(
    expense_report: Dict[str, Any],
) -> Tuple[bool, Optional[Dict[str, Any]]]:
    """
    Prüft, ob das Ticket im Backend existiert und gibt (exists, ticket_data) zurück.
    Fehler werden kurz ausgegeben (print).
    """
    exists = False
    ticket_data = None
    ticket_id = expense_report.get("ticket_id")

    if not ticket_id:
        print("No ticket_id in expense report; cannot verify ticket in backend.")
        return exists, ticket_data

    resp, error = _backend_request(
        "GET",
        "/travelTicket",
        params={"ticketID": ticket_id},
    )

    if error is not None:
        print(f"Error while checking ticket {ticket_id} in backend: {error}")
        return exists, ticket_data

    if resp is None:
        print(f"No response while checking ticket {ticket_id} in backend.")
        return exists, ticket_data

    if resp.status_code == 200:
        try:
            ticket_data = resp.json()
            print(f"Ticket {ticket_id} exists in backend system.")
            exists = True
        except Exception as e:
            print(f"Error parsing ticket JSON for {ticket_id}: {e}")
    elif resp.status_code == 404:
        print(f"Ticket {ticket_id} does NOT exist in backend system (404).")
    else:
        print(
            f"Unexpected response while checking ticket {ticket_id}: HTTP {resp.status_code}."
        )

    return exists, ticket_data


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
