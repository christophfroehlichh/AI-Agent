from typing import Any, Dict, Optional, Tuple

import requests
from config.settings import BASE_URL, USERNAME, PASSWORD
from models.expense import ApprovalDecision

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

