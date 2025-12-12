"""
Backend integration utilities.
Handles communication with the backend service.
"""

import logging
from typing import Any, Dict, Optional, Tuple

import requests

from config.settings import BASE_URL, PASSWORD, USERNAME
from models.expense import ApprovalDecision

logger = logging.getLogger(__name__)

AUTH = (USERNAME, PASSWORD)


def _backend_request(
    method: str,
    path: str,
    params: Optional[Dict[str, Any]] = None,
    payload: Optional[Dict[str, Any]] = None,
    timeout: float = 10.0,
) -> Tuple[Optional[requests.Response], Optional[Exception]]:
    """Executes a backend HTTP request and returns (response, error)."""
    url = f"{BASE_URL}{path}"
    try:
        resp = requests.request(
            method.upper(),
            url,
            auth=AUTH,
            params=params,
            json=payload,
            timeout=timeout,
        )
        return resp, None
    except Exception as e:
        return None, e


def get_allowances() -> Dict[str, float]:
    """Fetches allowance mapping from backend (GET /allowances)."""
    resp, error = _backend_request("GET", "/allowances")
    if error is not None:
        logger.warning("Failed to fetch allowances: %s", error)
        return {}
    if resp is None:
        logger.warning("No response when fetching allowances.")
        return {}

    if resp.status_code != 200:
        logger.warning("Unexpected HTTP %s while fetching allowances.", resp.status_code)
        return {}

    try:
        data = resp.json()
        logger.info("Loaded %d allowance entries.", len(data))
        return {k: float(v) for k, v in data.items()}

    except Exception as e:
        logger.warning("Failed to parse allowances JSON: %s", e)
        return {}


def check_ticket_exists(ticket_id: Optional[str]) -> Tuple[bool, Optional[Dict[str, Any]]]:
    """Checks if a travel ticket exists (GET /travelTicket) and returns (exists, ticket_data)."""
    if not ticket_id:
        return False, None

    resp, error = _backend_request("GET", "/travelTicket", {"ticketID": ticket_id})
    if error is not None:
        logger.warning("Failed to check ticket %s: %s", ticket_id, error)
        return False, None
    if resp is None:
        logger.warning("No response while checking ticket %s.", ticket_id)
        return False, None

    if resp.status_code == 200:
        try:
            data = resp.json()
            logger.info("Ticket %s found in backend.", ticket_id)
            return True, data
        except Exception as e:
            logger.warning("Failed to parse ticket JSON for %s: %s", ticket_id, e)
            return False, None

    if resp.status_code == 404:
        return False, None

    logger.warning("Unexpected HTTP %s while checking ticket %s.", resp.status_code, ticket_id)
    return False, None


def update_ticket_status(
    ticket_id: Optional[str],
    decision: ApprovalDecision,
    ticket_data: Optional[Dict[str, Any]],
) -> None:
    """Updates ticketStatus and comment and writes back the ticket (PUT /travelTicket)."""
    if not ticket_id:
        logger.warning("Cannot update ticket: missing ticket_id.")
        return
    if ticket_data is None:
        logger.warning("Cannot update ticket %s: missing ticket_data.", ticket_id)
        return

    ticket_data["ticketStatus"] = "APPROVED" if decision.approve else "REJECTED"
    ticket_data["comment"] = decision.comment

    resp, error = _backend_request("PUT", "/travelTicket", None, ticket_data)
    if error is not None:
        logger.warning("Failed to update ticket %s: %s", ticket_id, error)
        return
    if resp is None:
        logger.warning("No response while updating ticket %s.", ticket_id)
        return
    if resp.status_code not in (200, 204):
        logger.warning(
            "Unexpected HTTP %s while updating ticket %s: %s",
            resp.status_code,
            ticket_id,
            resp.text,
        )
        return

    logger.info("Ticket %s updated (status=%s).", ticket_id, ticket_data["ticketStatus"])
