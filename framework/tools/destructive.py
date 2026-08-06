"""Destructive example tools — gated behind approval via guardrails(require_approval=...)."""

import json

from langchain.tools import tool


@tool
def send_email(to: str, body: str) -> str:
    """Send an email to a recipient. DESTRUCTIVE — real-world side effect."""
    return json.dumps({"status": "sent", "to": to, "body_preview": body[:60]})


@tool
def delete_record(record_id: str) -> str:
    """Permanently delete a record by id. DESTRUCTIVE and irreversible."""
    return json.dumps({"status": "deleted", "record_id": record_id})
