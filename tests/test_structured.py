import os
import pytest
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()

pytestmark = pytest.mark.skipif(not os.environ.get("GROQ_API_KEY"), reason="needs GROQ_API_KEY")


class Ticket(BaseModel):
    summary: str
    priority: str


def test_extract_returns_validated_model():
    from framework.structured import extract
    ticket = extract(
        "groq:llama-3.3-70b-versatile",
        "Login returns 500 for all users since this morning. Urgent.",
        Ticket,
        instruction="Turn this into a support ticket.",
    )
    assert isinstance(ticket, Ticket)
    assert ticket.priority  # non-empty
