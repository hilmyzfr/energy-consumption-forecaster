import json
import os
import re
from datetime import date
from anthropic import Anthropic
from pydantic import BaseModel
from dotenv import load_dotenv
from src.agent.prompts import PARSER_SYSTEM_PROMPT

load_dotenv()
client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

class ParsedEvent(BaseModel):
    event_type: str
    start_date: str | None
    end_date: str | None
    confidence: float
    special_event: bool
    notes: str

def parse_email(raw_email: str) -> ParsedEvent:
    system = PARSER_SYSTEM_PROMPT.format(today=date.today().isoformat())
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=512,
        system=system,
        messages=[{"role": "user", "content": raw_email}]
    )
    raw = response.content[0].text
    clean = re.sub(r"```json|```", "", raw).strip()
    data = json.loads(clean)
    return ParsedEvent(**data)
