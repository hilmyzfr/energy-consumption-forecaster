from pathlib import Path
from src.agent.parser import parse_email

samples = Path("data/sample_emails")

def test_shutdown():
    email = (samples / "shutdown.txt").read_text()
    result = parse_email(email)
    assert result.special_event is True
    assert result.event_type == "shutdown"
    assert result.confidence > 0.7
    print(result.model_dump_json(indent=2))

def test_ambiguous():
    email = (samples / "ambiguous.txt").read_text()
    result = parse_email(email)
    print(result.model_dump_json(indent=2))
    # low confidence expected but not enforced — model judgment varies

if __name__ == "__main__":
    test_shutdown()
    test_ambiguous()
