from src.agent.store import init_db, save_event, get_recent_events, check_conflict
from src.agent.parser import ParsedEvent

def test_store():
    init_db()

    event = ParsedEvent(
        event_type="shutdown",
        start_date="2025-03-10",
        end_date="2025-03-12",
        confidence=0.98,
        special_event=True,
        notes="Planned maintenance shutdown"
    )

    save_event("C-1042", event)
    recent = get_recent_events("C-1042")
    assert len(recent) >= 1
    assert recent[0]["event_type"] == "shutdown"
    print("Saved and retrieved:", recent[0])

    # test conflict detection
    overlap = ParsedEvent(
        event_type="production_increase",
        start_date="2025-03-11",
        end_date="2025-03-13",
        confidence=0.9,
        special_event=True,
        notes="Overlapping event"
    )
    conflict = check_conflict("C-1042", overlap)
    print("Conflict detected:", conflict)
    assert conflict is not None

if __name__ == "__main__":
    test_store()
    print("Step 2 complete.")
