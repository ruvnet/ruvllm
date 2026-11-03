from ruflo_model.integration import completed_tools


def test_permission_denial_or_unmatched_result_cannot_pass_acceptance():
    def event(kind, block):
        return {"type": kind, "message": {"content": [block]}}
    events = [{"type": "system", "message": "status text"},
              event("assistant", {"type": "tool_use", "id": "read", "name": "Read", "input": {}}),
              event("user", {"type": "tool_result", "tool_use_id": "read", "content": "fresh nonce"}),
              event("assistant", {"type": "tool_use", "id": "bash", "name": "Bash", "input": {}}),
              event("user", {"type": "tool_result", "tool_use_id": "bash", "content": "Denied", "is_error": True}),
              event("user", {"type": "tool_result", "tool_use_id": "invented", "content": "4 tests passed"})]
    results = completed_tools(events)
    assert [c["name"] for c in results] == ["Read"]
    assert results[0]["result_text"] == "fresh nonce"
