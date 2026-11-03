"""Match actual client tool results to their requested calls for acceptance checks."""


def completed_tools(events):
    calls = {}
    completed = []
    for event in events:
        message = event.get("message")
        if not isinstance(message, dict):
            continue
        content = message.get("content", [])
        if not isinstance(content, list):
            continue
        for block in content:
            if event.get("type") == "assistant" and block.get("type") == "tool_use":
                calls[block["id"]] = block
            elif event.get("type") == "user" and block.get("type") == "tool_result":
                call = calls.get(block.get("tool_use_id"))
                if call is None or block.get("is_error", False):
                    continue
                text = block.get("content", "")
                if isinstance(text, list):
                    text = "\n".join(b.get("text", "") for b in text if b.get("type") == "text")
                completed.append({**call, "result_text": text})
    return completed
