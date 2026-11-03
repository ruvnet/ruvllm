import asyncio
import json
import httpx
from fastapi.testclient import TestClient

from ruflo_model.gateway import ALIAS, anthropic_stream, anthropic_to_openai, create_app, openai_stream
from ruflo_model.strict_tools import ensure_unique_parameters
import pytest

KEY = "test-only-key-with-at-least-24-characters"


class Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return list(range(sum(len(str(m)) for m in messages) // 4))


def test_tool_result_roundtrip_preserves_error():
    request = {"model": ALIAS, "messages": [
        {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "Read", "input": {"file_path": "/x"}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "is_error": True, "content": "Not found"}]}]}
    result = anthropic_to_openai(request)
    assert result["messages"][1] == {"role": "tool", "tool_call_id": "t1", "content": "Tool execution failed: Not found"}
    assert json.loads(result["messages"][0]["tool_calls"][0]["function"]["arguments"])["file_path"] == "/x"


def test_auth_context_and_model_validation():
    app = create_app("http://127.0.0.1:1", "/model", KEY, Tokenizer())
    with TestClient(app) as c:
        assert c.get("/v1/models").status_code == 401
        headers = {"x-api-key": KEY}
        assert c.get("/v1/models", headers=headers).json()["data"][0]["id"] == ALIAS
        assert c.post("/v1/messages", headers=headers, json={"model": ALIAS, "messages": [{"role": "user", "content": "x"}], "max_tokens": 9000}).status_code == 400
        assert c.post("/v1/messages", headers=headers, json={"model": "unknown", "messages": [{"role": "user", "content": "x"}]}).status_code == 404
        assert c.post("/v1/messages", headers=headers, json={"model": ALIAS, "messages": [{"role": "user", "content": [{"type": "image"}]}]}).status_code == 400


def test_stream_reassembles_split_tool_arguments():
    async def chunks():
        for delta, finish in [({"tool_calls": [{"index": 0, "id": "call_1", "function": {"name": "Read", "arguments": '{"file_'}}]}, None),
                              ({"tool_calls": [{"index": 0, "function": {"arguments": 'path":"/a"}'}}]}, None), ({}, "tool_calls")]:
            yield "data: " + json.dumps({"choices": [{"delta": delta, "finish_reason": finish}]})
        yield "data: " + json.dumps({"choices": [], "usage": {"completion_tokens": 17}})
    tools = [{"type": "function", "function": {"name": "Read", "parameters": {"type": "object", "properties": {"file_path": {"type": "string"}}, "required": ["file_path"]}}}]
    async def collect():
        return [s async for s in anthropic_stream(chunks(), tools, 23)]
    output = "".join(asyncio.run(collect()))
    assert '"stop_reason": "tool_use"' in output
    assert '"output_tokens": 17' in output
    assert output.count("event: content_block_start") == 1
    assert output.endswith('event: message_stop\ndata: {"type": "message_stop"}\n\n')


def test_actual_proxy_contract_with_mock_backend():
    def backend(request):
        body = json.loads(request.content)
        assert body["model"] == "/model"
        assert body["chat_template_kwargs"] == {"enable_thinking": False}
        return httpx.Response(200, json={"choices": [{"message": {"role": "assistant", "content": "4"}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 12, "completion_tokens": 1}})
    app = create_app("http://127.0.0.1:1", "/model", KEY, Tokenizer(), transport=httpx.MockTransport(backend))
    with TestClient(app) as c:
        response = c.post("/v1/messages", headers={"x-api-key": KEY}, json={"model": ALIAS, "messages": [{"role": "user", "content": "2+2"}]})
        assert response.status_code == 200
        assert response.json()["content"] == [{"type": "text", "text": "4"}]
        assert response.json()["usage"] == {"input_tokens": 12, "output_tokens": 1}


def test_ambiguous_or_incomplete_tools_are_not_forwarded():
    with pytest.raises(ValueError, match="Duplicate"):
        ensure_unique_parameters('<parameter=path>/a</parameter><parameter=path>/b</parameter>')
    async def chunks():
        yield 'data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}'
    async def collect():
        return [s async for s in openai_stream(chunks(), [])]
    with pytest.raises(ValueError, match="without a valid call"):
        asyncio.run(collect())
