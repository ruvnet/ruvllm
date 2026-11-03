"""Loopback gateway for a single pinned MLX model.

OpenAI Chat is the router contract. Anthropic Messages is provided for isolated
Claude Code compatibility checks. No tools execute in this service.
"""
import asyncio
import copy
import json
import secrets
import time
from urllib.parse import urlparse
import uuid

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
import httpx
import jsonschema

ALIAS = "ruflo-code-4b"
MAX_BODY = 4 * 1024 * 1024


def text_content(content):
    if isinstance(content, str):
        return content
    if content is None:
        return ""
    if not isinstance(content, list) or any(not isinstance(b, dict) or b.get("type") != "text" for b in content):
        raise HTTPException(400, "Only text content is supported by this model profile")
    return "\n".join(b.get("text", "") for b in content)


def anthropic_to_openai(body):
    messages = []
    if body.get("system"):
        messages.append({"role": "system", "content": text_content(body["system"])})
    for message in body.get("messages", []):
        role, content = message.get("role"), message.get("content")
        if role not in ("user", "assistant"):
            raise HTTPException(400, "Invalid message role")
        if isinstance(content, str):
            messages.append({"role": role, "content": content})
            continue
        if not isinstance(content, list):
            raise HTTPException(400, "Invalid message content")
        texts, calls, results = [], [], []
        for block in content:
            kind = block.get("type")
            if kind == "text":
                texts.append(block.get("text", ""))
            elif kind == "tool_use" and role == "assistant":
                calls.append({"id": block["id"], "type": "function", "function": {
                    "name": block["name"], "arguments": json.dumps(block["input"])}})
            elif kind == "tool_result" and role == "user":
                result = text_content(block.get("content", ""))
                if block.get("is_error"):
                    result = "Tool execution failed: " + result
                results.append({"role": "tool", "tool_call_id": block["tool_use_id"], "content": result})
            else:
                raise HTTPException(400, f"Unsupported content block: {kind}")
        messages.extend(results)
        if texts or calls:
            item = {"role": role, "content": "\n".join(texts)}
            if calls:
                item["tool_calls"] = calls
            messages.append(item)
    tools = [{"type": "function", "function": {"name": t["name"], "description": t.get("description", ""),
             "parameters": t["input_schema"]}} for t in body.get("tools", [])]
    request = {"messages": messages, "tools": tools, "model": body.get("model", ALIAS),
               "max_tokens": body.get("max_tokens", 1024), "stream": body.get("stream", False),
               "temperature": body.get("temperature", 0)}
    if "stop_sequences" in body:
        request["stop"] = body["stop_sequences"]
    if body.get("thinking", {}).get("type") not in (None, "disabled"):
        raise HTTPException(400, "This profile uses enable_thinking=false; route reasoning requests to another model")
    choice = body.get("tool_choice", {}).get("type", "auto")
    request["tool_choice"] = {"auto": "auto", "any": "required", "none": "none"}.get(choice)
    if choice == "tool":
        request["tool_choice"] = {"type": "function", "function": {"name": body["tool_choice"]["name"]}}
    if request["tool_choice"] is None:
        raise HTTPException(400, "Unsupported tool choice")
    return request


def validate_calls(calls, tools, tool_choice=None):
    if tool_choice == "none" and calls:
        raise ValueError("Tool use was disabled")
    if tool_choice == "required" and not calls:
        raise ValueError("A tool call was required")
    if isinstance(tool_choice, dict):
        required = tool_choice.get("function", {}).get("name")
        if not calls or any(c["function"]["name"] != required for c in calls):
            raise ValueError("Model did not use the required tool")
    schemas = {t["function"]["name"]: t["function"]["parameters"] for t in tools}
    for call in calls:
        function = call["function"]
        if function["name"] not in schemas:
            raise ValueError("Model returned an unknown tool")
        args = function["arguments"]
        args = json.loads(args) if isinstance(args, str) else args
        jsonschema.validate(args, schemas[function["name"]])
    return calls


def to_anthropic(response, tools):
    choice = response["choices"][0]
    message = choice["message"]
    content = []
    if message.get("content"):
        content.append({"type": "text", "text": message["content"]})
    for call in validate_calls(message.get("tool_calls", []), tools):
        f = call["function"]
        content.append({"type": "tool_use", "id": call["id"], "name": f["name"],
                        "input": json.loads(f["arguments"]) if isinstance(f["arguments"], str) else f["arguments"]})
    usage = response.get("usage", {})
    return {"id": "msg_" + uuid.uuid4().hex, "type": "message", "role": "assistant", "model": ALIAS,
            "content": content, "stop_reason": stop_reason(choice.get("finish_reason")), "stop_sequence": None,
            "usage": {"input_tokens": usage.get("prompt_tokens", 0), "output_tokens": usage.get("completion_tokens", 0)}}


def stop_reason(reason):
    return {"length": "max_tokens", "tool_calls": "tool_use"}.get(reason, "end_turn")


def event(kind, **data):
    return f"event: {kind}\ndata: {json.dumps({'type': kind, **data})}\n\n"


async def anthropic_stream(lines, tools, input_tokens, tool_choice=None):
    message_id = "msg_" + uuid.uuid4().hex
    yield event("message_start", message={"id": message_id, "type": "message", "role": "assistant",
        "content": [], "model": ALIAS, "stop_reason": None, "stop_sequence": None,
        "usage": {"input_tokens": input_tokens, "output_tokens": 0}})
    calls, text_started, finish, output_tokens = {}, False, None, 0
    async for line in lines:
        if not line.startswith("data: ") or line[6:] == "[DONE]":
            continue
        chunk = json.loads(line[6:])
        output_tokens = (chunk.get("usage") or {}).get("completion_tokens", output_tokens)
        for choice in chunk.get("choices", []):
            finish = choice.get("finish_reason") or finish
            delta = choice.get("delta", {})
            if delta.get("content"):
                if not text_started:
                    yield event("content_block_start", index=0, content_block={"type": "text", "text": ""})
                    text_started = True
                yield event("content_block_delta", index=0, delta={"type": "text_delta", "text": delta["content"]})
            for partial in delta.get("tool_calls", []):
                call = calls.setdefault(partial.get("index", 0), {"id": "", "type": "function", "function": {"name": "", "arguments": ""}})
                call["id"] += partial.get("id", "")
                call["function"]["name"] += partial.get("function", {}).get("name", "")
                call["function"]["arguments"] += partial.get("function", {}).get("arguments", "")
    if text_started:
        yield event("content_block_stop", index=0)
    if finish is None:
        raise ValueError("Upstream stream ended without a finish reason")
    ordered = [calls[k] for k in sorted(calls)]
    if finish == "tool_calls" and not ordered:
        raise ValueError("Upstream claimed tool use without a valid call")
    validate_calls(ordered, tools, tool_choice)
    for index, call in enumerate(ordered, start=int(text_started)):
        f = call["function"]
        yield event("content_block_start", index=index, content_block={"type": "tool_use", "id": call["id"], "name": f["name"], "input": {}})
        yield event("content_block_delta", index=index, delta={"type": "input_json_delta", "partial_json": f["arguments"]})
        yield event("content_block_stop", index=index)
    yield event("message_delta", delta={"stop_reason": stop_reason(finish), "stop_sequence": None}, usage={"output_tokens": output_tokens})
    yield event("message_stop")


async def openai_stream(lines, tools, tool_choice=None):
    """Stream text immediately; emit tool calls only after complete validation."""
    calls, last, finish, usage = {}, None, None, None
    async for line in lines:
        if not line.startswith("data: ") or line[6:] == "[DONE]":
            continue
        chunk = json.loads(line[6:])
        chunk["model"] = ALIAS
        if chunk.get("usage"):
            usage = chunk["usage"]
        last = chunk
        content_choices = []
        for choice in chunk.get("choices", []):
            finish = choice.get("finish_reason") or finish
            delta = dict(choice.get("delta", {}))
            for part in delta.pop("tool_calls", []):
                call = calls.setdefault(part.get("index", 0), {"id": "", "type": "function", "function": {"name": "", "arguments": ""}})
                call["id"] += part.get("id", "")
                call["function"]["name"] += part.get("function", {}).get("name", "")
                call["function"]["arguments"] += part.get("function", {}).get("arguments", "")
            if delta:
                content_choices.append({**choice, "delta": delta, "finish_reason": None})
        if content_choices:
            yield "data: " + json.dumps({**chunk, "choices": content_choices}) + "\n\n"
    if last is None or finish is None:
        raise ValueError("Incomplete upstream stream")
    ordered = [calls[k] for k in sorted(calls)]
    if finish == "tool_calls" and not ordered:
        raise ValueError("Upstream claimed tool use without a valid call")
    validate_calls(ordered, tools, tool_choice)
    if ordered:
        yield "data: " + json.dumps({**last, "choices": [{"index": 0, "delta": {"tool_calls": [dict(c,index=i) for i,c in enumerate(ordered)]}, "finish_reason": None}], "usage": None}) + "\n\n"
    yield "data: " + json.dumps({**last, "choices": [{"index": 0, "delta": {}, "finish_reason": finish}], "usage": usage}) + "\n\n"
    yield "data: [DONE]\n\n"


def create_app(upstream, model_path, api_key, tokenizer, *, transport=None, context_limit=16384):
    parsed = urlparse(upstream)
    if parsed.hostname not in ("127.0.0.1", "localhost", "::1") or parsed.scheme != "http":
        raise ValueError("This launcher only supports a loopback inference backend")
    if len(api_key) < 24:
        raise ValueError("Use a generated local API key with at least 24 characters")
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    semaphore = asyncio.Semaphore(1)
    metrics = {"requests": 0, "completed": 0, "errors": 0, "input_tokens": 0}

    async def authenticate(request):
        authorization = request.headers.get("authorization", "")
        supplied = request.headers.get("x-api-key", "") or (authorization[7:] if authorization.startswith("Bearer ") else "")
        if not secrets.compare_digest(supplied, api_key):
            raise HTTPException(401, "Invalid local API key")

    async def body_for(request, anthropic=False):
        await authenticate(request)
        raw = bytearray()
        async for chunk in request.stream():
            raw.extend(chunk)
            if len(raw) > MAX_BODY:
                raise HTTPException(413, "Request body exceeds 4 MiB")
        try:
            body = json.loads(raw)
            if not isinstance(body, dict):
                raise ValueError()
            body = anthropic_to_openai(body) if anthropic else copy.deepcopy(body)
            allowed = {"model", "messages", "tools", "tool_choice", "max_tokens", "max_completion_tokens",
                       "stream", "stream_options", "temperature", "top_p", "top_k", "stop", "seed", "n",
                       "presence_penalty", "frequency_penalty", "metadata", "user", "parallel_tool_calls"}
            if set(body) - allowed:
                raise HTTPException(400, "Unsupported request fields for this fixed model profile")
            if body.get("model") not in (ALIAS, model_path):
                raise HTTPException(404, "Unknown model; use ruflo-code-4b")
            if not isinstance(body.get("messages"), list) or not body["messages"]:
                raise HTTPException(400, "messages must be a nonempty list")
            for message in body["messages"]:
                message["content"] = text_content(message.get("content", ""))
            requested = body.get("max_completion_tokens", body.get("max_tokens", 1024))
            if type(requested) is not int or not 1 <= requested <= 8192:
                raise HTTPException(400, "max_tokens must be between 1 and 8192")
            body.pop("max_completion_tokens", None)
            body.update({"max_tokens": requested, "model": model_path,
                         "chat_template_kwargs": {"enable_thinking": False}})
            if body.get("n", 1) != 1:
                raise HTTPException(400, "Only n=1 is supported")
            tools = body.get("tools") or []
            template_messages = copy.deepcopy(body["messages"])
            for message in template_messages:
                for call in message.get("tool_calls", []):
                    arguments = call["function"]["arguments"]
                    if isinstance(arguments, str):
                        call["function"]["arguments"] = json.loads(arguments)
            tokens = tokenizer.apply_chat_template(template_messages, tools=tools or None,
                tokenize=True, add_generation_prompt=True, enable_thinking=False)
            if len(tokens) + requested > context_limit:
                raise HTTPException(400, f"Context exceeds this profile's {context_limit} token limit; use the router fallback")
            return body, len(tokens)
        except (KeyError, ValueError, TypeError) as e:
            raise HTTPException(400, "Invalid request structure") from e

    @app.get("/health")
    async def health():
        return {"status": "gateway_ready", "model": ALIAS, "context_limit": context_limit,
                "inference_backend_health": "check /v1/models and a completion"}

    @app.get("/v1/models")
    async def models(request: Request):
        await authenticate(request)
        return {"object": "list", "data": [{"id": ALIAS, "object": "model", "created": 0, "owned_by": "local"}]}

    @app.get("/metrics")
    async def counts(request: Request):
        await authenticate(request)
        return metrics

    @app.post("/v1/messages/count_tokens")
    async def count_tokens(request: Request):
        _, count = await body_for(request, True)
        return {"input_tokens": count}

    async def completion(request, anthropic):
        body, input_tokens = await body_for(request, anthropic)
        try:
            await asyncio.wait_for(semaphore.acquire(), timeout=5)
        except TimeoutError:
            raise HTTPException(429, "Local model busy; retry or use the router fallback")
        metrics["requests"] += 1
        metrics["input_tokens"] += input_tokens
        client = httpx.AsyncClient(timeout=httpx.Timeout(180, connect=5), transport=transport)
        try:
            if body.get("stream"):
                body["stream_options"] = {"include_usage": True}
                upstream_request = client.build_request("POST", upstream + "/v1/chat/completions", json=body)
                response = await client.send(upstream_request, stream=True)
                if response.status_code != 200:
                    await response.aclose()
                    raise HTTPException(502, "Inference backend rejected request")
                async def generate():
                    try:
                        if anthropic:
                            async for item in anthropic_stream(response.aiter_lines(), body.get("tools", []), input_tokens, body.get("tool_choice")):
                                yield item
                        else:
                            async for item in openai_stream(response.aiter_lines(), body.get("tools", []), body.get("tool_choice")):
                                yield item
                        metrics["completed"] += 1
                    except Exception:
                        metrics["errors"] += 1
                        if anthropic:
                            yield event("error", error={"type": "api_error", "message": "Incomplete or invalid inference response"})
                        else:
                            yield 'data: {"error":{"message":"Incomplete inference response"}}\n\n'
                    finally:
                        await response.aclose()
                        await client.aclose()
                        semaphore.release()
                return StreamingResponse(generate(), media_type="text/event-stream")
            response = await client.post(upstream + "/v1/chat/completions", json=body)
            response.raise_for_status()
            data = response.json()
            data["model"] = ALIAS
            if data["choices"][0].get("finish_reason") == "tool_calls" and not data["choices"][0]["message"].get("tool_calls"):
                raise ValueError("Upstream claimed tool use without a valid call")
            validate_calls(data["choices"][0]["message"].get("tool_calls", []), body.get("tools", []), body.get("tool_choice"))
            result = to_anthropic(data, body.get("tools", [])) if anthropic else data
            metrics["completed"] += 1
            await client.aclose()
            semaphore.release()
            return JSONResponse(result)
        except Exception as e:
            metrics["errors"] += 1
            await client.aclose()
            semaphore.release()
            if isinstance(e, HTTPException):
                raise
            raise HTTPException(502, "Inference backend failed or returned invalid output") from e

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        return await completion(request, False)

    @app.post("/v1/messages")
    async def messages(request: Request):
        return await completion(request, True)

    return app
