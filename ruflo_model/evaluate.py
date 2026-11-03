import argparse
import fcntl
import json
import platform
import re
import time
from pathlib import Path

from .common import ROOT, digest, now, read_jsonl, write_json
from .scoring import score_code, score_tool, summarize


def parse_calls(text, tokenizer, tools):
    if not tokenizer.has_tool_calling:
        return []
    pattern = re.escape(tokenizer.tool_call_start) + r"(.*?)" + re.escape(tokenizer.tool_call_end)
    result = []
    for content in re.findall(pattern, text, re.S):
        parsed = tokenizer.tool_parser(content.strip(), tools)
        result.extend(parsed if isinstance(parsed, list) else [parsed])
    return result


def evaluate(model_path, suite, output, adapter=None, limit=None, max_tokens=768):
    (ROOT / "runtime").mkdir(exist_ok=True)
    with (ROOT / "runtime/inference.lock").open("a") as lock:
        print("Waiting for exclusive local inference slot", flush=True)
        fcntl.flock(lock, fcntl.LOCK_EX)
        return _evaluate(model_path, suite, output, adapter, limit, max_tokens)


def _evaluate(model_path, suite, output, adapter=None, limit=None, max_tokens=768):
    import mlx.core as mx
    from mlx_lm import load, stream_generate
    from mlx_lm.sample_utils import make_sampler
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    cases = read_jsonl(suite)
    if limit is not None:
        cases = cases[:limit]
    pin = json.loads((ROOT / "model_pin.json").read_text())
    write_json(output / "protocol.json", {
        "started_at": now(), "model_pin": pin, "model_path": str(model_path),
        "adapter": str(adapter) if adapter else None,
        "model_config_sha256": digest(Path(model_path) / "config.json"),
        "adapter_sha256": digest(Path(adapter) / "adapters.safetensors") if adapter else None,
        "suite_sha256": digest(suite), "cases": len(cases), "max_tokens": max_tokens,
        "temperature": 0, "seed": 20261102, "enable_thinking": False,
        "platform": platform.platform(), "mlx": mx.__version__, "scoring": "executable tests and schema plus exact arguments"})
    model, tokenizer = load(str(model_path), adapter_path=str(adapter) if adapter else None)
    rows = []
    with (output / "responses.jsonl").open("x") as log:
        for case in cases:
            mx.random.seed(20261102)
            started = time.monotonic()
            prompt = tokenizer.apply_chat_template(case["messages"], tools=case["tools"] or None,
                add_generation_prompt=True, tokenize=True, enable_thinking=False)
            chunks, last, calls = [], None, []
            try:
                for response in stream_generate(model, tokenizer, prompt, max_tokens=max_tokens,
                        sampler=make_sampler(temp=0)):
                    chunks.append(response.text)
                    last = response
                text = "".join(chunks)
                if case["kind"] == "tool":
                    calls = parse_calls(text, tokenizer, case["tools"])
                    passed, reason = score_tool(case, calls)
                else:
                    passed, reason = score_code(case, text)
                if last is None or last.finish_reason == "length":
                    passed, reason = False, "truncated response"
            except Exception as e:
                text, passed, reason = "".join(chunks), False, f"{type(e).__name__}: {str(e)[:300]}"
            row = {"id": case["id"], "kind": case["kind"], "family": case["family"],
                   "passed": passed, "reason": reason, "text": text, "tool_calls": calls,
                   "finish_reason": last.finish_reason if last else "error",
                   "input_tokens": len(prompt), "output_tokens": last.generation_tokens if last else 0,
                   "seconds": round(time.monotonic() - started, 3),
                   "tokens_per_second": round(last.generation_tps, 2) if last else 0,
                   "peak_memory_gb": round(mx.get_peak_memory() / 1e9, 3)}
            log.write(json.dumps(row) + "\n")
            log.flush()
            rows.append(row)
            print(json.dumps({k: row[k] for k in ("id", "passed", "reason", "output_tokens", "seconds")}), flush=True)
            mx.clear_cache()
    summary = summarize(rows)
    summary.update({"completed_at": now(), "response_sha256": digest(output / "responses.jsonl")})
    write_json(output / "summary.json", summary)
    print(json.dumps(summary), flush=True)
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=str(ROOT / "models/base"))
    p.add_argument("--suite", default=str(ROOT / "data/dev.jsonl"))
    p.add_argument("--output", required=True)
    p.add_argument("--adapter")
    p.add_argument("--limit", type=int)
    p.add_argument("--max-tokens", type=int, default=768)
    a = p.parse_args()
    evaluate(a.model, a.suite, a.output, a.adapter, a.limit, a.max_tokens)


if __name__ == "__main__":
    main()
