"""Exercise the installed Claude Code CLI through an isolated, real CCR instance."""
import argparse
import json
import os
from pathlib import Path
import secrets
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ruflo_model.common import now, write_json, digest
from ruflo_model.integration import completed_tools


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", default="baseline")
    args = parser.parse_args()
    if not args.label.replace("-", "").isalnum():
        parser.error("label must be an identifier")
    run = ROOT / "runtime/integration" / args.label
    run.mkdir(parents=True, exist_ok=False)
    fixture = run / "fixture"
    fixture.mkdir()
    nonce = "RUFLO_" + secrets.token_hex(8)
    (fixture / "verification.txt").write_text("The verification value is: " + nonce + "\n")
    config = run / "claude-config"
    config.mkdir()
    env = {k: os.environ[k] for k in ("PATH", "HOME", "USER", "TMPDIR", "LANG") if k in os.environ}
    env.update({"CLAUDE_CONFIG_DIR": str(config), "ANTHROPIC_BASE_URL": "http://127.0.0.1:9110",
        "ANTHROPIC_AUTH_TOKEN": (ROOT / "runtime/ccr/client-key").read_text().strip(),
        "ANTHROPIC_MODEL": "ruflo-local/ruflo-code-4b", "CLAUDE_CODE_MAX_OUTPUT_TOKENS": "1024",
        "CLAUDE_CODE_MAX_RETRIES": "0", "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
        "DISABLE_AUTOUPDATER": "1", "MAX_THINKING_TOKENS": "0", "CLAUDE_CODE_DISABLE_ADAPTIVE_THINKING": "1"})
    claude = shutil.which("claude")
    if not claude:
        raise RuntimeError("Claude Code CLI is not installed")
    settings = {"permissions": {"allow": [f"Read(/{fixture}/**)"]}, "disableAllHooks": True,
                "autoMemoryEnabled": False, "alwaysThinkingEnabled": False}
    command = [claude, "--print", "--model", "ruflo-local/ruflo-code-4b",
        "--output-format", "stream-json", "--verbose", "--tools", "Read", "--allowedTools", "Read",
        "--permission-mode", "dontAsk", "--no-session-persistence", "--strict-mcp-config",
        "--mcp-config", '{"mcpServers":{}}', "--setting-sources", "", "--settings", json.dumps(settings),
        "--restricted", "--no-chrome",
        "Use the Read tool to read verification.txt in the current directory. Then return only its verification value. Do not guess the value."]
    started = time.monotonic()
    result = subprocess.run(command, cwd=fixture, env=env, text=True, capture_output=True, timeout=180)
    (run / "events.jsonl").write_text(result.stdout)
    (run / "stderr.log").write_text(result.stderr)
    events = []
    for line in result.stdout.splitlines():
        try:
            events.append(json.loads(line))
        except ValueError:
            pass
    calls = [b for e in events if e.get("type") == "assistant" for b in e.get("message", {}).get("content", []) if b.get("type") == "tool_use"]
    final = next((e for e in reversed(events) if e.get("type") == "result"), {})
    read_completed = any(c["name"] == "Read" and nonce in c["result_text"] for c in completed_tools(events))
    passed = result.returncode == 0 and not final.get("is_error", True) and nonce in final.get("result", "") and read_completed
    receipt = {"passed": passed, "timestamp": now(), "exit_code": result.returncode,
        "seconds": round(time.monotonic() - started, 3), "tool_calls": [{"name": c.get("name"), "input": c.get("input")} for c in calls],
        "nonce_returned": nonce in final.get("result", ""), "read_completed": read_completed, "result_subtype": final.get("subtype"),
        "claude_version": subprocess.check_output([claude, "--version"], text=True).strip(),
        "router_version": "3.1.3", "path": "Claude Code -> CCR -> OpenAI gateway -> MLX model",
        "model": "ruflo-local/ruflo-code-4b", "fixture_only": True, "permission_bypass": False}
    server = json.loads((ROOT / "runtime/server/process.json").read_text())
    loaded = json.loads((ROOT / "runtime/server/backend-state.json").read_text())
    binding = loaded["backend_pid"] == server["backend_pid"] and loaded["binding_verified"]
    receipt["adapter_sha256"] = loaded["adapter_sha256"]
    receipt["backend_binding_verified"] = binding
    receipt["matched_adapter_tensors"] = loaded["matched_adapter_tensors"]
    passed = passed and binding
    receipt["passed"] = passed
    write_json(run / "receipt.json", receipt)
    print(json.dumps(receipt, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
