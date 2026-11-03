"""Verify a real Read, Edit, Bash cycle using an isolated arithmetic fixture."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ruflo_model.common import now, write_json, digest
from ruflo_model.scoring import score_code
from ruflo_model.integration import completed_tools


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", default="selected-coding")
    a = parser.parse_args()
    if not a.label.replace("-", "").isalnum():
        parser.error("label must be an identifier")
    run = ROOT / "runtime/integration" / a.label
    fixture = run / "fixture"
    fixture.mkdir(parents=True, exist_ok=False)
    bug = fixture / "discount.py"
    bug.write_text("def discount(price, percent):\n    return price * (1 + percent / 100)\n")
    before = digest(bug)
    assertions = "assert discount(100,20)==80\nassert discount(50,0)==50\nassert discount(80,100)==0\nassert discount(0,70)==0\n"
    (fixture / "tests.py").write_text("from discount import discount\n" + assertions + "print('4 tests passed')\n")
    config = run / "claude-config"
    config.mkdir()
    env = {k: os.environ[k] for k in ("PATH", "HOME", "USER", "TMPDIR", "LANG") if k in os.environ}
    env.update({"CLAUDE_CONFIG_DIR": str(config), "ANTHROPIC_BASE_URL": "http://127.0.0.1:9110",
        "ANTHROPIC_AUTH_TOKEN": (ROOT / "runtime/ccr/client-key").read_text().strip(),
        "ANTHROPIC_MODEL": "ruflo-local/ruflo-code-4b", "CLAUDE_CODE_MAX_OUTPUT_TOKENS": "1536",
        "CLAUDE_CODE_MAX_RETRIES": "0", "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
        "DISABLE_AUTOUPDATER": "1", "MAX_THINKING_TOKENS": "0", "CLAUDE_CODE_DISABLE_ADAPTIVE_THINKING": "1"})
    test_command = f"{sys.executable} tests.py"
    permissions = [f"Read(/{fixture}/**)", f"Edit(/{fixture}/**)", f"Bash({test_command})"]
    settings = {"permissions": {"allow": permissions}, "disableAllHooks": True,
                "autoMemoryEnabled": False, "alwaysThinkingEnabled": False,
                "sandbox": {"enabled": True, "autoAllowBashIfSandboxed": False}}
    claude = shutil.which("claude")
    if not claude:
        raise RuntimeError("Claude Code is not installed")
    command = [claude, "--print", "--model", "ruflo-local/ruflo-code-4b", "--output-format", "stream-json", "--verbose",
        "--tools", "Read,Edit,Bash", "--allowedTools", *permissions, "--permission-mode", "dontAsk",
        "--no-session-persistence", "--strict-mcp-config", "--mcp-config", '{"mcpServers":{}}',
        "--setting-sources", "", "--settings", json.dumps(settings), "--restricted", "--no-chrome",
        f"Read discount.py and tests.py. Fix discount.py using Edit so discount subtracts the requested percentage. Then use Bash to run exactly: {test_command}. Report the test result. Work only in the current directory."]
    started = time.monotonic()
    result = subprocess.run(command, cwd=fixture, env=env, text=True, capture_output=True, timeout=240)
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
    correct, reason = score_code({"tests": assertions}, bug.read_text())
    completed = completed_tools(events)
    names = {c["name"] for c in completed}
    bash_completed = any(c["name"] == "Bash" and c["input"].get("command") == test_command and "4 tests passed" in c["result_text"] for c in completed)
    passed = result.returncode == 0 and not final.get("is_error", True) and correct and {"Read", "Edit", "Bash"} <= names and bash_completed and before != digest(bug)
    server = json.loads((ROOT / "runtime/server/process.json").read_text())
    loaded = json.loads((ROOT / "runtime/server/backend-state.json").read_text())
    binding = loaded["backend_pid"] == server["backend_pid"] and loaded["binding_verified"]
    passed = passed and binding
    receipt = {"passed": passed, "timestamp": now(), "seconds": round(time.monotonic()-started,3),
        "exit_code": result.returncode, "independent_tests_passed": correct, "test_reason": reason,
        "tool_names": [c["name"] for c in calls], "completed_tool_names": sorted(names), "bash_tests_completed": bash_completed,
        "before_sha256": before, "after_sha256": digest(bug),
        "adapter_sha256": loaded["adapter_sha256"], "backend_binding_verified": binding,
        "matched_adapter_tensors": loaded["matched_adapter_tensors"],
        "claude_version": subprocess.check_output([claude,"--version"],text=True).strip(),
        "router_version": "3.1.3", "permission_bypass": False, "fixture_only": True}
    write_json(run / "receipt.json", receipt)
    print(json.dumps(receipt, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
