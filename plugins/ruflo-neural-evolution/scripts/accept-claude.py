"""Exercise the native plugin bridge through installed Claude Code and local CCR.

Run only after model evaluation finishes. Uses normal permissions, one allowed
read tool, fresh local fixtures and a separate Claude configuration directory.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import time


PLUGIN = Path(__file__).resolve().parents[1]


def digest(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def write_json(path, value):
    with path.open("x") as output:
        output.write(json.dumps(value, indent=2) + "\n")
    path.chmod(0o600)


def plugin_inputs():
    return {path.relative_to(PLUGIN).as_posix(): digest(path)
            for path in sorted(PLUGIN.rglob("*"))
            if path.is_file() and "__pycache__" not in path.parts}


def completed_tools(events):
    calls = {}
    results = []
    for event in events:
        message = event.get("message", {})
        if not isinstance(message, dict):
            continue
        for block in message.get("content", []):
            if not isinstance(block, dict):
                continue
            if event.get("type") == "assistant" and block.get("type") == "tool_use":
                calls[block["id"]] = block
            elif event.get("type") == "user" and block.get("type") == "tool_result":
                call = calls.get(block.get("tool_use_id"))
                if call is None or block.get("is_error", False):
                    continue
                text = block.get("content", "")
                if isinstance(text, list):
                    text = "\n".join(b.get("text", "") for b in text if b.get("type") == "text")
                results.append({**call, "result_text": text})
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=PLUGIN.parents[1])
    parser.add_argument("--label", default="neural-plugin-selected")
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--binary-sha256", required=True)
    args = parser.parse_args()
    if not args.label.replace("-", "").isalnum():
        parser.error("label must be an identifier")
    binary = args.binary.resolve(strict=True)
    if digest(binary) != args.binary_sha256:
        parser.error("native executable does not match the expected digest")
    root = args.project_root.resolve(strict=True)
    claude = shutil.which("claude")
    node = shutil.which("node")
    if not claude or not node:
        parser.error("Claude Code and Node.js must be installed")
    inputs_sha256 = plugin_inputs()
    run = root / "runtime/integration" / args.label
    run.mkdir(parents=True, mode=0o700, exist_ok=False)
    fixture = run / "fixture"
    fixture.mkdir(mode=0o700)
    output = run / "native-output"
    output.mkdir(mode=0o700)
    config = run / "claude-config"
    config.mkdir(mode=0o700)
    header = json.dumps({"layer.weight": {"dtype": "BF16", "shape": [2, 2], "data_offsets": [0, 8]}}, separators=(",", ":")).encode()
    header += b" " * ((-len(header)) % 8)
    checkpoint = fixture / "inspect.safetensors"
    checkpoint.write_bytes(struct.pack("<Q", len(header)) + header + struct.pack("<4H", 0x3F80, 0x4000, 0x4040, 0x4080))
    checkpoint.chmod(0o600)
    expected = digest(checkpoint)
    tensor_list = fixture / "tensors.json"
    write_json(tensor_list, ["layer.weight"])
    runtime_config = run / "native-runtime.json"
    write_json(runtime_config, {"schema": "ruflo-neural-runtime/v1", "binaryPath": str(binary), "binarySha256": args.binary_sha256,
        "readRoots": [str(fixture)], "writeRoot": str(output), "policySha256s": [], "timeoutMs": 10000})
    tool = "mcp__neural__weight_inspect"
    mcp = {"mcpServers": {"neural": {"command": node, "args": [str(PLUGIN / "scripts/server.mjs")], "env": {"RUFLO_NEURAL_CONFIG": str(runtime_config)}}}}
    token = (root / "runtime/ccr/client-key").read_text().strip()
    env = {key: os.environ[key] for key in ("PATH", "HOME", "USER", "TMPDIR", "LANG") if key in os.environ}
    env.update({"CLAUDE_CONFIG_DIR": str(config), "ANTHROPIC_BASE_URL": "http://127.0.0.1:9110", "ANTHROPIC_AUTH_TOKEN": token,
        "ANTHROPIC_MODEL": "ruflo-local/ruflo-code-4b", "CLAUDE_CODE_MAX_OUTPUT_TOKENS": "1024", "CLAUDE_CODE_MAX_RETRIES": "0",
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1", "DISABLE_AUTOUPDATER": "1", "MAX_THINKING_TOKENS": "0", "CLAUDE_CODE_DISABLE_ADAPTIVE_THINKING": "1"})
    settings = {"permissions": {"allow": [tool]}, "disableAllHooks": True, "autoMemoryEnabled": False, "alwaysThinkingEnabled": False}
    prompt = f'Call {tool} with source={json.dumps(str(checkpoint))} and tensorList={json.dumps(str(tensor_list))}. Return the source_sha256 from its successful result. Do not calculate or guess it. Use no other tools.'
    command = [claude, "--print", "--model", "ruflo-local/ruflo-code-4b", "--output-format", "stream-json", "--verbose",
        "--tools", "", "--allowedTools", tool, "--permission-mode", "dontAsk", "--no-session-persistence", "--strict-mcp-config",
        "--mcp-config", json.dumps(mcp), "--plugin-dir", str(PLUGIN), "--setting-sources", "", "--settings", json.dumps(settings),
        "--restricted", "--no-chrome", prompt]
    started = time.monotonic()
    result = subprocess.run(command, cwd=fixture, env=env, text=True, capture_output=True, timeout=180)
    stdout = result.stdout.replace(token, "[REDACTED]")
    stderr = result.stderr.replace(token, "[REDACTED]")
    (run / "events.jsonl").write_text(stdout)
    (run / "stderr.log").write_text(stderr)
    events = []
    for line in stdout.splitlines():
        try:
            events.append(json.loads(line))
        except ValueError:
            pass
    calls = completed_tools(events)
    complete = any(call["name"] == tool and call.get("input") == {"source": str(checkpoint), "tensorList": str(tensor_list)} and expected in call["result_text"] for call in calls)
    final = next((event for event in reversed(events) if event.get("type") == "result"), {})
    server = json.loads((root / "runtime/server/process.json").read_text())
    loaded = json.loads((root / "runtime/server/backend-state.json").read_text())
    binding = loaded["backend_pid"] == server["backend_pid"] and loaded["binding_verified"]
    inputs_unchanged = inputs_sha256 == plugin_inputs()
    passed = result.returncode == 0 and not final.get("is_error", True) and complete and expected in final.get("result", "") and binding and inputs_unchanged
    router_path = root / "vendor/ccr/node_modules/@musistudio/claude-code-router/package.json"
    router_version = json.loads(router_path.read_text())["version"] if router_path.exists() else None
    receipt = {"schema": "ruflo-neural-plugin-claude-acceptance/v1", "passed": passed, "timestamp": datetime.now(timezone.utc).isoformat(),
        "seconds": round(time.monotonic() - started, 3), "exitCode": result.returncode, "tool": tool, "toolCompleted": complete,
        "sourceSha256": expected, "nativeRuntimeSha256": args.binary_sha256, "claudeVersion": subprocess.check_output([claude, "--version"], text=True).strip(),
        "routerVersion": router_version, "model": "ruflo-local/ruflo-code-4b", "adapterSha256": loaded["adapter_sha256"], "backendBindingVerified": bool(binding),
        "pluginDirectory": str(PLUGIN), "strictMcpConfiguration": True, "builtInTools": [], "permissionBypass": False,
        "pluginInputsSha256": inputs_sha256, "pluginInputsUnchangedAfterRun": inputs_unchanged,
        "eventsSha256": digest(run / "events.jsonl"), "stderrSha256": digest(run / "stderr.log"),
        "fixtureOnly": True, "productionActivation": False, "path": "Claude Code -> CCR -> MLX coding model -> local native MCP bridge"}
    write_json(run / "receipt.json", receipt)
    print(json.dumps(receipt, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
