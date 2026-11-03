"""Run an isolated CCR 3.1.3 instance without editing global agent settings."""
import argparse
import json
import os
from pathlib import Path
import secrets
import signal
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ruflo_model.common import write_json


def local_key(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as f:
            f.write(secrets.token_urlsafe(32))
    return path.read_text().strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["start", "stop", "config"])
    args = parser.parse_args()
    root = ROOT / "runtime/ccr"
    config_dir = root / "home/.claude-code-router"
    config_dir.mkdir(parents=True, exist_ok=True)
    api_key = local_key(ROOT / "runtime/server/api-key")
    client_key = local_key(root / "client-key")
    management_key = local_key(root / "management-key")
    config = {"HOST": "127.0.0.1", "PORT": 9110, "APIKEY": client_key,
              "API_TIMEOUT_MS": 180000,
              "Providers": [{"name": "ruflo-local", "api_base_url": "http://127.0.0.1:9100/v1",
                  "api_key": api_key, "models": ["ruflo-code-4b"], "capabilities": [{"type": "openai_chat_completions", "baseUrl": "http://127.0.0.1:9100/v1"}]}],
              "preferredProvider": "ruflo-local", "Router": {"default": "ruflo-local,ruflo-code-4b",
                  "builtInRules": {"claude-code": {"enabled": False}, "codex": {"enabled": False}}, "rules": []},
              "gateway": {"enabled": True, "host": "127.0.0.1", "port": 9110, "corePort": 9111},
              "observability": {"requestLogs": False, "requestLogBodyCapture": "none"},
              "profile": {"enabled": False, "profiles": [], "claudeCode": {"enabled": False}, "codex": {"enabled": False}},
              "proxy": {"enabled": False, "systemProxy": False, "upstream": {"mode": "none"}}, "autoStart": False}
    if args.action == "config":
        safe = json.loads(json.dumps(config))
        safe["APIKEY"] = "<read runtime/ccr/client-key>"
        safe["Providers"][0]["api_key"] = "<read runtime/server/api-key>"
        print(json.dumps(safe, indent=2))
        return
    if args.action == "stop":
        receipt = json.loads((root / "process.json").read_text())
        os.kill(receipt["pid"], signal.SIGTERM)
        print("CCR stop requested")
        return
    if (root / "process.json").exists():
        receipt = json.loads((root / "process.json").read_text())
        try:
            os.kill(receipt["pid"], 0)
        except ProcessLookupError:
            pass
        else:
            raise RuntimeError("Existing scoped CCR process is still running")
    # Seed only a fresh instance. Current CCR migrates this legacy JSON to SQLite.
    if not (config_dir / "config.sqlite").exists():
        write_json(config_dir / "config.json", config)
        os.chmod(config_dir / "config.json", 0o600)
    env = dict(os.environ, CCR_INTERNAL_HOME_DIR=str(root / "home"),
               CCR_INTERNAL_APP_DATA_DIR=str(root / "app-data"), CCR_INTERNAL_USER_DATA_DIR=str(root / "user-data"),
               CCR_WEB_AUTH_TOKEN=management_key, CCR_REMOTE_SYNC_ENABLED="0")
    command = ["node", str(ROOT / "vendor/ccr/node_modules/@musistudio/claude-code-router/dist/main/cli.js"),
               "serve", "--host", "127.0.0.1", "--port", "9112", "--no-open"]
    with (root / "service.log").open("a") as log:
        proc = subprocess.Popen(command, env=env, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL, start_new_session=True)
    write_json(root / "process.json", {"pid": proc.pid, "gateway": "http://127.0.0.1:9110", "management": "http://127.0.0.1:9112"})
    print(json.dumps({"pid": proc.pid, "gateway": "http://127.0.0.1:9110", "client_key_file": str(root / "client-key")}))


if __name__ == "__main__":
    main()
