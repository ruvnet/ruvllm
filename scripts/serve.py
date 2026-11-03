import argparse
import json
import os
from pathlib import Path
import secrets
import signal
import socket
import subprocess
import sys
import time

import httpx
import uvicorn
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ruflo_model.gateway import ALIAS, create_app
from ruflo_model.common import write_json, digest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str(ROOT / "models/base"))
    parser.add_argument("--adapter")
    parser.add_argument("--port", type=int, default=9100)
    parser.add_argument("--backend-port", type=int, default=9101)
    args = parser.parse_args()
    args.model = str(Path(args.model).resolve())
    args.adapter = str(Path(args.adapter).resolve()) if args.adapter else None
    for port in (args.port, args.backend_port):
        with socket.socket() as probe:
            try:
                probe.bind(("127.0.0.1", port))
            except OSError as error:
                raise RuntimeError(f"Port {port} is occupied; stop the existing service first") from error
    runtime = ROOT / "runtime/server"
    runtime.mkdir(parents=True, exist_ok=True)
    keyfile = runtime / "api-key"
    if not keyfile.exists():
        fd = os.open(keyfile, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as f:
            f.write(secrets.token_urlsafe(32))
    os.chmod(keyfile, 0o600)
    key = keyfile.read_text().strip()
    def stop_during_startup(*_):
        raise SystemExit(0)
    # Uvicorn takes over signals while serving. This covers model startup too.
    signal.signal(signal.SIGTERM, stop_during_startup)
    command = [sys.executable, "-m", "ruflo_model.backend", "--model", args.model, "--host", "127.0.0.1",
               "--port", str(args.backend_port), "--chat-template-args", '{"enable_thinking":false}']
    if args.adapter:
        command += ["--adapter-path", args.adapter]
    (runtime / "backend-state.json").unlink(missing_ok=True)
    with (runtime / "backend.log").open("a") as log:
        backend = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        write_json(runtime / "process.json", {"launcher_pid": os.getpid(), "backend_pid": backend.pid,
            "gateway": f"http://127.0.0.1:{args.port}", "model": args.model, "adapter": args.adapter})
        try:
            for _ in range(120):
                if backend.poll() is not None:
                    raise RuntimeError("Backend exited; see runtime/server/backend.log")
                try:
                    if httpx.get(f"http://127.0.0.1:{args.backend_port}/v1/models", timeout=1).status_code == 200 and (runtime / "backend-state.json").exists():
                        break
                except httpx.HTTPError:
                    pass
                time.sleep(1)
            else:
                raise TimeoutError("Backend did not become ready")
            loaded = json.loads((runtime / "backend-state.json").read_text())
            expected_sha = digest(Path(args.adapter) / "adapters.safetensors") if args.adapter else None
            if not loaded["binding_verified"] or loaded["backend_pid"] != backend.pid or loaded["adapter_sha256"] != expected_sha:
                raise RuntimeError("Backend did not verify the configured adapter")
            tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
            app = create_app(f"http://127.0.0.1:{args.backend_port}", args.model, key, tokenizer)
            print(f"Model: {ALIAS}. Router URL: http://127.0.0.1:{args.port}/v1. Local key file: {keyfile}", flush=True)
            uvicorn.run(app, host="127.0.0.1", port=args.port, access_log=False)
        finally:
            if backend.poll() is None:
                os.killpg(backend.pid, signal.SIGTERM)
                try:
                    backend.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(backend.pid, signal.SIGKILL)


if __name__ == "__main__":
    main()
