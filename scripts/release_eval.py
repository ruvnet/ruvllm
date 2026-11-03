"""Frozen external validation. Never feeds its results into candidate selection."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ruflo_model.common import now, write_json


def main():
    run = ROOT / "runtime/pilot-v2"
    out = ROOT / "runtime/release-validation"
    out.mkdir(parents=True, exist_ok=True)
    state = {"pid": os.getpid(), "status": "running", "started_at": now(), "selection_use": False}
    stopped = False
    started = time.monotonic()
    def stop(*_):
        nonlocal stopped
        stopped = True
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    def heartbeat():
        state["heartbeat_at"] = now()
        write_json(out / "state.json", state)
    def guard():
        heartbeat()
        if stopped or (out / "STOP").exists():
            raise InterruptedError("release validation stopped")
        if time.monotonic() - started > 14400:
            raise TimeoutError("release validation reached four hour cap")
    def evaluate(name, adapter=None):
        result = out / name
        if (result / "summary.json").exists():
            return json.loads((result / "summary.json").read_text())
        if result.exists():
            raise RuntimeError("Partial external evaluation exists; inspect before retry")
        state["stage"] = name
        cmd = [sys.executable, "-m", "ruflo_model.evaluate", "--model", str(ROOT / "models/base-4bit"),
               "--suite", str(ROOT / "data/external/humaneval.jsonl"), "--output", str(result)]
        if adapter:
            cmd += ["--adapter", adapter]
        with (out / (name + ".log")).open("a") as log:
            child = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            state["child_pid"] = child.pid
            try:
                while child.poll() is None:
                    guard()
                    time.sleep(2)
                if child.returncode:
                    raise RuntimeError("external evaluation failed")
            finally:
                if child.poll() is None:
                    os.killpg(child.pid, signal.SIGTERM)
                    try:
                        child.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                        child.wait()
                state.pop("child_pid", None)
        return json.loads((result / "summary.json").read_text())
    try:
        reference = json.loads((ROOT / "runtime/humaneval-reference-check.json").read_text())
        if reference["passed"] != 164:
            raise RuntimeError("HumanEval reference preflight did not pass")
        baseline = evaluate("baseline-humaneval")
        state["stage"] = "waiting-for-frozen-selection"
        while True:
            guard()
            worker = json.loads((run / "state.json").read_text())
            if worker["status"] in ("failed", "stopped"):
                raise RuntimeError("optimization worker did not finish")
            if worker["status"] == "completed":
                break
            time.sleep(5)
        selection = json.loads((run / "selection.json").read_text())
        adapter = selection["selected_adapter"]
        selected = evaluate("selected-humaneval", adapter) if adapter else baseline
        report = {"baseline": baseline, "selected": selected, "selected_adapter": adapter,
                  "no_aggregate_regression": selected["passed"] >= baseline["passed"],
                  "synthetic_release_passed": worker["release_gate_passed"],
                  "candidate_qualified": bool(adapter) and worker["release_gate_passed"] and selected["passed"] >= baseline["passed"],
                  "sota_verified": False, "selection_use": False, "completed_at": now()}
        write_json(out / "report.json", report)
        state.update(status="completed", stage="finished", candidate_qualified=report["candidate_qualified"])
    except InterruptedError as e:
        state.update(status="stopped", error=str(e))
    except Exception as e:
        state.update(status="failed", error=f"{type(e).__name__}: {e}")
    finally:
        heartbeat()


if __name__ == "__main__":
    main()
