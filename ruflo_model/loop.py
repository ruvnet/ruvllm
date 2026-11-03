"""Finite, resumable controller. Every expensive stage is a bounded subprocess.

OS flock prevents two workers. SIGTERM and the STOP file cancel the process group.
Failed and partial runs are preserved; resumption never silently overwrites them.
"""
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from .common import ROOT, digest, now, write_json
from .selection import promote


def run(run_dir):
    run_dir = Path(run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    lock = (ROOT / "runtime/worker.lock").open("a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    config = json.loads((ROOT / "configs/experiment.json").read_text())
    state_path = run_dir / "state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {
        "started_at": now(), "started_epoch": time.time(), "iteration": 0, "completed_stages": [], "trials": []}
    if state.get("error"):
        state.setdefault("previous_attempts", []).append({"error": state.pop("error"), "status": state["status"],
            "heartbeat_at": state["heartbeat_at"], "config_sha256": state["config_sha256"]})
    state.update({"pid": os.getpid(), "status": "running", "local_worker_cloud_dispatches": 0,
                  "cloud_budget_usd": config["cloud_budget_usd"], "config_sha256": digest(ROOT / "configs/experiment.json")})
    stop = False
    def request_stop(*_):
        nonlocal stop
        stop = True
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    def heartbeat():
        state["heartbeat_at"] = now()
        budget_path = ROOT / "runtime/cloud-budget.json"
        if budget_path.exists():
            ledger = json.loads(budget_path.read_text())
            state["cloud_reserved_max_usd"] = sum(r["reserved_micro_usd"] for r in ledger["reservations"]) / 1e6
        write_json(state_path, state)

    def stage(name, command):
        if name in state["completed_stages"]:
            return
        if stop or (run_dir / "STOP").exists():
            raise InterruptedError("stop requested")
        state.update({"stage": name, "stage_started_at": now()})
        heartbeat()
        attempt = 1 + len(list(run_dir.glob(name + ".*.log")))
        with (run_dir / f"{name}.{attempt}.log").open("x") as log:
            child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            started = time.time()
            state["child_pid"] = child.pid
            try:
                while child.poll() is None:
                    heartbeat()
                    if stop or (run_dir / "STOP").exists():
                        raise InterruptedError("stop requested")
                    if time.time() - started > config["stage_timeout_seconds"] or time.time() - state["started_epoch"] > config["wall_timeout_seconds"]:
                        raise TimeoutError("experiment time cap reached")
                    time.sleep(2)
                if child.returncode:
                    raise RuntimeError(f"stage {name} exited {child.returncode}; see {log.name}")
            finally:
                if child.poll() is None:
                    os.killpg(child.pid, signal.SIGTERM)
                    try:
                        child.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                        child.wait()
                state.pop("child_pid", None)
        state["completed_stages"].append(name)
        heartbeat()

    def evaluate(name, split, adapter=None):
        out = run_dir / name
        if name not in state["completed_stages"] and out.exists():
            raise RuntimeError(f"partial evidence at {out}; inspect and archive before explicit retry")
        command = [sys.executable, "-m", "ruflo_model.evaluate", "--model", str(ROOT / config["model"]), "--suite", str(ROOT / f"data/{split}.jsonl"), "--output", str(out)]
        if adapter:
            command += ["--adapter", str(adapter)]
        stage(name, command)
        return json.loads((out / "summary.json").read_text())

    try:
        manifest = json.loads((ROOT / "data/manifest.json").read_text())
        for file, sha in manifest["files"].items():
            if digest(ROOT / "data" / file) != sha:
                raise ValueError("dataset hash mismatch")
        baseline = evaluate("baseline-dev", "dev")
        state["iteration"] = 1
        train_dir = run_dir / "training"
        if config.get("training_backend") == "cloud":
            stage("train", [sys.executable, "-m", "ruflo_model.cloud_train", str(train_dir)])
        else:
            stage("train", [sys.executable, "-m", "mlx_lm", "lora", "--config", config["train_config"], "--adapter-path", str(train_dir)])
        best, best_adapter, stale = baseline, None, 0
        for step in config["candidate_steps"][:config["max_iterations"] - 1]:
            name = f"step-{step:03d}"
            adapter = run_dir / name
            adapter.mkdir(exist_ok=True)
            source = train_dir / f"{step:07d}_adapters.safetensors"
            if not source.exists():
                raise FileNotFoundError(source)
            shutil.copyfile(source, adapter / "adapters.safetensors")
            shutil.copyfile(train_dir / "adapter_config.json", adapter / "adapter_config.json")
            result = evaluate(name + "-dev", "dev", adapter)
            accepted, reason = promote(best, result, config["promotion"])
            trial = {"name": name, "accepted": accepted, "reason": reason,
                     "adapter_sha256": digest(adapter / "adapters.safetensors"), "summary": result}
            state["trials"] = [t for t in state["trials"] if t["name"] != name] + [trial]
            state["iteration"] += 1
            if accepted:
                best, best_adapter, stale = result, adapter, 0
            else:
                stale += 1
            heartbeat()
            if stale >= config["stop_after_nonimprovements"]:
                break
        state["selected_adapter"] = str(best_adapter) if best_adapter else None
        write_json(run_dir / "selection.json", {"selected_adapter": state["selected_adapter"],
            "baseline": baseline, "selected": best, "trials": state["trials"], "selected_at": now(),
            "release_used_for_selection": False})
        release_base = evaluate("baseline-release", "release")
        release = evaluate("selected-release", "release", best_adapter) if best_adapter else release_base
        release_ok, release_reason = promote(release_base, release, config["promotion"]) if best_adapter else (False, "no improved adapter selected")
        state.update({"status": "completed", "completed_at": now(), "stage": "finished",
                      "release_gate_passed": release_ok, "release_reason": release_reason,
                      "sota_verified": False})
    except InterruptedError as e:
        state.update({"status": "stopped", "error": str(e)})
    except Exception as e:
        state.update({"status": "failed", "error": f"{type(e).__name__}: {e}"})
    finally:
        heartbeat()
        lock.close()
    return state


if __name__ == "__main__":
    run(sys.argv[1])
