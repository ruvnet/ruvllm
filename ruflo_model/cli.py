import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

from .common import ROOT, now, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["start", "resume", "status", "stop", "data"])
    parser.add_argument("--run", default="pilot-v1")
    args = parser.parse_args()
    if not args.run.replace("-", "").replace("_", "").isalnum():
        parser.error("run must be a plain identifier")
    run = ROOT / "runtime" / args.run
    if args.action == "data":
        from .data import build
        build()
    elif args.action in ("start", "resume"):
        run.mkdir(parents=True, exist_ok=True)
        if args.action == "start" and (run / "state.json").exists():
            parser.error("run already exists; inspect its evidence before starting a new named run")
        if args.action == "resume":
            if not (run / "state.json").exists():
                parser.error("unknown run")
            previous = json.loads((run / "state.json").read_text())
            if previous["status"] not in ("failed", "stopped"):
                parser.error("only failed or stopped runs may resume")
            (run / "STOP").unlink(missing_ok=True)
        with (run / "controller.log").open("a") as log:
            process = subprocess.Popen([sys.executable, "-m", "ruflo_model.loop", str(run)],
                                       cwd=ROOT, start_new_session=True, stdin=subprocess.DEVNULL,
                                       stdout=log, stderr=subprocess.STDOUT)
        write_json(run / "launch.json", {"pid": process.pid, "launched_at": now()})
        print(json.dumps({"pid": process.pid, "run": str(run)}))
    elif args.action == "stop":
        if not run.exists():
            parser.error("unknown run")
        (run / "STOP").write_text(now() + "\n")
        release = ROOT / "runtime/release-validation"
        if release.exists():
            (release / "STOP").write_text(now() + "\n")
        print("Stop requested. Worker will terminate its current process group within 12 seconds.")
    else:
        path = run / "state.json"
        if not path.exists():
            parser.error("no worker state yet")
        state = json.loads(path.read_text())
        try:
            os.kill(state["pid"], 0)
            state["process_exists"] = True
        except ProcessLookupError:
            state["process_exists"] = False
        state["heartbeat_age_seconds"] = round((datetime.now(timezone.utc) - datetime.fromisoformat(state["heartbeat_at"])).total_seconds(), 1)
        state["worker_live"] = state["process_exists"] and state["status"] == "running" and state["heartbeat_age_seconds"] < 20
        print(json.dumps(state, indent=2))


if __name__ == "__main__":
    main()
