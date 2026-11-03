"""Sequential immutable arms with a shared absolute deadline and process cleanup."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from common import HERE, clean_environment, read, sha, verify_protocol, write


def main(work, deadline):
    protocol,protocol_sha=verify_protocol()
    deadline=min(deadline,time.time()+protocol["maximum_wall_seconds"])
    child=None
    def stop(*_):
        raise InterruptedError("Supervisor stopped")
    signal.signal(signal.SIGTERM,stop)
    signal.signal(signal.SIGINT,stop)
    log_dir=work/"logs"
    log_dir.mkdir(parents=True,exist_ok=True)
    state={"status":"running","protocol_sha256":protocol_sha,"deadline_epoch":deadline,"pid":os.getpid()}
    try:
        for arm in protocol["arms"]:
            out=work/"outputs"/arm["id"]
            if out.exists():
                if not (out/"complete.json").exists(): raise ValueError("Partial arm exists; no silent retry")
                complete=read(out/"complete.json")
                if complete["protocol_sha256"]!=protocol_sha or sha(out/"responses.jsonl")!=complete["responses_sha256"]:
                    raise ValueError("Completed evidence changed")
                continue
            state["arm"]=arm["id"]
            write(work/"run-state.json",state,exclusive=False)
            with (log_dir/(arm["id"]+".log")).open("x") as log:
                child=subprocess.Popen([sys.executable,str(HERE/"worker.py"),"--work",str(work),"--arm",arm["id"],
                                        "--deadline",str(deadline)],stdout=log,stderr=subprocess.STDOUT,
                                       env=clean_environment(),start_new_session=True)
                while child.poll() is None:
                    if time.time()>=deadline or (work/"STOP").exists(): raise TimeoutError("Absolute experiment deadline or STOP")
                    time.sleep(2)
                if child.returncode: raise RuntimeError(f"Arm failed: {arm['id']}; inspect preserved log")
                child=None
        completes=[read(work/"outputs"/a["id"]/"complete.json") for a in protocol["arms"]]
        total=sum(r["output_tokens"] for r in completes)
        count=sum(r["rows"] for r in completes)
        assert count==protocol["total_generations"] and total<=protocol["maximum_output_tokens"]
        state.update(status="completed",generations=count,output_tokens=total)
    except BaseException as error:
        state.update(status="failed",error=f"{type(error).__name__}: {error}")
        raise
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try: child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid,signal.SIGKILL)
                child.wait()
        write(work/"run-state.json",state,exclusive=False)
        print(json.dumps(state),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--work",type=Path,required=True)
    p.add_argument("--deadline",type=float,required=True)
    args=p.parse_args()
    main(args.work,args.deadline)
