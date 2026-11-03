"""HF Jobs dispatch with conservative reservations and provider enforced timeouts.

Reservations are never automatically refunded. Ambiguous submission errors keep
their full reservation and must be reconciled before a retry. No persistent GPU.
"""
import argparse
import base64
import fcntl
import gzip
import json
import math
import os
import re
import uuid
from pathlib import Path

import httpx

from .common import ROOT, digest, now, read_jsonl, write_json

CAP_MICRO_USD = 200_000_000
IMAGE = "pytorch/pytorch@sha256:77f17f843507062875ce8be2a6f76aa6aa3df7f9ef1e31d9d7432f4b0f563dee"


def namespace_value(value):
    if not isinstance(value, str) or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,95}", value) is None:
        raise ValueError("A valid explicit Hugging Face namespace is required")
    return value


def authenticated_namespace(api):
    """New jobs use the account identified by the configured HF credential."""
    identity = api.whoami()
    if not isinstance(identity, dict):
        raise ValueError("Hugging Face did not return an authenticated identity")
    return namespace_value(identity.get("name"))


def receipt_namespace(receipt):
    """Existing jobs remain bound to their recorded namespace, without a fallback."""
    if not isinstance(receipt, dict) or "namespace" not in receipt:
        raise ValueError("Existing job receipt is missing namespace; reconcile its identity before use")
    return namespace_value(receipt["namespace"])


class Budget:
    def __init__(self, path=ROOT / "runtime/cloud-budget.json"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def update(self, action):
        with self.path.with_suffix(".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            ledger = json.loads(self.path.read_text()) if self.path.exists() else {
                "cap_micro_usd": CAP_MICRO_USD, "reservations": [], "note": "reserved maximum cost, not an invoice"}
            result = action(ledger)
            write_json(self.path, ledger)
            return result

    def reserve(self, amount, meta):
        if not isinstance(amount, int) or amount <= 0:
            raise ValueError("positive integer micro USD required")
        def action(ledger):
            if ledger["cap_micro_usd"] > CAP_MICRO_USD:
                raise ValueError("budget file exceeds authorized USD 200")
            if any(r["status"] == "submission_failed_or_unknown" for r in ledger["reservations"]):
                raise ValueError("Reconcile ambiguous cloud submission before another dispatch")
            if sum(r["reserved_micro_usd"] for r in ledger["reservations"]) + amount > ledger["cap_micro_usd"]:
                raise ValueError("cloud budget exhausted")
            item = {**meta, "id": uuid.uuid4().hex, "created_at": now(), "reserved_micro_usd": amount,
                    "status": "reserved"}
            ledger["reservations"].append(item)
            return item
        return self.update(action)

    def annotate(self, reservation_id, **fields):
        if set(fields) - {"status", "job_id", "error", "provider_status", "estimated_billable_upper_usd"}:
            raise ValueError("Budget reservations are immutable")
        def action(ledger):
            entry = next(r for r in ledger["reservations"] if r["id"] == reservation_id)
            entry.update(fields)
        self.update(action)


def submit_benchmark():
    from huggingface_hub import HfApi
    root = ROOT / "runtime/cloud"
    root.mkdir(parents=True, exist_ok=True)
    if (root / "job.json").exists():
        raise ValueError("job receipt already exists; no implicit redispatch")
    api = HfApi()
    namespace = authenticated_namespace(api)
    hardware = httpx.get("https://huggingface.co/api/jobs/hardware", timeout=20)
    hardware.raise_for_status()
    price = next(h for h in hardware.json() if h["name"] == "a100-large")
    if price["unitLabel"] != "minute" or price["unitCostMicroUSD"] > 50_000:
        raise ValueError("unexpected cloud price; review before dispatch")
    timeout = 3600
    reserve = (math.ceil(timeout / 60) + 1) * price["unitCostMicroUSD"]
    budget = Budget()
    payload = [{k: r[k] for k in ("id", "messages", "tools")} for r in read_jsonl(ROOT / "data/dev.jsonl")]
    script = ROOT / "scripts/cloud_benchmark.py"
    entry = budget.reserve(reserve, {"purpose": "coding baseline", "flavor": "a100-large",
        "timeout_seconds": timeout, "price": price, "script_sha256": digest(script),
        "suite_sha256": digest(ROOT / "data/dev.jsonl"), "image": IMAGE, "namespace": namespace})
    # The job has no HF token or other secret; the pinned base model is public.
    env = {"RUFLO_CASES_GZIP": base64.b64encode(gzip.compress(json.dumps(payload).encode())).decode(),
           "RUFLO_SCRIPT_GZIP": base64.b64encode(gzip.compress(script.read_bytes())).decode(),
           "HF_HUB_DISABLE_PROGRESS_BARS": "1", "TOKENIZERS_PARALLELISM": "false"}
    bootstrap = (
        "import os,base64,gzip,subprocess,sys; "
        "subprocess.check_call([sys.executable,'-m','pip','install','--quiet','transformers==5.12.1','accelerate==1.12.0','safetensors==0.7.0']); "
        "exec(compile(gzip.decompress(base64.b64decode(os.environ['RUFLO_SCRIPT_GZIP'])),'cloud_benchmark.py','exec'))"
    )
    try:
        job = api.run_job(image=IMAGE, command=["python", "-u", "-c", bootstrap], env=env,
            secrets={}, flavor="a100-large", timeout=timeout, namespace=namespace,
            name="ruflo-code-baseline-v1", labels={"project": "ruflo-code-model", "reservation": entry["id"]})
        receipt = {"job_id": job.id, "url": job.url, "namespace": namespace, "reservation": entry["id"],
                   "reserved_max_usd": reserve / 1e6, "timeout_seconds": timeout, "submitted_at": now()}
        write_json(root / "job.json", receipt)
        budget.annotate(entry["id"], status="submitted", job_id=job.id)
        return receipt
    except Exception as e:
        budget.annotate(entry["id"], status="submission_failed_or_unknown", error=type(e).__name__)
        raise


def inspect():
    from huggingface_hub import HfApi
    root = ROOT / "runtime/cloud"
    receipt = json.loads((root / "job.json").read_text())
    namespace = receipt_namespace(receipt)
    api = HfApi()
    job = api.inspect_job(job_id=receipt["job_id"], namespace=namespace)
    status = {"job_id": job.id, "stage": str(job.status.stage), "checked_at": now(),
              "reserved_max_usd": receipt["reserved_max_usd"], "namespace": namespace}
    write_json(root / "status.json", status)
    logs = list(api.fetch_job_logs(job_id=job.id, namespace=namespace, follow=False))
    (root / "job.log").write_text("\n".join(str(line) for line in logs))
    return status


def cancel():
    from huggingface_hub import HfApi
    receipt = json.loads((ROOT / "runtime/cloud/job.json").read_text())
    namespace = receipt_namespace(receipt)
    HfApi().cancel_job(job_id=receipt["job_id"], namespace=namespace)
    return inspect()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=["submit-benchmark", "status", "cancel"])
    a = p.parse_args()
    if a.action == "submit-benchmark":
        print(json.dumps(submit_benchmark()))
    elif a.action == "status":
        print(json.dumps(inspect()))
    else:
        print(json.dumps(cancel()))
