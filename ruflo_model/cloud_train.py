import base64
import gzip
import json
import math
from pathlib import Path
import re
import signal
import sys
import time

import httpx
from huggingface_hub import HfApi, Volume

from .cloud import Budget, IMAGE, authenticated_namespace, receipt_namespace
from .common import ROOT, digest, now, read_jsonl, write_json

def convert(source, destination):
    import mlx.core as mx
    source, destination = Path(source), Path(destination)
    manifest = json.loads((source / "manifest.json").read_text())
    for file, sha in manifest["files"].items():
        if digest(source / file) != sha:
            raise ValueError("Cloud checkpoint hash mismatch")
    if manifest["revision"] != json.loads((ROOT / "model_pin.json").read_text())["revision"]:
        raise ValueError("Wrong base revision")
    if manifest["training_sha256"] != digest(ROOT / "data/training/train.jsonl"):
        raise ValueError("Checkpoint belongs to different training data")
    destination.mkdir(parents=True, exist_ok=True)
    receipts = []
    for step in (32, 64, 96):
        weights = mx.load(str(source / f"step-{step:03d}/adapter_model.safetensors"))
        converted = {}
        for name, value in weights.items():
            match = re.search(r"layers\.(\d+)\.mlp\.down_proj\.lora_([AB])\.weight$", name)
            if match is None or not 16 <= int(match[1]) < 32:
                raise ValueError("Unexpected adapter tensor name")
            if value.ndim != 2 or not bool(mx.all(mx.isfinite(value))):
                raise ValueError("Invalid adapter tensor")
            key = f"language_model.model.layers.{match[1]}.mlp.down_proj.lora_{match[2].lower()}"
            converted[key] = value.T
        if len(converted) != 32:
            raise ValueError("Incomplete adapter")
        if not any(bool(mx.any(v != 0)) for k, v in converted.items() if k.endswith("lora_b")):
            raise ValueError("Untrained adapter")
        path = destination / f"{step:07d}_adapters.safetensors"
        mx.save_safetensors(str(path), converted)
        receipts.append({"step": step, "sha256": digest(path), "tensors": len(converted),
            "parameters": sum(v.size for v in converted.values())})
    # PEFT scales alpha/r; MLX stores the actual multiplicative scale.
    write_json(destination / "adapter_config.json", {"fine_tune_type": "lora", "num_layers": 16,
        "lora_parameters": {"rank": 8, "scale": 2.0, "dropout": 0.0, "keys": ["mlp.down_proj"]}})
    write_json(destination / "conversion.json", {"source_manifest_sha256": digest(source / "manifest.json"),
        "mapping": "PEFT A and B transposed; MLX scale = PEFT alpha / rank = 2", "checkpoints": receipts})
    return receipts


def run(destination):
    root = ROOT / "runtime/cloud-train"
    root.mkdir(parents=True, exist_ok=True)
    api, budget = HfApi(), Budget()
    receipt_path = root / "job.json"
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        namespace = receipt_namespace(receipt)
    else:
        namespace = authenticated_namespace(api)
        response = httpx.get("https://huggingface.co/api/jobs/hardware", timeout=20)
        response.raise_for_status()
        price = next(p for p in response.json() if p["name"] == "a100-large")
        if price["unitLabel"] != "minute" or price["unitCostMicroUSD"] > 50_000:
            raise ValueError("Unexpected hardware price")
        script = ROOT / "scripts/cloud_train.py"
        payload = {"train": read_jsonl(ROOT / "data/training/train.jsonl"),
                   "valid": read_jsonl(ROOT / "data/training/valid.jsonl"),
                   "training_sha256": digest(ROOT / "data/training/train.jsonl")}
        timeout = 2700
        reservation = budget.reserve((math.ceil(timeout / 60) + 1) * price["unitCostMicroUSD"],
            {"purpose": "BF16 LoRA training", "flavor": "a100-large", "timeout_seconds": timeout,
             "script_sha256": digest(script), "image": IMAGE, "price": price, "namespace": namespace})
        bucket = f"{namespace}/ruflo-code-model-{reservation['id'][:12]}"
        api.create_bucket(bucket, private=True, exist_ok=False)
        if not api.bucket_info(bucket).private:
            raise ValueError("Checkpoint bucket must be private")
        env = {"RUFLO_TRAIN_GZIP": base64.b64encode(gzip.compress(json.dumps(payload).encode())).decode(),
               "RUFLO_SCRIPT_GZIP": base64.b64encode(gzip.compress(script.read_bytes())).decode(),
               "HF_HUB_DISABLE_PROGRESS_BARS": "1", "TOKENIZERS_PARALLELISM": "false"}
        bootstrap = (
            "import os,base64,gzip,subprocess,sys; "
            "subprocess.check_call([sys.executable,'-m','pip','install','--quiet','transformers==5.12.1','accelerate==1.12.0','peft==0.21.2','safetensors==0.7.0']); "
            "exec(compile(gzip.decompress(base64.b64decode(os.environ['RUFLO_SCRIPT_GZIP'])),'cloud_train.py','exec'))"
        )
        try:
            job = api.run_job(image=IMAGE, command=["python", "-u", "-c", bootstrap], env=env, secrets={},
                flavor="a100-large", timeout=timeout, namespace=namespace, name="ruflo-code-train-v1",
                labels={"project": "ruflo-code-model", "reservation": reservation["id"]},
                volumes=[Volume(type="bucket", source=bucket, mount_path="/checkpoints", read_only=False)])
            receipt = {"job_id": job.id, "url": job.url, "bucket": bucket, "namespace": namespace, "reservation": reservation["id"],
                       "submitted_at": now(), "timeout_seconds": timeout,
                       "reserved_max_usd": reservation["reserved_micro_usd"] / 1e6}
            write_json(receipt_path, receipt)
            budget.annotate(reservation["id"], status="submitted", job_id=job.id)
        except Exception as e:
            budget.annotate(reservation["id"], status="submission_failed_or_unknown", error=type(e).__name__)
            raise
    namespace = receipt_namespace(receipt)
    bucket = receipt["bucket"]
    if not isinstance(bucket, str) or re.fullmatch(re.escape(namespace) + r"/[A-Za-z0-9][A-Za-z0-9_-]{0,127}", bucket) is None:
        raise ValueError("Checkpoint bucket must belong to the recorded job namespace")
    canceled = False
    def cancel(*_):
        nonlocal canceled
        canceled = True
        api.cancel_job(job_id=receipt["job_id"], namespace=namespace)
        raise InterruptedError("cloud training canceled")
    signal.signal(signal.SIGTERM, cancel)
    signal.signal(signal.SIGINT, cancel)
    started = time.monotonic()
    while True:
        job = api.inspect_job(job_id=receipt["job_id"], namespace=namespace)
        status = {"stage": str(job.status.stage), "checked_at": now(), "job_id": job.id}
        write_json(root / "status.json", status)
        print(json.dumps(status), flush=True)
        if status["stage"] in ("COMPLETED", "ERROR", "CANCELED", "DELETED"):
            break
        if time.monotonic() - started > 3000:
            cancel()
        time.sleep(10)
    logs = list(api.fetch_job_logs(job_id=job.id, namespace=namespace, follow=False))
    (root / "job.log").write_text("\n".join(str(x) for x in logs))
    if status["stage"] != "COMPLETED":
        raise RuntimeError("Cloud training failed; see runtime/cloud-train/job.log")
    source = root / "download"
    source.mkdir(exist_ok=True)
    api.sync_bucket(source=f"hf://buckets/{bucket}/training-v1", dest=str(source), quiet=True)
    receipts = convert(source, destination)
    write_json(root / "completed.json", {"job_id": job.id, "downloaded_at": now(), "converted": receipts})
    return receipts


if __name__ == "__main__":
    print(json.dumps(run(sys.argv[1])))
