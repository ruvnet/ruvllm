"""Shared deterministic serialization and artifact checks; no provider credentials."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def clean_environment():
    """Pass only runtime paths and GPU controls; provider and HF credentials stay out."""
    allowed={"PATH","HOME","USER","LANG","LC_ALL","LD_LIBRARY_PATH","CUDA_HOME","CUDA_PATH",
             "CUDA_VISIBLE_DEVICES","NVIDIA_VISIBLE_DEVICES","NVIDIA_DRIVER_CAPABILITIES",
             "NVIDIA_REQUIRE_CUDA","TMPDIR","XDG_CACHE_HOME","HF_HOME","PYTHONPATH"}
    result={k:v for k,v in os.environ.items() if k in allowed}
    result.update(HF_HUB_DISABLE_IMPLICIT_TOKEN="1",HF_HUB_DISABLE_TELEMETRY="1",HF_TOKEN_PATH="/dev/null",
                  VLLM_NO_USAGE_STATS="1",DO_NOT_TRACK="1",TOKENIZERS_PARALLELISM="false")
    return result


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def object_sha(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value, exclusive=True):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x" if exclusive else "w") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def tree(path):
    """Hash all inference relevant bytes, excluding caches and our own receipts."""
    path = Path(path)
    files = {str(p.relative_to(path)): sha(p) for p in sorted(path.rglob("*"))
             if p.is_file() and ".cache" not in p.parts
             and p.suffix in (".safetensors", ".json", ".jinja", ".txt", ".model")}
    if not files or not any(n.endswith(".safetensors") for n in files):
        raise ValueError("Model has no Safetensors weights")
    return {"files": files, "sha256": object_sha(files)}


def verify_protocol(folder=HERE):
    folder = Path(folder)
    protocol = read(folder / "protocol.json")
    frozen = read(folder / "freeze.json")
    if sha(folder / "protocol.json") != frozen["protocol_sha256"]:
        raise ValueError("Frozen protocol changed")
    for name, expected in protocol["input_sha256"].items():
        if sha(folder / name) != expected:
            raise ValueError(f"Frozen input changed: {name}")
    for name, expected in protocol["implementation_sha256"].items():
        if sha(folder / name) != expected:
            raise ValueError(f"Frozen implementation changed: {name}")
    return protocol, frozen["protocol_sha256"]
