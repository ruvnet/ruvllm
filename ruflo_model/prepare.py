"""Download the pinned public base and reproduce its local quantization."""
import json
from pathlib import Path
import subprocess
import sys

from huggingface_hub import snapshot_download

from .common import ROOT, digest, write_json


def main():
    pin = json.loads((ROOT / "model_pin.json").read_text())
    base = ROOT / pin["path"]
    snapshot_download(repo_id=pin["repo_id"], revision=pin["revision"], token=False, local_dir=base,
        allow_patterns=["*.json", "*.jinja", "*.safetensors", "LICENSE*", "README.md", "vocab*", "merges*"])
    out = ROOT / "models/base-4bit"
    expected = ROOT / "quantization.json"
    if out.exists():
        manifest = json.loads(expected.read_text())
        for name, sha in manifest["files"].items():
            if digest(out / name) != sha:
                raise ValueError(f"Existing model does not match the pinned quantization: {name}")
        print("Pinned quantized model verified")
        return
    temporary = ROOT / "models/base-4bit.pending"
    if temporary.exists():
        raise ValueError("Partial conversion exists; inspect it before retry")
    subprocess.run([sys.executable, "-m", "mlx_lm", "convert", "--hf-path", str(base), "--mlx-path", str(temporary),
                    "-q", "--q-bits", "4", "--q-group-size", "64"], check=True)
    hashes = {p.name: digest(p) for p in sorted(temporary.iterdir()) if p.is_file()}
    if expected.exists():
        expected_hashes = json.loads(expected.read_text())["files"]
        if hashes != expected_hashes:
            write_json(ROOT / "runtime/quantization-mismatch.json", {"expected": expected_hashes, "actual": hashes})
            raise ValueError("Quantization differs from the reference; inspect runtime/quantization-mismatch.json")
    temporary.rename(out)
    print("Pinned model prepared")


if __name__ == "__main__":
    main()
