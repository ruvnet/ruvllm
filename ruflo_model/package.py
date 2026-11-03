"""Prepare a reviewable local model release. Never publishes to an external service."""
import argparse
import json
from pathlib import Path
import shutil
import tarfile

from .common import ROOT, digest, now, write_json
from .report import build


def sanitized_copy(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.suffix in (".json", ".jsonl", ".md", ".log"):
        destination.write_text(source.read_text().replace(str(ROOT), ".").replace(str(Path.home()), "~"))
    else:
        shutil.copyfile(source, destination)


def package(output, experimental=False):
    output = Path(output).resolve()
    archive = Path(str(output) + ".tar.gz")
    if output.exists() or archive.exists():
        raise ValueError("Release directory or archive already exists; never overwrite a reviewed artifact")
    report = build()
    run = ROOT / "runtime/pilot-v2"
    selection = json.loads((run / "selection.json").read_text())
    if not selection["selected_adapter"]:
        raise ValueError("No adapter qualified for selection")
    selected = Path(selection["selected_adapter"])
    weight_sha = digest(selected / "adapters.safetensors")
    external = json.loads((ROOT / "runtime/release-validation/report.json").read_text())
    read_receipt = json.loads((ROOT / "runtime/integration/selected/receipt.json").read_text())
    code_receipt = json.loads((ROOT / "runtime/integration/selected-coding/receipt.json").read_text())
    loaded = json.loads((ROOT / "runtime/server/backend-state.json").read_text())
    integration = all(r["passed"] and r["adapter_sha256"] == weight_sha
                      and r.get("backend_binding_verified") and r.get("matched_adapter_tensors") == 32
                      for r in (read_receipt, code_receipt))
    integration = integration and loaded["binding_verified"] and loaded["matched_adapter_tensors"] == 32 and loaded["adapter_sha256"] == weight_sha and loaded["adapter_config_sha256"] == digest(selected / "adapter_config.json")
    qualified = external["candidate_qualified"] and integration
    if not qualified and not experimental:
        raise ValueError("Release checks did not qualify this candidate")
    output.mkdir(parents=True)
    peft = ROOT / "runtime/cloud-train/download" / selected.name
    for name in ("adapter_model.safetensors", "adapter_config.json"):
        shutil.copyfile(peft / name, output / name)
    config = json.loads((output / "adapter_config.json").read_text())
    config["revision"] = json.loads((ROOT / "model_pin.json").read_text())["revision"]
    write_json(output / "adapter_config.json", config)
    (output / "mlx").mkdir()
    for name in ("adapters.safetensors", "adapter_config.json"):
        shutil.copyfile(selected / name, output / "mlx" / name)
    for name in ("LICENSE", "NOTICE", "model_pin.json", "quantization.json", "verify_release.py"):
        shutil.copyfile(ROOT / name, output / name)
    for name in ("ruflo_model", "scripts", "configs", "data", "docs", "tests"):
        shutil.copytree(ROOT / name, output / "reproduction" / name,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for name in ("pyproject.toml", "requirements.lock", "README.md", "model_pin.json", "quantization.json", "LICENSE", "NOTICE", "verify.py", "verify_release.py"):
        if (ROOT / name).exists():
            shutil.copyfile(ROOT / name, output / "reproduction" / name)
    for name in ("REPORT.md", "report.json"):
        sanitized_copy(ROOT / "results" / name, output / "reproduction/results" / name)
    evidence = {
        "baseline-bf16-dev": ROOT / "runtime/pilot-v1/baseline-dev",
        "baseline-4bit-dev": run / "baseline-dev",
        "selected-dev": run / (selected.name + "-dev"),
        "baseline-release": run / "baseline-release",
        "selected-release": run / "selected-release",
        "baseline-humaneval": ROOT / "runtime/release-validation/baseline-humaneval",
        "selected-humaneval": ROOT / "runtime/release-validation/selected-humaneval",
        "coder-30b-dev": ROOT / "runtime/cloud",
    }
    for label, directory in evidence.items():
        for name in ("protocol.json", "summary.json", "responses.jsonl"):
            if (directory / name).exists():
                sanitized_copy(directory / name, output / "evidence" / label / name)
    for name, path in {"selection.json": run / "selection.json", "worker.json": run / "state.json",
                       "external.json": ROOT / "runtime/release-validation/report.json",
                       "read-integration.json": ROOT / "runtime/integration/selected/receipt.json",
                       "coding-integration.json": ROOT / "runtime/integration/selected-coding/receipt.json",
                       "training.json": ROOT / "runtime/cloud-train/download/manifest.json",
                       "conversion.json": run / "training/conversion.json"}.items():
        sanitized_copy(path, output / "evidence" / name)
    for path in (ROOT / "results/cloud").iterdir():
        sanitized_copy(path, output / "evidence/cloud" / path.name)
    for directory in sorted((ROOT / "runtime/integration").iterdir()):
        if not directory.is_dir():
            continue
        label = directory.name
        for name in ("events.jsonl", "receipt.json", "stderr.log", "invalidated.json"):
            if (directory / name).exists():
                sanitized_copy(directory / name, output / "evidence/integration" / label / name)
        for path in (directory / "fixture").glob("*"):
            if path.is_file():
                sanitized_copy(path, output / "evidence/integration" / label / "fixture" / path.name)
    for label, path in [("bf16-training.log", ROOT / "runtime/pilot-v1/train.1.log"),
                        ("quantized-training.log", run / "train.1.log"),
                        ("bf16-worker.json", ROOT / "runtime/pilot-v1/state.json")]:
        sanitized_copy(path, output / "evidence/failed-attempts" / label)
    sanitized_copy(ROOT / "results/report.json", output / "evidence/report.json")
    sanitized_copy(ROOT / "results/REPORT.md", output / "evidence/REPORT.md")
    for name in ("adapter-audit.json", "base-integrity.json", "http-adapter-parity.json"):
        sanitized_copy(ROOT / "results" / name, output / "evidence" / name)
    sanitized_copy(ROOT / "runtime/server/backend-state.json", output / "evidence/backend-loaded.json")
    if (ROOT / "results/invalidated-releases.json").exists():
        sanitized_copy(ROOT / "results/invalidated-releases.json", output / "evidence/invalidated-releases.json")
    sanitized_copy(ROOT / "runtime/humaneval-reference-check.json", output / "evidence/reference-preflight.json")
    if (ROOT / "runtime/humaneval-reference-recheck.json").exists():
        sanitized_copy(ROOT / "runtime/humaneval-reference-recheck.json", output / "evidence/reference-recheck.json")
    dev = selection["selected"]
    release = json.loads((run / "selected-release/summary.json").read_text())
    base_ext, ext = external["baseline"], external["selected"]
    status = "Experimental preview with passing local release checks" if qualified else "Research artifact that did not pass release checks"
    card = f'''---
license: apache-2.0
base_model: Qwen/Qwen3.5-4B
library_name: peft
tags:
  - coding
  - tool-use
  - lora
  - mlx
---

# RuFlo Code 4B

{status}. SOTA is not established.

This adapter specializes [Qwen3.5 4B](https://huggingface.co/Qwen/Qwen3.5-4B) for narrow coding and tool requests routed from Claude Code. This experiment used no Claude weights, Claude generated training traces, private repositories, or user conversations. [Anthropic does not support non Claude models](https://code.claude.com/docs/en/llm-gateway); compatibility was tested locally using Claude Code 2.1.316 and [Claude Code Router](https://github.com/musistudio/claude-code-router) 3.1.3.

The selected checkpoint is {selected.name}. It updates 1,507,328 parameters across the last 16 MLP output projections using rank 8 LoRA. The base stays frozen. The training pool contains 288 original synthetic traces. This checkpoint consumed 128 examples; the full 96 step run consumed 192. The validation pool contains 36 traces, with 8 used for loss checks. Training used BF16 on one A100 80GB. Deployment evaluation used a separately pinned MLX 4 bit base. The PEFT and MLX formats describe the same low rank update, with a checked transpose and scale conversion.

| Evaluation | Base | Candidate |
| :--- | :--- | :--- |
| Synthetic development | {selection['baseline']['passed']}/35 | {dev['passed']}/35 |
| Synthetic release | {json.loads((run / 'baseline-release/summary.json').read_text())['passed']}/35 | {release['passed']}/35 |
| HumanEval local protocol | {base_ext['passed']}/164 | {ext['passed']}/164 |

Development selected the checkpoint. The other sets were excluded from selection. HumanEval used greedy chat prompting, a 768 token cap, Python 3.12, and original tests in an OS sandbox. These scores are not an official leaderboard submission. The candidate gained {len(report['external_paired']['improved_ids'])} cases and lost {len(report['external_paired']['regressed_ids'])}; paired McNemar p = {report['external_paired']['exact_mcnemar_p']:.3f}. This pilot does not establish a statistically conclusive coding improvement. Base pretraining contamination is unknown. Raw outputs, protocols, confidence intervals, paired changes, and cloud receipts are under `evidence/`.

Actual Claude Code integration verified a fresh file read plus a Read, Edit, Bash repair cycle with independent assertions. The server binds the configured base and adapter together and verifies all 32 loaded adapter tensors against the artifact. Earlier HTTP integration runs were invalidated because MLX LM 0.31.3 dropped its configured adapter during model lookup. A separate fixture permission error was also corrected. These attempts remain in the evidence. The offline benchmarks loaded the adapter directly and were unaffected. This does not establish long session or repository benchmark performance. Treat this as a specialist preview and keep a stronger model in the router for general work.

## Local use

The `reproduction` directory contains the source and pinned environment. On Apple Silicon:

```sh
cd reproduction
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.lock
.venv/bin/python -m ruflo_model.prepare
.venv/bin/python scripts/serve.py --model models/base-4bit --adapter ../mlx
```

Configure an OpenAI Chat provider with URL `http://127.0.0.1:9100/v1`, model `ruflo-code-4b`, and the key in `runtime/server/api-key`. Set `CLAUDE_CODE_MAX_OUTPUT_TOKENS=4096`, `MAX_THINKING_TOKENS=0`, and `CLAUDE_CODE_DISABLE_ADAPTIVE_THINKING=1` in the client environment. The profile is text only, allows 8192 output tokens within a 16384 total token cap, and runs one request at a time. Full context quality is not validated. The internal backend is for a trusted single user machine.

For Transformers, load the pinned base with `AutoModelForImageTextToText`, then attach this directory using `PeftModel.from_pretrained`. GPU serving with vLLM or other runtimes has not been validated in this experiment.

## Limitations

Synthetic tool templates are correlated and small gains are uncertain. HumanEval covers short Python functions. Other languages, large repositories, long context, complex tool recovery, adversarial robustness, and concurrent serving need separate evaluation. Do not use these pilot results as a broad SOTA claim.

Base and adapter license: Apache 2.0. HumanEval data retains its MIT license. See LICENSE, NOTICE, data provenance, and checksums in SHA256SUMS.json. No automatic publication occurs. Verify the extracted package with `python3 verify_release.py` before adding model downloads or other files to it.
'''
    (output / "README.md").write_text(card)
    write_json(output / "release.json", {"created_at": now(), "version": "0.1.0-preview", "qualified_local_preview": qualified,
        "sota_verified": False, "selected_checkpoint": selected.name, "mlx_adapter_sha256": weight_sha,
        "peft_adapter_sha256": digest(output / "adapter_model.safetensors"), "integration_passed": integration,
        "public_upload_performed": False})
    hashes = {str(p.relative_to(output)): digest(p) for p in sorted(output.rglob("*")) if p.is_file()}
    write_json(output / "SHA256SUMS.json", hashes)
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(output, arcname=output.name)
    write_json(ROOT / "results/release-artifact.json", {"directory": str(output.relative_to(ROOT)),
        "archive": str(archive.relative_to(ROOT)), "archive_sha256": digest(archive),
        "archive_bytes": archive.stat().st_size, "qualified_local_preview": qualified})
    return archive


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(ROOT / "dist/ruflo-code-4b-preview"))
    parser.add_argument("--experimental", action="store_true")
    args = parser.parse_args()
    print(package(args.output, args.experimental))
