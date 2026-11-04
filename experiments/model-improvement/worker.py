#!/usr/bin/env python3
"""Offline CUDA inference worker. CPU tests establish contracts, never model quality."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import re
import signal
import time

from measurement import file_sha, load_bundle, object_sha

EOS_IDS = [248044, 248046]
MODEL_REPO = "Qwen/Qwen3.5-4B"
MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
RUNTIME_PINS = {"torch": "2.11.0", "transformers": "5.12.1", "peft": "0.21.2", "safetensors": "0.7.0"}


def local_directory(value):
    path = Path(value)
    if not path.is_absolute() or not path.is_dir():
        raise ValueError("Input directories must be absolute existing local paths")
    return path


def checked_file(directory, name, digest):
    if not isinstance(name, str) or Path(name).name != name or name in {".", ".."}:
        raise ValueError("Asset filename must be a single local filename")
    path = directory / name
    if not path.is_file() or file_sha(path) != digest:
        raise ValueError(f"Asset SHA256 mismatch: {name}")
    return path


def verify_assets(protocol, model_dir, adapter_dir):
    identity = protocol["model"]
    if identity["repo_id"] != MODEL_REPO or identity["revision"] != MODEL_REVISION:
        raise ValueError("Unsupported model identity/revision")
    files = identity["weight_files"]
    if not isinstance(files, dict) or not files:
        raise ValueError("Pinned model weights are required")
    for name, digest in files.items():
        if not re.fullmatch(r"model(?:-\d{5}-of-\d{5}|\.safetensors-\d{5}-of-\d{5})?\.safetensors", name):
            raise ValueError("Only safe model weight filenames are supported")
        checked_file(model_dir, name, digest)
    actual = {p.name for p in model_dir.glob("model*.safetensors")}
    if actual != set(files):
        raise ValueError("Model shard set differs from the pinned protocol")
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text())
        if set(index["weight_map"].values()) != set(files):
            raise ValueError("Model index does not reference exactly the pinned shards")
    elif len(files) != 1 or "model.safetensors" not in files:
        raise ValueError("Sharded weights require a local weight index")
    assets = identity["asset_files"]
    if not {"config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja"} <= set(assets):
        raise ValueError("Pinned config, tokenizer and template assets are required")
    for name, digest in assets.items():
        checked_file(model_dir, name, digest)
    allowed = set(files) | set(assets) | {"README.md", "LICENSE", ".gitattributes"}
    if any(path.is_file() and path.name not in allowed for path in model_dir.iterdir()):
        raise ValueError("Unpinned model files could alter loading or generation")
    config = json.loads((model_dir / "config.json").read_text())
    if config.get("model_type") != "qwen3_5" or config.get("architectures") != ["Qwen3_5ForConditionalGeneration"]:
        raise ValueError("Unexpected model architecture")
    if config.get("_commit_hash", MODEL_REVISION) != MODEL_REVISION:
        raise ValueError("Model config revision differs from the protocol")
    checked_file(adapter_dir, "adapter_model.safetensors", protocol["adapter"]["weights_sha256"])
    adapter_path = checked_file(adapter_dir, "adapter_config.json", protocol["adapter"]["config_sha256"])
    adapter = json.loads(adapter_path.read_text())
    required = {"base_model_name_or_path": MODEL_REPO, "revision": MODEL_REVISION,
                "peft_type": "LORA", "r": 8, "lora_alpha": 16, "lora_dropout": 0.0,
                "bias": "none", "target_modules": ["down_proj"], "layers_to_transform": list(range(16, 32)),
                "use_dora": False, "use_rslora": False}
    if any(adapter.get(key) != value for key, value in required.items()):
        raise ValueError("Unexpected adapter configuration")
    if adapter.get("rank_pattern") or adapter.get("alpha_pattern") or adapter.get("modules_to_save"):
        raise ValueError("Adapter overrides are not supported")


def verify_runtime(protocol, version=importlib.metadata.version):
    if protocol["runtime_versions"] != RUNTIME_PINS:
        raise ValueError("Protocol runtime pins differ from supported versions")
    installed = {name: version(name) for name in RUNTIME_PINS}
    # CUDA wheels append a build tag (e.g. +cu130) to the package version.
    if any(installed[name].split("+", 1)[0] != pin for name, pin in RUNTIME_PINS.items()):
        raise ValueError(f"Runtime version mismatch: {installed}")
    return installed


def render_prompt(tokenizer, case):
    ids = tokenizer.apply_chat_template(case["messages"], tools=case["tools"], tokenize=True,
                                        add_generation_prompt=True, enable_thinking=False,
                                        return_dict=False)
    if not isinstance(ids, list) or not ids or any(type(i) is not int or i < 0 for i in ids):
        raise ValueError("Tokenizer did not return a nonempty untruncated token list")
    return ids


def verify_adapter(model, expected, actual, equal):
    names = {f"base_model.model.model.language_model.layers.{layer}.mlp.down_proj.lora_{side}.weight"
             for layer in range(16, 32) for side in ("A", "B")}
    if set(expected) != names or set(actual) != names:
        raise ValueError("Adapter must contain exactly the 32 pinned LoRA tensors")
    for name in sorted(names):
        shape = (8, 9216) if ".lora_A." in name else (2560, 8)
        if tuple(expected[name].shape) != shape or tuple(actual[name].shape) != shape:
            raise ValueError(f"Unexpected adapter shape: {name}")
        if actual[name].dtype != expected[name].dtype:
            raise ValueError(f"Adapter tensor dtype changed: {name}")
        if not equal(actual[name].detach().cpu(), expected[name]):
            raise ValueError(f"Loaded adapter tensor differs from safetensors source: {name}")
    modules = [module for module in model.modules() if hasattr(module, "lora_A")]
    if len(modules) != 16 or any(module.scaling.get("default") != 2.0 for module in modules):
        raise ValueError("Unexpected adapter module count or alpha/r scaling")


def verify_precision(model, torch):
    for name, parameter in model.named_parameters():
        if parameter.device.type != "cuda":
            raise ValueError("All model parameters must reside on CUDA")
        if ".lora_" in name:
            if parameter.dtype != torch.float32:
                raise ValueError("Trained LoRA parameters must retain FP32 source precision")
        elif parameter.dtype != torch.bfloat16:
            # Qwen's recurrent decay/time-step stability scalars may retain FP32.
            if name.rsplit(".", 1)[-1] not in {"A_log", "dt_bias"} or parameter.dtype != torch.float32:
                raise ValueError(f"Unexpected backbone parameter precision: {name}")


def write_json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def run(args):
    started = time.monotonic()
    deadline = started + args.deadline_seconds
    if args.deadline_seconds <= 0 or not math.isfinite(args.deadline_seconds):
        raise ValueError("Deadline must be positive and finite")
    def timeout(_signum, _frame):
        raise TimeoutError("Measurement deadline exceeded; no completion manifest written")
    previous_handler = signal.signal(signal.SIGALRM, timeout)
    signal.setitimer(signal.ITIMER_REAL, args.deadline_seconds)
    try:
        return run_until(args, started, deadline)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)


def run_until(args, started, deadline):
    bundle, model_dir, adapter_dir = map(local_directory, (args.bundle, args.model_dir, args.adapter_dir))
    out = Path(args.out)
    if not out.is_absolute() or out.exists():
        raise ValueError("Output must be an absolute new directory")
    protocol, cases, protocol_sha = load_bundle(bundle)
    if protocol["generation"] != {"max_new_tokens": 512, "do_sample": False, "thinking": False, "seed": 20261103}:
        raise ValueError("Unsupported generation configuration")
    versions = verify_runtime(protocol)
    verify_assets(protocol, model_dir, adapter_dir)
    # Deterministic CUDA matrix multiplication requires this before CUDA initializes.
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    # Imports and all from_pretrained calls happen only after immutable inputs pass.
    import torch
    from transformers import AutoTokenizer, GenerationConfig, Qwen3_5ForConditionalGeneration, StoppingCriteria, StoppingCriteriaList
    from peft import PeftModel, get_peft_model_state_dict
    from safetensors.torch import load_file
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(including_emulation=False):
        raise RuntimeError("A CUDA GPU with native BF16 support is required")
    torch.manual_seed(protocol["generation"]["seed"])
    torch.cuda.manual_seed_all(protocol["generation"]["seed"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.cuda.reset_peak_memory_stats()
    gpu = torch.cuda.get_device_properties(0)
    gpu_uuid = str(getattr(gpu, "uuid", "") or "")
    if not gpu_uuid:
        raise ValueError("CUDA runtime must expose a physical GPU UUID for paired binding")
    runtime = {"schema": "ruvllm-measurement-runtime/v1", "versions": versions, "device": "cuda",
               "gpu_name": torch.cuda.get_device_name(), "gpu_uuid": gpu_uuid, "gpu_total_memory_bytes": gpu.total_memory,
               "torch_cuda_version": torch.version.cuda, "python_version": platform.python_version(), "fixture": False,
               "backbone_precision": "bf16-with-native-fp32-stability-parameters", "adapter_precision": "float32"}
    model_sha = object_sha({**protocol["model"]["weight_files"], **protocol["model"]["asset_files"]})
    adapter_sha = protocol["adapter"]["weights_sha256"]
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True, trust_remote_code=False, token=False)
    if tokenizer.eos_token_id not in EOS_IDS:
        raise ValueError("Tokenizer EOS token does not match model generation configuration")
    base = Qwen3_5ForConditionalGeneration.from_pretrained(str(model_dir), local_files_only=True,
            trust_remote_code=False, token=False, dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa")
    # Eager allocation keeps the FP32 trained adapter bit-exact; backbone stays BF16.
    # low_cpu_mem_usage=True materializes LoRA tensors in the BF16 base dtype first,
    # rounding them before the FP32 upcast (caught by verify_adapter below).
    model = PeftModel.from_pretrained(base, str(adapter_dir), local_files_only=True, is_trainable=False,
                                      low_cpu_mem_usage=False, autocast_adapter_dtype=True)
    expected = load_file(str(adapter_dir / "adapter_model.safetensors"), device="cpu")
    # Disable embedding auto-detection: it may query the adapter's remote base repo.
    verify_adapter(model, expected, get_peft_model_state_dict(model, save_embedding_layers=False), torch.equal)
    model.eval()
    verify_precision(model, torch)
    generation_config = GenerationConfig(max_new_tokens=512, do_sample=False, num_beams=1,
                                         eos_token_id=EOS_IDS, pad_token_id=tokenizer.pad_token_id or EOS_IDS[0], use_cache=True)
    load_seconds = time.monotonic() - started
    class DeadlineStop(StoppingCriteria):
        def __call__(self, input_ids, scores, **kwargs):
            if time.monotonic() >= deadline:
                raise TimeoutError("Measurement inference deadline exceeded")
            return False
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "runtime.json", runtime)
    runtime_sha = file_sha(out / "runtime.json")
    count = 0
    with (out / "responses.jsonl").open("x", encoding="utf-8") as stream:
        with torch.inference_mode(), (model.disable_adapter() if args.arm == "base" else nullcontext()):
            for case in cases:
                ids = render_prompt(tokenizer, case)
                if len(ids) + 512 > model.config.text_config.max_position_embeddings:
                    raise ValueError("Prompt would exceed the untruncated context window")
                inputs = torch.tensor([ids], dtype=torch.long, device="cuda")
                torch.cuda.synchronize()
                tick = time.monotonic()
                output = model.generate(input_ids=inputs, attention_mask=torch.ones_like(inputs),
                        generation_config=generation_config, stopping_criteria=StoppingCriteriaList([DeadlineStop()]))
                torch.cuda.synchronize()
                latency = time.monotonic() - tick
                if output.shape[0] != 1 or not torch.equal(output[0, :len(ids)], inputs[0]):
                    raise ValueError("Generation changed the input prefix")
                generated = output[0, len(ids):].tolist()
                if not 0 < len(generated) <= 512 or time.monotonic() >= deadline:
                    raise ValueError("Empty, over-budget, or expired generation")
                finish = "stop" if generated[-1] in EOS_IDS else "length"
                if finish == "length" and len(generated) != 512:
                    raise ValueError("Generation stopped without EOS or token limit")
                # Preserve special tokens: scorer handles closed reasoning/tool delimiters.
                row = {"id": case["id"], "run_id": protocol["run_id"], "protocol_sha256": protocol_sha,
                       "arm": args.arm, "prompt_sha256": object_sha({"messages": case["messages"], "tools": case["tools"]}),
                       "text": tokenizer.decode(generated, skip_special_tokens=False), "finish_reason": finish,
                       "output_tokens": len(generated), "token_ids": generated, "latency_seconds": latency,
                       "model_sha256": model_sha, "adapter_sha256": adapter_sha, "runtime_sha256": runtime_sha}
                stream.write(json.dumps(row, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n")
                stream.flush()
                count += 1
                print(json.dumps({"arm": args.arm, "completed": count, "total": len(cases)}), flush=True)
    if count != protocol["case_count"]:
        raise ValueError("Incomplete generation count")
    write_json(out / "completion.json", {"schema": "ruvllm-measurement-completion/v1", "run_id": protocol["run_id"],
               "arm": args.arm, "protocol_sha256": protocol_sha, "responses_sha256": file_sha(out / "responses.jsonl"),
               "model_sha256": model_sha, "adapter_sha256": adapter_sha, "runtime_sha256": runtime_sha,
               "generation_count": count, "inference_performed": True,
               "metrics": {"load_seconds": load_seconds, "elapsed_seconds": time.monotonic() - started,
                           "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated()}})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("bundle", "model-dir", "adapter-dir", "out"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--arm", choices=("base", "adapter"), required=True)
    parser.add_argument("--deadline-seconds", type=float, required=True)
    args = parser.parse_args()
    try:
        run(args)
    except Exception as exc:
        parser.exit(1, f"Measurement failed: {type(exc).__name__}: {exc}\n")


if __name__ == "__main__":
    main()
