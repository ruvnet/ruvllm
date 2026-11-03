"""Pinned BF16 LoRA training. Only final assistant target tokens enter the loss."""
import base64
import gzip
import hashlib
import json
import os
from pathlib import Path
import random
import time

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForImageTextToText, AutoTokenizer

MODEL = "Qwen/Qwen3.5-4B"
REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"


def emit(kind, data):
    print("RUFLO_TRAIN " + json.dumps({"kind": kind, "data": data}), flush=True)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    seed = 20261102
    torch.manual_seed(seed)
    random.seed(seed)
    torch.set_num_threads(8)
    payload = json.loads(gzip.decompress(base64.b64decode(os.environ["RUFLO_TRAIN_GZIP"])))
    destination = Path("/checkpoints") / "training-v1"
    destination.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, token=False, trust_remote_code=False)

    def encode(row):
        ids = tokenizer.apply_chat_template(row["messages"], tools=row["tools"], return_dict=False)
        prefix = tokenizer.apply_chat_template(row["messages"][:-1], tools=row["tools"],
            add_generation_prompt=True, enable_thinking=False, return_dict=False)
        # Prefix equality prevents an off by one mask or supervising prompt tokens.
        if ids[:len(prefix)] != prefix or not len(prefix) < len(ids) <= 1152:
            raise ValueError("Training template prefix or length mismatch")
        return ids, len(prefix)

    train = [encode(r) for r in payload["train"]]
    valid = [encode(r) for r in payload["valid"]]
    model = AutoModelForImageTextToText.from_pretrained(MODEL, revision=REVISION, token=False,
        trust_remote_code=False, dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa")
    model.config.use_cache = False
    config = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.0, bias="none",
        target_modules=["down_proj"], layers_to_transform=list(range(16, 32)), layers_pattern="layers")
    model = get_peft_model(model, config)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    params = [p for p in model.parameters() if p.requires_grad]
    names = [n for n,p in model.named_parameters() if p.requires_grad]
    if len(names) != 32 or not all("mlp.down_proj.lora_" in n for n in names):
        raise ValueError("Unexpected trainable parameters")
    emit("setup", {"model": MODEL, "revision": REVISION, "trainable_parameters": sum(p.numel() for p in params),
        "tensor_names": names, "train_examples": len(train), "valid_examples": len(valid),
        "gpu": torch.cuda.get_device_name(), "seed": seed})
    optimizer = torch.optim.AdamW(params, lr=5e-5, weight_decay=0)

    def loss_for(example):
        ids, offset = example
        target_count = len(ids) - offset
        inputs = torch.tensor([ids[:-1]], device="cuda", dtype=torch.long)
        labels = torch.tensor(ids[offset:], device="cuda", dtype=torch.long)
        # Computing vocabulary logits for prompt positions would waste >90% of
        # this objective's output projection memory. Keep the target positions.
        logits = model(input_ids=inputs, use_cache=False, logits_to_keep=target_count).logits[0]
        return torch.nn.functional.cross_entropy(logits.float(), labels)

    def validation(step):
        model.eval()
        with torch.no_grad():
            losses = [float(loss_for(row)) for row in valid[:8]]
        emit("validation", {"step": step, "loss": sum(losses) / len(losses)})
        model.train()

    validation(0)
    order = list(range(len(train)))
    random.shuffle(order)
    started = time.monotonic()
    for step in range(1, 97):
        optimizer.zero_grad(set_to_none=True)
        loss_total = 0.0
        for micro in range(2):
            loss = loss_for(train[order[((step - 1) * 2 + micro) % len(order)]])
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            loss_total += float(loss.detach()) / 2
            (loss / 2).backward()
        norm = torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
        optimizer.step()
        if step % 4 == 0:
            emit("step", {"step": step, "loss": loss_total, "gradient_norm": float(norm),
                "elapsed_seconds": round(time.monotonic() - started, 2),
                "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3)})
        if step % 32 == 0:
            validation(step)
            checkpoint = destination / f"step-{step:03d}"
            model.save_pretrained(checkpoint, safe_serialization=True)
            emit("checkpoint", {"step": step, "files": {p.name: sha(p) for p in checkpoint.iterdir() if p.is_file()}})
    manifest = {"model": MODEL, "revision": REVISION, "steps": 96, "training_backend": "transformers-peft-cuda-bf16",
        "seed": seed, "rank": 8, "alpha": 16, "layers": list(range(16,32)), "learning_rate": 5e-5,
        "optimizer": "AdamW", "weight_decay": 0, "gradient_clip": 1.0, "effective_batch_size": 2,
        "target_only_loss": True, "training_sha256": payload["training_sha256"],
        "files": {str(p.relative_to(destination)): sha(p) for p in destination.rglob("*") if p.is_file()}}
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2))
    emit("completed", {"steps": 96, "manifest_sha256": sha(destination / "manifest.json")})


if __name__ == "__main__":
    main()
