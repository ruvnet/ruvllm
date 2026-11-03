"""Anonymous pinned model downloads and audited, model specific adapter merge."""
from __future__ import annotations
import argparse
import gc
import os
from pathlib import Path
import shutil
import time
from common import HERE, clean_environment, read, sha, tree, verify_protocol, write


def verify_assets(work):
    for name, expected in read(HERE / "assets.json").items():
        if sha(work / "inputs" / name) != expected:
            raise ValueError(f"Input digest mismatch: {name}")


def download(work, key):
    from huggingface_hub import snapshot_download
    spec=next(s for s in read(HERE/"models.json") if s["key"]==key)
    path=work/"models"/key
    snapshot_download(spec["model"], revision=spec["revision"], local_dir=str(path), token=False,
                      allow_patterns=["*.json","*.safetensors","*.jinja","*.txt","*.model","LICENSE*"], max_workers=8)
    for name, expected in spec["weights_sha256"].items():
        if sha(path/name) != expected: raise ValueError(f"Source weight mismatch: {name}")
    manifest={**tree(path), "variant":key,"source":spec,"operation":"pinned original"}
    receipt=work/"models"/(key+".manifest.json")
    if receipt.exists():
        if read(receipt) != manifest: raise ValueError("Existing model manifest differs")
    else: write(receipt,manifest)
    return path,manifest


def merge(work):
    """No PEFT dispatch ambiguity: exactly W' = BF16(FP32(W) + 2 BA)."""
    import torch
    from safetensors.torch import load_file, save_file
    verify_assets(work)
    source=work/"models/qwen035-4b-base"
    base_manifest=read(work/"models/qwen035-4b-base.manifest.json")
    if tree(source)["sha256"] != base_manifest["sha256"]: raise ValueError("Base changed")
    target=work/"models/qwen035-4b-lora"
    target.mkdir(parents=True,exist_ok=False)
    adapter=load_file(str(work/"inputs/adapter_model.safetensors"),device="cpu")
    config=read(work/"inputs/adapter_config.json")
    assert config["r"] == 8 and config["lora_alpha"] == 16
    assert config["target_modules"] == ["down_proj"] and config["layers_to_transform"] == list(range(16,32))
    assert not config["use_dora"] and not config["use_rslora"] and not config["fan_in_fan_out"]
    expected={f"base_model.model.model.language_model.layers.{n}.mlp.down_proj.lora_{ab}.weight" for n in range(16,32) for ab in ("A","B")}
    assert set(adapter)==expected and all(t.dtype==torch.float32 and torch.isfinite(t).all() for t in adapter.values())
    changed=[]
    unchanged=0
    started=time.monotonic()
    for file in sorted(source.iterdir()):
        if not file.is_file(): continue
        if file.suffix != ".safetensors":
            shutil.copyfile(file,target/file.name)
            continue
        tensors=load_file(str(file),device="cpu")
        original_hash={k:__import__('hashlib').sha256(t.view(torch.uint8).numpy().tobytes()).hexdigest() for k,t in tensors.items()}
        for n in range(16,32):
            tensor=f"model.language_model.layers.{n}.mlp.down_proj.weight"
            if tensor not in tensors: continue
            a=adapter[f"base_model.model.{tensor[:-7]}.lora_A.weight"]
            b=adapter[f"base_model.model.{tensor[:-7]}.lora_B.weight"]
            assert a.shape==(8,9216) and b.shape==(2560,8)
            old=tensors[tensor]
            assert old.dtype==torch.bfloat16 and old.shape==(2560,9216)
            delta=2.0*(b@a)
            updated=(old.float()+delta).to(torch.bfloat16)
            assert torch.isfinite(updated).all() and torch.any(updated!=old)
            changed.append({"tensor":tensor,"changed_values":int((updated!=old).sum()),
                            "source_sha256":original_hash[tensor],
                            "edited_sha256":__import__('hashlib').sha256(updated.view(torch.uint8).numpy().tobytes()).hexdigest(),
                            "l2":float(torch.linalg.vector_norm(updated.float()-old.float()))})
            tensors[tensor]=updated
        save_file(tensors,str(target/file.name),metadata={"format":"pt"})
        saved=load_file(str(target/file.name),device="cpu")
        assert set(saved)==set(tensors)
        target_names={r["tensor"] for r in changed}
        for name, value in saved.items():
            assert torch.equal(value,tensors[name])
            if name not in target_names:
                assert __import__('hashlib').sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest()==original_hash[name]
                unchanged+=1
        del tensors,saved
        gc.collect()
    assert len(changed)==16
    assert tree(source)["sha256"] == base_manifest["sha256"]
    manifest={**tree(target),"variant":"qwen035-4b-lora","operation":"FP32 W + 2 BA, BF16 output",
              "base_sha256":base_manifest["sha256"],"adapter_sha256":sha(work/"inputs/adapter_model.safetensors"),
              "changed_tensors":changed,"untouched_tensor_hashes_verified":unchanged,"seconds":time.monotonic()-started}
    write(work/"models/qwen035-4b-lora.manifest.json",manifest)
    return manifest


def register_native(work):
    source=work/"models/qwen035-08b-base"
    target=work/"models/qwen035-08b-edited"
    original=read(work/"models/qwen035-08b-base.manifest.json")
    assert tree(source)["sha256"]==original["sha256"]
    name="model.safetensors-00001-of-00001.safetensors"
    assert sha(target/name)=="bd5c223dc7d0ed48a6857f798a94da6ec46c40b9ff4e7bc31872dba10f943ae5"
    for path in source.iterdir():
        if path.is_file() and path.name != name:
            other=target/path.name
            if other.exists(): assert sha(other)==sha(path)
            else: shutil.copyfile(path,other)
    actual=tree(target)
    assert {k for k,v in actual["files"].items() if v != original["files"].get(k)}=={name}
    write(work/"models/qwen035-08b-edited.manifest.json",{**actual,"variant":"qwen035-08b-edited",
          "operation":"native bounded published BF16 sparse patch", "base_sha256":original["sha256"],
          "patch_sha256":sha(work/"inputs/patch.json"),"policy_sha256":sha(work/"inputs/policy.json")})


if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("action",choices=["download","merge","register-native"])
    p.add_argument("--work",type=Path,required=True)
    p.add_argument("--model")
    args=p.parse_args()
    clean=clean_environment()
    os.environ.clear()
    os.environ.update(clean)
    verify_protocol()
    verify_assets(args.work)
    if args.action=="download": download(args.work,args.model)
    elif args.action=="merge": merge(args.work)
    else: register_native(args.work)
