import json
import numpy as np
from safetensors.numpy import save_file, load_file

from ruflo_model.cloud_train import convert
from ruflo_model.common import ROOT, digest, write_json


def test_peft_to_mlx_delta_parity(tmp_path):
    source, out = tmp_path / "source", tmp_path / "out"
    files = {}
    rng = np.random.default_rng(7)
    a = rng.normal(size=(8, 13)).astype(np.float32)
    b = rng.normal(size=(5, 8)).astype(np.float32)
    for step in (32, 64, 96):
        folder = source / f"step-{step:03d}"
        folder.mkdir(parents=True)
        weights = {}
        for layer in range(16,32):
            weights[f"base_model.model.model.language_model.layers.{layer}.mlp.down_proj.lora_A.weight"] = a
            weights[f"base_model.model.model.language_model.layers.{layer}.mlp.down_proj.lora_B.weight"] = b
        path = folder / "adapter_model.safetensors"
        save_file(weights, str(path))
        files[str(path.relative_to(source))] = digest(path)
    revision = json.loads((ROOT / "model_pin.json").read_text())["revision"]
    write_json(source / "manifest.json", {"files": files, "revision": revision,
        "training_sha256": digest(ROOT / "data/training/train.jsonl")})
    convert(source, out)
    converted = load_file(str(out / "0000032_adapters.safetensors"))
    config = json.loads((out / "adapter_config.json").read_text())
    x = rng.normal(size=(3,13)).astype(np.float32)
    peft_result = (b @ a @ x.T).T * 2
    mlx_result = x @ converted["language_model.model.layers.16.mlp.down_proj.lora_a"] @ converted["language_model.model.layers.16.mlp.down_proj.lora_b"] * config["lora_parameters"]["scale"]
    np.testing.assert_allclose(peft_result, mlx_result, rtol=1e-5, atol=1e-5)
