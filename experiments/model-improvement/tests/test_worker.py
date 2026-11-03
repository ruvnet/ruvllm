"""CPU contract tests only: these do not generate or measure model responses."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("measurement_worker", ROOT / "worker.py")
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


class Tensor:
    def __init__(self, shape, value=1):
        self.shape, self.value, self.dtype = shape, value, "float32"

    def detach(self):
        return self

    def cpu(self):
        return self


class WorkerContracts(unittest.TestCase):
    def test_runtime_pins_reject_before_inference(self):
        protocol = {"runtime_versions": worker.RUNTIME_PINS.copy()}
        result = worker.verify_runtime(protocol, lambda name: worker.RUNTIME_PINS[name] + "+cu130")
        self.assertEqual(result["torch"], "2.11.0+cu130")
        with self.assertRaisesRegex(ValueError, "version mismatch"):
            worker.verify_runtime(protocol, lambda name: "0.0.0")
        protocol["runtime_versions"]["torch"] = "0.0.0"
        version = Mock(side_effect=AssertionError("No runtime imports for bad protocol"))
        with self.assertRaisesRegex(ValueError, "Protocol runtime"):
            worker.verify_runtime(protocol, version)
        version.assert_not_called()

    def test_prompt_contract_tools_and_thinking_disabled(self):
        tokenizer = Mock()
        tokenizer.apply_chat_template.return_value = [123, 456]
        case = {"messages": [{"role": "user", "content": "read file"}], "tools": [{"type": "function"}]}
        self.assertEqual(worker.render_prompt(tokenizer, case), [123, 456])
        tokenizer.apply_chat_template.assert_called_once_with(case["messages"], tools=case["tools"],
                tokenize=True, add_generation_prompt=True, enable_thinking=False, return_dict=False)
        for bad in ([], [[1]], [True], [1, -1], "tokens"):
            tokenizer.apply_chat_template.return_value = bad
            with self.assertRaises(ValueError):
                worker.render_prompt(tokenizer, case)

    def fixture(self, root):
        """Dummy bytes test integrity rejection, never inference or accuracy."""
        model, adapter = root / "model", root / "adapter"
        model.mkdir()
        adapter.mkdir()
        (model / "model.safetensors").write_bytes(b"dummy weight contract bytes")
        documents = {"config.json": {"model_type": "qwen3_5", "architectures": ["Qwen3_5ForConditionalGeneration"]},
                     "tokenizer.json": {}, "tokenizer_config.json": {}, "chat_template.jinja": "template"}
        for name, value in documents.items():
            (model / name).write_text(json.dumps(value) if name.endswith("json") else value)
        source = ROOT.parents[1] / "weights" / "coding-4b" / "adapter_config.json"
        (adapter / "adapter_config.json").write_bytes(source.read_bytes())
        (adapter / "adapter_model.safetensors").write_bytes(b"dummy adapter contract bytes")
        protocol = {"model": {"repo_id": worker.MODEL_REPO, "revision": worker.MODEL_REVISION,
                       "weight_files": {"model.safetensors": worker.file_sha(model / "model.safetensors")},
                       "asset_files": {name: worker.file_sha(model / name) for name in documents}},
                    "adapter": {"weights_sha256": worker.file_sha(adapter / "adapter_model.safetensors"),
                                "config_sha256": worker.file_sha(adapter / "adapter_config.json")}}
        return protocol, model, adapter

    def test_complete_hash_validation_rejects_tampered_assets_and_weights(self):
        for filename in ("model.safetensors", "config.json", "tokenizer.json", "chat_template.jinja"):
            with self.subTest(filename=filename), tempfile.TemporaryDirectory() as tmp:
                protocol, model, adapter = self.fixture(Path(tmp))
                worker.verify_assets(protocol, model, adapter)
                (model / filename).write_bytes(b"tampered")
                with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                    worker.verify_assets(protocol, model, adapter)
        with tempfile.TemporaryDirectory() as tmp:
            protocol, model, adapter = self.fixture(Path(tmp))
            (adapter / "adapter_model.safetensors").write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "adapter_model.safetensors"):
                worker.verify_assets(protocol, model, adapter)

    def test_unpinned_generation_config_and_extra_shards_rejected(self):
        for name in ("generation_config.json", "model-00001-of-00001.safetensors"):
            with self.subTest(filename=name), tempfile.TemporaryDirectory() as tmp:
                protocol, model, adapter = self.fixture(Path(tmp))
                (model / name).write_text("{}")
                with self.assertRaises(ValueError):
                    worker.verify_assets(protocol, model, adapter)

    def test_model_identity_and_path_escape_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            protocol, model, adapter = self.fixture(Path(tmp))
            bad = copy.deepcopy(protocol)
            bad["model"]["revision"] = "unreviewed"
            with self.assertRaisesRegex(ValueError, "revision"):
                worker.verify_assets(bad, model, adapter)
            with self.assertRaisesRegex(ValueError, "single local filename"):
                worker.checked_file(model, "../secret", "unused")

    def adapter_state(self):
        return {f"base_model.model.model.language_model.layers.{layer}.mlp.down_proj.lora_{side}.weight":
                Tensor((8, 9216) if side == "A" else (2560, 8))
                for layer in range(16, 32) for side in ("A", "B")}

    def test_adapter_exact_values_shapes_count_and_scale(self):
        model = types.SimpleNamespace(modules=lambda: [types.SimpleNamespace(lora_A=True, scaling={"default": 2.0})] * 16)
        expected, actual = self.adapter_state(), self.adapter_state()
        equal = lambda a, b: a.value == b.value
        worker.verify_adapter(model, expected, actual, equal)
        name = next(iter(actual))
        for attribute, bad in (("value", 2), ("shape", (8, 99)), ("dtype", "bfloat16")):
            actual = self.adapter_state()
            setattr(actual[name], attribute, bad)
            with self.subTest(attribute=attribute), self.assertRaises(ValueError):
                worker.verify_adapter(model, expected, actual, equal)
        actual = self.adapter_state()
        del actual[name]
        with self.assertRaisesRegex(ValueError, "32 pinned"):
            worker.verify_adapter(model, expected, actual, equal)
        model.modules = lambda: [types.SimpleNamespace(lora_A=True, scaling={"default": 1.0})] * 16
        with self.assertRaisesRegex(ValueError, "scaling"):
            worker.verify_adapter(model, expected, self.adapter_state(), equal)

    def test_json_output_is_exclusive_and_rejects_nonfinite(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "runtime.json"
            worker.write_json(path, {"fixture": False})
            with self.assertRaises(FileExistsError):
                worker.write_json(path, {})
            with self.assertRaises(ValueError):
                worker.write_json(Path(tmp) / "bad.json", {"value": float("nan")})

    def test_bf16_backbone_allows_only_native_stability_scalars_in_fp32(self):
        torch = types.SimpleNamespace(bfloat16="bfloat16", float32="float32")
        def parameter(dtype, device="cuda"):
            return types.SimpleNamespace(dtype=dtype, device=types.SimpleNamespace(type=device))
        rows = [("base.weight", parameter("bfloat16")), ("linear_attn.A_log", parameter("float32")),
                ("linear_attn.dt_bias", parameter("float32")), ("down_proj.lora_A.default.weight", parameter("float32"))]
        model = types.SimpleNamespace(named_parameters=lambda: rows)
        worker.verify_precision(model, torch)
        for name, item in (("base.weight", parameter("float32")), ("base.weight", parameter("bfloat16", "cpu")),
                           ("down_proj.lora_A.default.weight", parameter("bfloat16"))):
            model.named_parameters = lambda: [(name, item)]
            with self.subTest(name=name, dtype=item.dtype), self.assertRaises(ValueError):
                worker.verify_precision(model, torch)

    def test_rejected_runtime_writes_no_inference_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            protocol, model, adapter = self.fixture(root)
            protocol["runtime_versions"] = dict(worker.RUNTIME_PINS, torch="unapproved")
            protocol["generation"] = dict(max_new_tokens=512, do_sample=False, thinking=False, seed=20261103)
            bundle = root / "bundle"
            bundle.mkdir()
            out = root / "out"
            args = types.SimpleNamespace(bundle=str(bundle), model_dir=str(model), adapter_dir=str(adapter), out=str(out), arm="base")
            with unittest.mock.patch.object(worker, "load_bundle", return_value=(protocol, [], "unused")):
                with self.assertRaisesRegex(ValueError, "Protocol runtime"):
                    worker.run_until(args, 0, 100)
            self.assertFalse(out.exists(), "Rejected runs must not emit evidence or completion")


if __name__ == "__main__":
    unittest.main()
