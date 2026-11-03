"""Pinned MLX server with fixed adapter binding and checked loaded tensors."""
import os
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten
from mlx_lm import server
from mlx_lm.tool_parsers import qwen3_coder

from .common import ROOT, digest, now, write_json
from .strict_tools import ensure_unique_parameters

original_parser = qwen3_coder.parse_tool_call


def strict_parser(text, tools=None):
    ensure_unique_parameters(text)
    return original_parser(text, tools)


qwen3_coder.parse_tool_call = strict_parser


class BoundModelProvider(server.ModelProvider):
    def load(self, model_path, adapter_path=None, draft_model_path=None):
        configured_model = self.cli_args.model
        configured_adapter = self.cli_args.adapter_path
        if model_path not in ("default_model", configured_model):
            raise ValueError("Only the configured model is available")
        if adapter_path not in (None, configured_adapter):
            raise ValueError("Only the configured adapter is available")
        if draft_model_path not in (None, "default_model") or self.cli_args.draft_model:
            raise ValueError("Draft models are unsupported by this fixed profile")
        # MLX LM 0.31.3 resolves the model alias before looking up its adapter.
        # Bind both together here so a concrete model path cannot drop LoRA.
        key = (configured_model, configured_adapter, None)
        if self.model_key != key:
            self._load(*key)
        if self.model_key != key:
            raise RuntimeError("Loaded model identity differs from its configuration")
        return self.model, self.tokenizer

    def _load(self, model_path, adapter_path=None, draft_model_path=None):
        super()._load(model_path, adapter_path, draft_model_path)
        matched = 0
        if adapter_path:
            expected = mx.load(str(Path(adapter_path) / "adapters.safetensors"))
            actual = dict(tree_flatten(self.model.parameters()))
            for name, value in expected.items():
                if name not in actual or not bool(mx.array_equal(value, actual[name])):
                    raise ValueError("Loaded adapter tensor mismatch: " + name)
                matched += 1
            if matched != 32:
                raise ValueError("Expected 32 adapter tensors for this model profile")
        write_json(ROOT / "runtime/server/backend-state.json", {
            "loaded_at": now(), "backend_pid": os.getpid(), "model": model_path,
            "adapter": adapter_path, "matched_adapter_tensors": matched,
            "adapter_sha256": digest(Path(adapter_path) / "adapters.safetensors") if adapter_path else None,
            "adapter_config_sha256": digest(Path(adapter_path) / "adapter_config.json") if adapter_path else None,
            "binding_verified": True})


if __name__ == "__main__":
    server.ModelProvider = BoundModelProvider
    server.main()
