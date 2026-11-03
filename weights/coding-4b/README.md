# Coding 4B adapter

This is the selected step 64 adapter from the completed local coding pilot. Both formats contain the same 1,507,328 parameter low rank update for Qwen3.5 4B revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.

| Format | Weight file | Configuration |
| :--- | :--- | :--- |
| PEFT | `adapter_model.safetensors` | `adapter_config.json` |
| MLX | `mlx/adapters.safetensors` | `mlx/adapter_config.json` |

The PEFT update has rank 8 and alpha 16. The MLX conversion transposes A and B and uses multiplicative scale 2. The base model stays frozen. The selected checkpoint consumed 128 original synthetic examples from a 288 trace training pool. The 36 trace validation pool was separate; 8 traces were used for loss checks.

The measured local deployment uses the pinned MLX 4 bit base. HumanEval improved from 123/164 to 128/164, with 13 gains and 8 regressions, exact paired p = 0.383. Synthetic release improved from 29/35 to 32/35. This is a specialist preview, with no state of the art or production qualification claim. See [completed evidence](../../benchmarks/README.md).

The completed A100 GPU experiment merged the FP32 PEFT update into BF16 base tensors. HumanEval rose from 133/164 to 135/164, with 7 gains and 5 regressions, exact paired p = 0.774. Tool fixtures rose from 24/27 to 27/27; procedural math stayed at 127/128. These are separate measurements from the local quantized experiment, and the coding improvement remains inconclusive. See the [full cloud report](../../research/vast_cross_model/results/cross-model-20261103/README.md).

Run the serving command from the [main README](../../README.md). The server verifies all 32 adapter tensors after loading. Base weights download separately. Do not substitute a different base revision or quantization and reuse these scores.

`manifest.json` records each distributed file's SHA256 and size and the exact source archive digest. The adapter files are unchanged from the verified preview archive. The MLX adapter digest is `52e5adefc83cc77357158ab4d4ea667287b3eaef5f87611b40e7a3397fabb0a4`. The PEFT adapter digest is `729d235e0c70c545ad15d73afd371f36a84fe828328ff366f450dc2f0a69f674`.

License: Apache 2.0. Training used original synthetic examples, with no Claude weights, private conversations or private repository data.
