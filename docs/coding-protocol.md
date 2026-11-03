# Experiment protocol

Objective: reduce the cost of narrow coding and tool use requests routed from Claude Code while preserving correctness. An open base and small adapter support distribution without redistributing Claude weights.

## Inputs and outputs

1. Model preparation takes the pinned Apache 2.0 Qwen3.5 4B repository and produces a hashed MLX 4 bit model with group size 64. The original BF16 download remains intact.
2. Training takes only the original training and validation traces and produces PEFT adapters at steps 32, 64, and 96. Only 16 MLP output projections receive rank 8 updates. The remaining base parameters stay frozen. Training is BF16 on an A100; deployment uses the separately measured quantized base.
3. Development evaluation takes each checkpoint and the same frozen prompts, temperature zero, seed 20261102, disabled thinking, and 768 output tokens. It produces raw responses, executable scores, token counts, completion status, and wall time.
4. Selection requires at least one additional correct case without group regression, or equal correctness with at least 10% fewer output tokens. Two consecutive nonimprovements stop checkpoint evaluation. No failed trial is removed.
5. Release evaluation takes the selected checkpoint once. The synthetic release set and 164 task HumanEval suite measure behavior after selection. Synthetic group pass counts and the external aggregate pass count must not fall. This gate can pass even when individual cases regress; report both gains and losses. These tests do not establish broad repository task quality or SOTA.
6. Integration runs a real isolated Claude Code Router and the installed Claude Code CLI against scratch fixtures. It verifies a physical file read with a fresh verification value, then a Read, Edit, Bash repair cycle with independent assertions. Tool requests alone never count as successful execution.

Before accepting traffic, the fixed server compares all 32 in memory adapter tensors with the saved artifact and records the weight and configuration hashes. HTTP model discovery alone does not establish loaded model readiness. Integration receipts must match this loaded tensor record.

## Controls

The controller allows at most four evaluation iterations, three training checkpoints, four hours total wall time, and one hour per stage. A file lock prevents duplicate workers. A stop request terminates the active local process group and cancels its cloud training job. Failed stages require explicit inspection and resume. Cloud jobs also have provider timeouts independent of the local controller.

The USD 200 cloud cap uses integer micro USD reservations before dispatch. The full timeout plus one billing minute is reserved. Reservations remain charged against the local cap even if a job fails or its actual bill is lower. Network ambiguity never triggers automatic redispatch. Billing estimates use the live hardware API. Actual invoices remain provider records.

Cloud inputs are original synthetic data and public model weights. Checkpoints live in a private bucket mounted by the provider. Cloud workers receive no user access token, local repositories, or private conversations. No public model upload occurs automatically.

Evaluation code runs under a macOS Seatbelt profile with a minimal environment, no network or filesystem writes, a CPU deadline, and a restricted import and introspection policy. Unsupported operations are reported as failures. This evaluator fails closed on other platforms. It is intended for the controlled benchmark, not as a general untrusted code hosting service.

## Known limitations

The pilot has few independent template families, one decoding seed, and only 8 synthetic coding tasks per split. Small pass count improvements can be statistical noise. HumanEval measures short Python functions and may have appeared in base pretraining. Real repository editing, long context, multilingual programming, sustained tool error recovery, concurrent serving, and long running sessions need separate evaluation.

Quantization and adapter training are separate interventions. Compare the quantized baseline with the adapter on the same quantized base. Report the BF16 baseline separately. A100 and Mac timings use different hardware and runtimes and are not direct latency comparisons.

Anthropic's gateway documentation does not support non Claude models. CCR compatibility is verified locally and may change with client updates. The deployment profile supports text only and 16384 total tokens. Use a stronger router fallback for other requests.

Sources: [Qwen model](https://huggingface.co/Qwen/Qwen3.5-4B), [MLX LM](https://github.com/ml-explore/mlx-lm), [PEFT](https://github.com/huggingface/peft), [CCR](https://github.com/musistudio/claude-code-router), [HumanEval](https://github.com/openai/human-eval), [Anthropic gateway documentation](https://code.claude.com/docs/en/llm-gateway), [HF Jobs pricing](https://huggingface.co/docs/hub/jobs-pricing).
