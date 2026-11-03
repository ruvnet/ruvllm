# Frozen Vast cloud comparison

This experiment compares five BF16 artifacts from three models across two Qwen families. It measures two distinct interventions: the published Qwen3.5 0.8B sparse repair and the previously selected Qwen3.5 4B coding adapter. Qwen3 Coder 30B A3B is an unchanged coding reference. No edit is transferred between architectures. No result authorizes deployment or establishes state of the art.

The prior local 16 problem repair experiment failed its quality gate. This cloud experiment tests broader behavior with stronger output budgets and records failures as well as successes.

| Panel | Cases per direct arm | Maximum output tokens per case |
| :--- | ---: | ---: |
| HumanEval, original tests with frozen complete function chat prompts | 164 | 2048 |
| Existing synthetic coding tool release fixtures | 27 | 2048 |
| Original procedural integer arithmetic | 128 | 8192 |

Five direct arms and two additional 0.8B thinking math arms produce 1,851 generations, at most 9,295,872 output tokens. At an assumed aggregate 500 tokens per second the token cap alone takes 5.2 hours. This is a planning estimate, not a measured speed. The worker stops at an absolute external deadline and at 20,700 seconds; the provider lifecycle guard owns the shorter lease limit and destroys compute separately. Downloads, model loading, package installation, and native compilation consume lease time.

## Pins and scope

The three [official model cards](https://huggingface.co/Qwen/Qwen3.5-0.8B), [4B model card](https://huggingface.co/Qwen/Qwen3.5-4B), and [Coder model card](https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct) declare Apache 2.0 licensing. `models.json` records immutable revisions and published Safetensors SHA256 values. The original [HumanEval source](https://github.com/openai/human-eval/tree/b9453e8f1865df0a0057a1383767caa400acfbcf) is MIT licensed; this chat evaluation is not its official leaderboard protocol. Pretraining contamination is unknown.

Runtime is vLLM 0.21.0, Transformers 5.12.1 and Torch 2.11.0, using the pinned amd64 container in `protocol.json`. The [vLLM 0.21 registry](https://github.com/vllm-project/vllm/blob/v0.21.0/vllm/model_executor/models/registry.py) supports the relevant Qwen architectures. Its [Docker source](https://github.com/vllm-project/vllm/blob/v0.21.0/docker/Dockerfile) uses CUDA 13.0.2. Runtime versions, all installed packages, GPU identity, driver, model tree digests, frozen prompt digests, token IDs, termination reason, and batch timing are recorded. Temperature is zero, the seed and batch order are fixed, prefix caching is disabled, and every arm starts in a new process. Hardware dependent numeric differences remain possible.

The FP32 LoRA matrices merge as `BF16(FP32(W) + 2 B A)` into the 16 authorized 4B down projection tensors. Every untouched tensor is compared to its original bytes, and the base checkpoint remains unchanged. This measures BF16 deployment of the selected adapter, separately from earlier MLX 4 bit measurements.

Both official Coder EOS IDs, 151645 and 151643, are explicit stops. Qwen3.5 uses tokenizer chat end token 248046 and model configuration end of text token 248044. The unused initial freeze was archived before any GPU generation when the secondary Coder stop was identified. Corpus and budgets did not change.

The first cloud runtime attempt is preserved under `archive/unused-runtime-attempt-01`. Its unscored preflight stopped before LLM construction because the declared stop set omitted the tokenizer chat end token 248046. Inspection also confirmed that Transformers 5.12.1 defaults to a dictionary from `apply_chat_template`. Before any model generation, the Qwen3.5 stop sets were corrected to include both declared end tokens, and the worker explicitly requested a flat token ID list with `return_dict=False`. Original source files, freezes, runtime signature, EOS configuration evidence, and the exact failure log are archived. The intermediate tokenizer only fix was never uploaded or run and is separately archived. All other experimental settings and fixtures are unchanged.

## Public reproduction scope

This directory preserves the frozen research implementation, case corpus, protocol, model revisions, licenses and correction history. It does not include the original provider controller, cloud account configuration, private lifecycle receipts, complete worker filesystem or compiled historical native executable. The 1,851 rows above describe the planned experiment, not a completed result in this source directory. See [the bundled benchmark evidence](../../benchmarks/README.md) for the completed local experiments.

The [standalone native engine](../../native/README.md) is the portable entry point for tensor inspection, patching, rollback and RVF verification. Its source is traced to the same audited implementation, but its smaller dependency graph and newly compiled executable have distinct digests. Do not substitute its digest into a historical receipt.

The procedure below records the original full workspace workflow. `prepare.py prepare` expects patch and adapter files under the original `runtime/` layout, and `prepare.py freeze` overwrites the protocol. Neither should be run against this preserved snapshot. For a new experiment, work in a separate checkout, explicitly supply the public input artifacts matching `assets.json`, provision your own bounded compute lifecycle, and freeze a new protocol before generating model output. The shipped [PEFT adapter](../../weights/coding-4b/README.md) identifies the selected coding weights; its packaging metadata may differ from the original input configuration, so verify every frozen input digest separately.

The historical [native preparation notes](native/README.md) contain the original workstation paths and require a source archive from the original full RuVector commit. Those paths and archive receipts are evidence, not runnable defaults for a fresh clone. This repository does not distribute that full source archive or the `scripts/vast_native_prepare.sh` orchestration wrapper. A separately prepared native build or changed orchestration belongs to a newly recorded run.

## Original workspace procedure

1. Run `python research/vast_cross_model/prepare.py prepare`. It writes the cases, public prompts, model pins, and input hashes, and copies the existing patch, policy, and adapter to `runtime/vast-cross-model/inputs`.
2. Run the local tests, then `python research/vast_cross_model/prepare.py freeze` before any benchmark generation. Commit the freeze and runner together. Never edit a frozen protocol after observing model output.
3. Copy the checkout and public input bundle to the worker. Supply only public source, models, synthetic prompts, and the trained public adapter. Keep provider credentials on the control host. Install exactly `transformers==5.12.1` inside the pinned vLLM container and verify the three pinned package versions before model loading.
4. Run `models_prepare.py download --work WORK --model KEY` for each of the three original model keys in `models.json`. Build and apply the native 0.8B patch using `scripts/vast_native_prepare.sh`. Run `models_prepare.py register-native --work WORK` and `models_prepare.py merge --work WORK`.
5. Run `worker.py --work WORK --arm qwen035-08b-base-direct --preflight --deadline EPOCH` and the corresponding unscored Coder preflight. A runtime failure is not a scored model failure; preserve logs and stop before scored inference. A revision to the implementation requires a new freeze before benchmark generation.
6. Run `run_all.py --work WORK --deadline EPOCH`. It refuses partial arm retries, preserves completed arms, and kills child process groups at the absolute deadline. The outer provider guard remains required.
7. Retrieve the full raw outputs and receipts. Run `score.py --work WORK --output RESULTS` locally on macOS with the existing protected Python test runner. Model output is never executed on the GPU host. Perform native rollback and compare the original checkpoint bytes independently.

`answer_correct`, `completed`, and `delivered_correct` are separate. A boxed answer inside unfinished reasoning does not count as delivered. A length stop requires exactly the declared output cap. Tool scoring rejects duplicate arguments, malformed envelopes, incorrect schema, and unauthorized argument scope. HumanEval execution uses the existing macOS Seatbelt sandbox and a restricted AST. Batched throughput is not single request latency. Paired gain and regression counts and exploratory exact McNemar values are reported per panel without selecting a model on these results.

Acceptance: all 1,851 frozen rows have the declared model and protocol bindings, each raw output digest verifies, the local scorer produces all seven arms, every source checkpoint remains unchanged, and native rollback restores the exact original 0.8B SHA256. Any missing arm makes the report incomplete.
