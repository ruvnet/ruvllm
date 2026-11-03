<p align="center"><img src="docs/assets/ruvllm-hero.svg" alt="RuVLLM preserves the original model, changes an approved copy, and compares both on the same tasks." width="100%"></p>

# RuVLLM

**Change a model. Measure the difference. Keep the original.**

RuVLLM is a research toolkit for making small, controlled changes to AI models. You can inspect a model's weights—the numbers it learned during training—edit an approved set of values, and check whether the change helps. The original file stays untouched, and the tools verify that they can restore its exact bytes.

The repository also includes a small coding adapter, Claude Code integrations, experiment scheduling, and published evaluation data. You can try the core editing workflow on your CPU before downloading a model or renting a GPU.

[Try it](#quick-start-no-gpu) · [How it works](#how-it-works) · [Results](#what-have-we-measured) · [Claude Code](#use-it-with-claude-code-or-ruflo) · [GPU evaluation](#measure-a-model-on-a-gpu) · [Guides](#find-your-way-around)

## Pick a starting point

| I want to… | Start here | What I need |
| :--- | :--- | :--- |
| Check the published evidence | [Verify the saved results](#quick-start-no-gpu) | Python 3.12+ |
| Try a real weight edit and restoration | [Run the CPU demo](#try-a-real-edit-on-a-tiny-file) | Python and Rust 1.97.1 |
| Run experiments from Claude Code | [Install the experiment plugin](#use-it-with-claude-code-or-ruflo) | Node 22+, Rust, Python and Claude Code |
| Compare the base model and adapter | [Run the paired GPU evaluation](#measure-a-model-on-a-gpu) | A prepared CUDA GPU host and model weights |
| Serve the included adapter locally | [Apple Silicon setup](#serve-the-coding-adapter-on-a-mac) | Apple Silicon, Python and model storage |

**Where things stand:** CPU editing, exact restoration and local integration tests pass. Historical model evaluations show small, specific adapter gains that remain statistically inconclusive. A new 256-task evaluation is frozen and ready to run; it has **no fresh GPU results yet**. Neither a passing benchmark nor an evidence file automatically deploys a model.

## Quick start: no GPU

First, verify the evidence already included in the repository. These commands do not download model weights, call a provider, or execute model-generated programs.

```sh
git clone https://github.com/ruvnet/ruvllm.git
cd ruvllm
python3 scripts/verify_benchmarks.py
python3 scripts/check_repository.py
```

Use Python 3.12 or newer. Look for `"verified": true`. The benchmark verifier checks 462 saved generations and recomputes their totals. It does not rerun model inference or HumanEval programs.

<details>
<summary><strong>Optional: verify the full GPU evidence archive</strong></summary>

The separate archive is about 39 MB. These additional checks validate its 438 files and 1,851 response bindings without executing archived code or generated programs. Use a trusted repository commit and keep the expected archive digest from its publication manifest.

```sh
python3 -m venv .venv
.venv/bin/python -m pip install jsonschema==4.26.0
mkdir -p runtime/cloud-evidence
curl --fail --location \
  https://github.com/ruvnet/ruvllm/releases/download/cloud-evidence-20261103/ruvllm-cloud-evidence-20261103.tar.gz \
  --output runtime/cloud-evidence/ruvllm-cloud-evidence-20261103.tar.gz
.venv/bin/python scripts/verify_cloud_publication.py \
  --publication research/vast_cross_model/results/cross-model-20261103/publication.json \
  --archive runtime/cloud-evidence/ruvllm-cloud-evidence-20261103.tar.gz
```

Saved HumanEval verdicts are checked for consistency; their generated programs are not rerun.

</details>

### Try a real edit on a tiny file

With the pinned Rust 1.97.1 toolchain available, run:

```sh
cd native
python3 verify_source.py
cargo test --locked --workspace
cargo build --locked --release --bin weight_evolution
python3 smoke.py
cd ..
```

The demo creates a tiny model-weight file, changes one allowed value, rejects forbidden edits, restores the original bytes, and checks the saved evidence. Look for `"passed": true` and matching `source_sha256` / `restored_sha256` values.

This is a real file-editing test with synthetic evaluation scores. It does not demonstrate that a model became smarter. See the [native guide](native/README.md) for editing your own Safetensors files, a weight-file format used by many open models.

## How it works

<p align="center"><img src="docs/assets/governed-lifecycle.svg" alt="Freeze the model, tasks and limits; edit a copy; compare answers; verify exact restoration. Deployment requires separate approval." width="100%"></p>

1. **Fix the rules first.** Record the model version, test questions and allowed changes before seeing the results.
2. **Create a candidate.** Write the edited weights to a new file. A policy limits which values can change and by how much.
3. **Compare both versions.** Give the original and candidate the same tasks and generation settings. Keep the raw answers, including failures.
4. **Check the recovery path.** Restore the original into another file and compare every byte. Keep the results and file fingerprints together.

The native editor checks weight files and evaluation receipts. A separate inference worker must actually run the model and measure its answers. A receipt is a record of evidence; checking its contents is not the same as generating that evidence independently.

### Two different ways to change a model

| Approach | Plain-language explanation | Included example |
| :--- | :--- | :--- |
| **Direct weight edit** | Change specific numbers in a copy of a model, within explicit limits. | A bounded edit to Qwen3.5 0.8B, with exact restoration. |
| **LoRA adapter** | Train a small add-on while keeping the base model frozen. | A 1.5-million-parameter coding adapter for Qwen3.5 4B. |

These are separate experiments. The editor does not discover useful changes on its own, capture model activations, or automatically protect every capability. [ADR 011](docs/adr/ADR-011-governed-neural-weight-evolution.md) explains the implemented boundaries.

## What have we measured?

### Model quality: small gains for the 4B adapter

The following **historical GPU results** compare the same Qwen3.5 4B model with and without its included adapter. Both used BF16 (a 16-bit number format) on one A100 80GB GPU. A pass requires a correct answer that finished within the output budget.

<p align="center"><img src="docs/assets/benchmark-evidence.svg" alt="Historical 4B GPU results: tool passes increased from 24 of 27 to 27 of 27; code passes increased from 133 of 164 to 135 of 164. The gains remain statistically inconclusive." width="100%"></p>

| Task set | Base model | With adapter | Difference |
| :--- | ---: | ---: | :--- |
| Tool-use fixtures | 24 / 27 | 27 / 27 | 3 more passes |
| HumanEval code tasks | 133 / 164 | 135 / 164 | 2 more passes overall |
| Procedural arithmetic | 127 / 128 | 127 / 128 | No change |

The code result includes **7 gains and 5 regressions**, with paired p = 0.774. The tool result has 3 gains and no regressions, with p = 0.25. Those comparisons do not provide strong enough statistical evidence to conclude that the adapter reliably improves unseen tasks. The 27 tool fixtures are also a small synthetic panel.

The full cloud experiment produced 1,851 responses across seven configurations. Read the [report](research/vast_cross_model/results/cross-model-20261103/README.md), [protocol](research/vast_cross_model/README.md), and [public evidence archive](https://github.com/ruvnet/ruvllm/releases/tag/cloud-evidence-20261103) for raw outputs, model identities and comparison limits.

### Faster software does not mean a better model

| Measurement | Observed result | What it establishes |
| :--- | :--- | :--- |
| Native Rust workspace | 508 tests passed, plus 2 documentation tests | Tested software behavior |
| CPU experiment orchestration | Median 238.42 ms → 114.58 ms with a reused Node process | 51.94% less orchestration time on a tiny fixture |
| New model-quality evaluation | 192 tool tasks + 64 arithmetic tasks frozen | A prepared experiment; GPU results are pending |

The orchestration benchmark uses synthetic scores and an 88-byte checkpoint. It does not establish faster large-model inference or better answers. [Benchmark method](docs/benchmarks/metaharness-cpu.md) · [Fresh evaluation status](experiments/model-improvement/STATUS.json)

<details>
<summary><strong>More results: the local pilot, the failed weight edit, and historical costs</strong></summary>

The Apple M4 Pro pilot used MLX 4-bit inference with and without the selected 4B adapter. These results are separate from the GPU BF16 measurements above.

| Local evaluation | Base | With adapter |
| :--- | ---: | ---: |
| Synthetic release | 29 / 35 | 32 / 35 |
| Tool subset of that release | 25 / 27 | 27 / 27 |
| Code subset of that release | 4 / 8 | 5 / 8 |
| HumanEval | 123 / 164 | 128 / 164 |
| HumanEval elapsed time | 547.5 s | 607.4 s |

HumanEval had 13 gains and 8 regressions, paired p = 0.383. The tool and code subsets overlap the 35-case release. A single Claude Code Read/Edit/Bash repair fixture passed four assertions in 20.8 seconds; that is not a repository-wide coding evaluation. [Local evidence](benchmarks/README.md)

The separate 0.8B direct edit changed 130,188 BF16 values. In the local 16-question direct-mode test, delivered correct answers fell from 13 to 12, despite 20.40% fewer tokens and 20.27% less generation time. Its quality gate failed. In the GPU direct-mode comparison, code fell from 50/164 to 49/164, tools from 15/27 to 11/27, and math from 114/128 to 113/128. Fewer tokens did not establish better answers.

The 0.8B thinking configuration improved from 0/128 to 29/128 delivered math answers, but the unedited direct configuration already delivered 114/128. This supports a narrow completion repair, not stronger general reasoning. The edit reproduces a [published intervention](https://rikiyat.github.io/blog/weight-space-programming/); it is not a newly discovered repair.

The historical Vast comparison had a $3.75 elapsed GPU/storage quote estimate and a $30 reservation. These are not final invoices or price guarantees; transfer and guard charges remain unreconciled. The historical rental and temporary guard resources were verified absent. [Closeout receipt](research/vast_cross_model/results/cross-model-20261103/closeout.json)

</details>

## Use it with Claude Code or Ruflo

The two optional plugins use MCP, the protocol that connects an assistant to local tools:

| Plugin | Use it for |
| :--- | :--- |
| [Native tools](plugins/ruflo-neural-evolution/README.md) | Inspect, patch, validate, export, restore and verify individual artifacts. |
| [MetaHarness experiments](plugins/ruflo-metaharness/README.md) | Run several bounded candidates, rank supplied evaluation scores, and advance a GPU job loop. |

The project-local installer builds the native engine and creates a private runtime profile and Claude launcher. You choose the allowed input directories, output directory and approved policy files. It leaves global Claude settings unchanged.

```sh
bash scripts/install-metaharness.sh --help
```

Follow the [installer examples](integrations/metaharness/README.md#configure-real-experiment-inputs), then launch Claude using the generated `launch-claude.sh`. The `/ruflo-metaharness:weight-experiment` command runs a reviewed plan. Ruflo and other MCP hosts can register the same local server.

**What MetaHarness adds:** Flywheel checks candidate qualification, Darwin ranks development candidates, and Weight-EFT exports filtered training data and training plans. Training plans are explicitly **not executed**. These tools organize experiments; they do not by themselves improve model weights.

For a CPU-only experiment after building the native binary:

```sh
(cd integrations/metaharness && npm ci --ignore-scripts)
mkdir -p runtime
node integrations/metaharness/src/cli.mjs fixture \
  --dir "$PWD/runtime/metaharness-fixture" \
  --binary "$PWD/native/target/release/weight_evolution"
```

Use a new fixture directory for another run. Scores in this demo are synthetic. [Experiment guide](integrations/metaharness/README.md)

## Measure a model on a GPU

The [new paired evaluation](experiments/model-improvement/README.md) compares the pinned 4B base model and included adapter on 256 frozen cases. It checks model files, adapter tensors, physical GPU identity, generation settings and complete outputs before scoring. Generated tools and programs are never executed.

A prepared GPU host can run both versions and score them with one command; see the guide for model preparation and `run-pair.py`. **No fresh results have been collected for this panel.** It evaluates the existing adapter, not a new adapter trained by MetaHarness.

The [GPU loop](integrations/metaharness/GPU-LOOP.md) adds restart-safe scheduling, budget reservations, deadlines and cleanup confirmation. A trusted host resolves ruOS secret references and performs Vast.ai operations. The included file handoff does not contact either service. The example has a zero budget, so submissions are disabled. Live provider execution and Claude `/loop` acceptance remain unverified.

## Serve the coding adapter on a Mac

This serving path requires **Apple Silicon**, Python 3.12+, and roughly 12 GB of download/model storage. The adapter is included; the base weights download separately.

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.lock
.venv/bin/python -m pip install -e .
.venv/bin/python -m ruflo_model.prepare
.venv/bin/python scripts/serve.py \
  --model models/base-4bit \
  --adapter weights/coding-4b/mlx
```

The local OpenAI-compatible API uses `http://127.0.0.1:9100/v1`, model name `ruflo-code-4b`, and the generated key in `runtime/server/api-key`. The server checks all 32 loaded adapter tensors. Keep this backend on a trusted single-user computer.

The tested profile handles text and tool messages, one request at a time, with a 16,384-token total context limit and at most 8,192 output tokens. Large repositories, longer conversations and concurrent serving are not validated. [Adapter details](weights/coding-4b/README.md) · [Coding protocol](docs/coding-protocol.md)

<details>
<summary><strong>Optional: configure Claude Code Router</strong></summary>

Configure an OpenAI Chat provider using the local API, model name and generated key above. For the recorded client profile, set `CLAUDE_CODE_MAX_OUTPUT_TOKENS=4096`, `MAX_THINKING_TOKENS=0`, and `CLAUDE_CODE_DISABLE_ADAPTIVE_THINKING=1`.

The historical fixtures used CCR 3.1.3 and Claude Code 2.1.316. That compatibility test does not establish official support for non-Claude models. Client tool permissions remain authoritative.

```sh
mkdir -p vendor/ccr
cp configs/ccr-package.json vendor/ccr/package.json
cp configs/ccr-package-lock.json vendor/ccr/package-lock.json
npm ci --prefix vendor/ccr
.venv/bin/python scripts/router_ccr.py start
.venv/bin/python scripts/accept_router.py --label read-check
.venv/bin/python scripts/accept_coding.py --label repair-check
```

</details>

## Find your way around

| Directory | What's inside |
| :--- | :--- |
| [native/](native/README.md) | Rust editor, locked dependencies and CPU demo |
| [weights/coding-4b/](weights/coding-4b/README.md) | Adapter files for PyTorch (PEFT) and Apple (MLX), with model version and checksums |
| [integrations/metaharness/](integrations/metaharness/README.md) | Experiment runner, data adapters and GPU loop |
| [experiments/model-improvement/](experiments/model-improvement/README.md) | New frozen panel, CUDA worker and paired scorer |
| [plugins/](plugins/ruflo-metaharness/README.md) | Claude Code and Ruflo integration guides |
| [benchmarks/](benchmarks/README.md) | Historical local results and verification |
| [research/vast_cross_model/](research/vast_cross_model/README.md) | Historical GPU comparison and full evidence |
| [docs/adr/](docs/adr/ADR-011-governed-neural-weight-evolution.md) | Architecture decisions, including experiment and measurement boundaries |

A **checkpoint** is a saved set of model weights. A **hash** is a file fingerprint used to detect changed bytes. An **RVF artifact** packages edits and evidence. None of these proves model quality or grants deployment permission on its own.

## Reproducibility and limits

Model files, prompts, policies and runtimes have recorded identities. New builds and experiments get new identities; historical receipts are not reused as proof of a new run. The native engine is extracted from RuVector with its original source and tests preserved. [Source provenance](native/upstream-provenance.json)

Verifiers establish consistency with the expected files. Obtain the repository commit or manifest digest from a trusted source: an archive replaced together with its verifier can still agree with itself. The full GPU release also requires its separate archive; the optional download steps appear above. [Publication manifest](research/vast_cross_model/results/cross-model-20261103/publication.json)

Native restoration and software tests do not establish broad capability, safety certification, state-of-the-art performance or recursive self-improvement. The scorer for executing generated code requires macOS Seatbelt and refuses unsupported platforms. The native `evaluate` operation validates receipt contracts; it does not run inference. Core Memory authorization remains an external reference, not a production activation service included here.

## License and attribution

Python research and adapter material use [Apache 2.0](LICENSE). The native engine retains its [MIT license](native/LICENSE); RVF and upstream model components retain their own terms. HumanEval is MIT licensed. Published intervention data retain CC BY 4.0 attribution. Base model checkpoints are not included.

[Project notices](NOTICE) · [Native notices](native/THIRD_PARTY_NOTICES.md) · [Research attribution](research/vast_cross_model/THIRD_PARTY_NOTICES.md)
