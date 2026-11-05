<p align="center"><img src="docs/assets/ruvllm-hero.svg" alt="RuVLLM preserves the original model, changes an approved copy, and compares both on the same tasks." width="100%"></p>

# RuVLLM

**Change a model. Prove it. Keep the original.**

RuVLLM trains and edits small adapters for open language models, then measures each one against the unchanged base on the same tasks, under a rule written before the results. Every decision to keep or reject an adapter is backed by every raw answer, a paired significance test and a local re-score. The original weights restore byte for byte.

[**Interactive tour and tutorial →**](https://ruvnet.github.io/ruvllm/) · [npm](https://www.npmjs.com/package/@ruvector/ruvllm-gpu-loop) · [Results](#results) · [Compare](#how-it-compares) · [Quick start](#quick-start)

| | |
| :--- | :--- |
| **+50 tool passes, 0 regressions** | `coding-4b` on Qwen3.5 4B: 123 → 173 of 192, exact McNemar p = 1.8 × 10⁻¹⁵ |
| **Confirmed on untouched tasks** | Fresh held-out panel: 149 → 176 of 192, arithmetic 62 → 64 |
| **Under $1 per paired evaluation** | 256 tasks × 2 arms on a rented RTX 3090/4090: $0.27–$0.91 |
| **Reproducible to the byte** | Two independent base runs: 256/256 identical outputs; local re-score matches remote |

## Why RuVLLM

| Benefit | How it works |
| :--- | :--- |
| 🔒 **Rules locked first** | Tasks, expected answers, generation settings and the keep rule are frozen into a SHA-256-pinned bundle before any model output exists. |
| ⚖️ **Paired, per-task statistics** | Base and candidate answer identical tasks on the same GPU. Gains and losses are tested with an exact McNemar test, not compared as two averages. |
| 🔍 **Regressions stay visible** | Results break down by tool family and by case, with a failure taxonomy, so a gain in one place can't hide a loss in another. |
| ♻️ **Byte-exact restore** | The Rust engine edits a copy inside an explicit policy and proves restoration with matching source and restored hashes. |
| 💸 **Cheap, capped GPU runs** | Four spending limits, an independent destroy timer and confirmed cleanup on every rental. |
| 🧰 **Works where you work** | npm CLI, Claude Code plugins over MCP, a Ruflo-compatible server and an OpenAI-compatible local API on Apple Silicon. |

## Results

<p align="center"><img src="docs/assets/heldout-results.svg" alt="Held-out results on 256 untouched tasks: tool passes 149, 176 and 192 of 192 for base, coding-4b and coding-4b-r2; arithmetic 62, 64 and 46 of 64. coding-4b is kept; coding-4b-r2 is rejected." width="100%"></p>

| Run (Qwen3.5 4B, BF16) | Tool passes | Gains / losses | Arithmetic | Paired p | Decision | GPU cost |
| :--- | ---: | ---: | ---: | ---: | :--- | ---: |
| Confirmation · `coding-4b` | 123 → 173 / 192 | 50 / 0 | 63 → 63 | 1.8e-15 | ✅ Keep | $0.27 |
| Held out · `coding-4b` | 149 → 176 / 192 | 40 / 13 | 62 → 64 | 2.7e-4 | ✅ Keep | $0.91 |
| Held out · `coding-4b-r2` | 149 → 192 / 192 | 43 / 0 | 62 → 46 | — | ❌ Reject | $0.76 |

The second training round (`coding-4b-r2`, $0.09 to train) fixed every tool failure but lost 16 arithmetic answers. The rule frozen before training says capability anchors must not regress, so it was rejected without debate. That's the workflow doing its job.

<p align="center"><img src="docs/assets/tool-families.svg" alt="Tool passes per family out of 24 for base, coding-4b and coding-4b-r2: grep 0, 24, 24; test 8, 24, 24; write 21, 8, 24; read, edit, glob, slice and injection 24 for every arm." width="100%"></p>

Raw responses, both reports, receipts, failure reports and dashboards for every run are in [experiments/model-improvement/results](experiments/model-improvement/results/). [Measurement status](experiments/model-improvement/STATUS.json) · [Protocol (ADR 016)](docs/adr/ADR-016-paired-model-improvement-measurement.md)

## How it works

<p align="center"><img src="docs/assets/governed-lifecycle.svg" alt="Freeze the model, tasks and limits; edit a copy; compare answers; verify exact restoration. Deployment requires separate approval." width="100%"></p>

1. **Freeze.** Pin the model revision, the task panel, the generation settings and the keep rule.
2. **Change a copy.** Train a LoRA adapter or apply a bounded direct weight edit. The original file is never modified.
3. **Measure in pairs.** Run base and candidate on identical tasks, keep every raw answer and score them with the frozen scorer.
4. **Decide and restore.** Apply the preregistered rule, then verify the original restores byte for byte.

| Two ways to change a model | What it is | Included |
| :--- | :--- | :--- |
| **LoRA adapter** | A small add-on trained while the base stays frozen | `coding-4b`: 1.5M parameters for Qwen3.5 4B, in PEFT and MLX formats |
| **Direct weight edit** | Policy-bounded changes to specific values in a copy | Rust editor with validation, RVF evidence and exact restoration |

### Rent GPUs without surprises

<p align="center"><img src="docs/assets/gpu-job-loop.svg" alt="The GPU job loop checks a spending ceiling, rents the offer and arms a destroy timer, runs the pinned job, pulls raw outputs, checks them locally, then destroys the rental and confirms it is gone." width="100%"></p>

| Limit | What it stops |
| :--- | :--- |
| Admission budget | Total spend across the whole job queue |
| Worst-case offer price | Any offer whose hourly price × timeout exceeds the job's ceiling |
| Accrued cost | A running job whose accrued cost passes its ceiling |
| Independent destroy timer | A rental outliving its deadline, even if the control host goes away |

The loop is restart-safe: jobs carry idempotent labels, state is durable, and a job only counts as complete when the local re-score matches. Drive it from the CLI, from Claude Code `/loop`, or from any MCP host. [GPU loop guide](integrations/metaharness/GPU-LOOP.md) · [Vast adapter (ADR 017)](docs/adr/ADR-017-vast-host-adapter-and-live-paired-run.md)

## How it compares

| Question | Typical ad-hoc fine-tune loop | RuVLLM |
| :--- | :--- | :--- |
| When is the success rule set? | Often after looking at scores | Before generation, SHA-pinned |
| How are models compared? | Aggregate accuracy from separate runs | Same tasks, same GPU, per-task exact McNemar |
| Are regressions visible? | Averaged into one number | Per family and per case, with a failure taxonomy |
| Held-out tasks? | Often reused from development | Fresh panel, checked disjoint from training data |
| Can someone else re-check? | Usually summary numbers only | Raw outputs plus frozen scorer, re-scored locally |
| What happens to the original? | Overwritten or untracked | Untouched; byte-exact restore verified |
| GPU spend control? | Manual, easy to leave running | Four limits, destroy timer, confirmed cleanup |
| Can a shiny result be rejected? | Rarely in practice | Yes: 192/192 tools rejected for an arithmetic loss |

## Quick start

The [interactive tutorial](https://ruvnet.github.io/ruvllm/#tutorial) walks through these steps with expected output.

| Step | Command | Needs |
| :--- | :--- | :--- |
| Install the CLI | `npm install @ruvector/ruvllm-gpu-loop` then `npx ruvllm-gpu --help` | Node 22+ |
| Verify the evidence | `python3 scripts/check_repository.py` and `python3 scripts/verify_benchmarks.py` | Python 3.12+ |
| Edit and restore weights | `cd native && cargo build --locked --release --bin weight_evolution && python3 smoke.py` | Rust 1.97.1 |
| Re-score a real GPU run | `regression-dashboard.py` on the held-out results (below) | Python 3.12+ |
| Schedule GPU work | `npx ruvllm-gpu-loop status --config /abs/private/gpu-loop.json` | A Vast.ai key on the control host |

```sh
git clone https://github.com/ruvnet/ruvllm.git && cd ruvllm
python3 scripts/check_repository.py          # look for "verified": true

mkdir -p runtime
R=experiments/model-improvement
python3 $R/regression-dashboard.py \
  --bundle $R/frozen/v4 \
  --base $R/results/qwen35-4b-heldout-v4-coding-4b/base \
  --adapter $R/results/qwen35-4b-heldout-v4-coding-4b/adapter \
  --out runtime/v4-dashboard.md             # Keep for research · 149 → 176
```

### Run bounded experiments from Claude Code or Ruflo

| Plugin | Use it for |
| :--- | :--- |
| [Native tools](plugins/ruflo-neural-evolution/README.md) | Inspect, patch, validate, export, restore and verify individual weight files |
| [MetaHarness experiments](plugins/ruflo-metaharness/README.md) | Run several bounded candidates, rank them, and advance the GPU job loop |

```sh
bash scripts/install-metaharness.sh --help
```

The installer builds the native engine and creates a private runtime profile and Claude launcher, scoped to the input and output directories you choose. Then `/ruflo-metaharness:weight-experiment` runs a reviewed plan. [Experiment guide](integrations/metaharness/README.md)

### Serve the coding adapter on a Mac

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.lock
.venv/bin/python -m pip install -e .
.venv/bin/python -m ruflo_model.prepare
.venv/bin/python scripts/serve.py --model models/base-4bit --adapter weights/coding-4b/mlx
```

This gives you an OpenAI-compatible API at `http://127.0.0.1:9100/v1` (model `ruflo-code-4b`, key in `runtime/server/api-key`) on Apple Silicon, with all 32 adapter tensors checked at load. It works with Claude Code Router as a local single-user backend. [Adapter details](weights/coding-4b/README.md) · [Coding protocol](docs/coding-protocol.md)

## What's in the box

| Directory | What's inside |
| :--- | :--- |
| [native/](native/README.md) | Rust weight editor, locked dependencies and CPU demo |
| [weights/coding-4b/](weights/coding-4b/README.md) | The kept adapter for PyTorch (PEFT) and Apple (MLX), with checksums |
| [experiments/model-improvement/](experiments/model-improvement/README.md) | Frozen panels, CUDA worker, paired scorer and every GPU result |
| [tools/eval-report/](tools/eval-report/) | Rust failure taxonomy, three-way comparison, SFT pool and panel freezing |
| [integrations/metaharness/](integrations/metaharness/README.md) | npm package: experiment runner, GPU loop and Vast.ai host adapter |
| [plugins/](plugins/ruflo-metaharness/README.md) | Claude Code and Ruflo integrations |
| [research/vast_cross_model/](research/vast_cross_model/README.md) | Earlier cross-model GPU study with its full evidence archive |
| [docs/adr/](docs/adr/ADR-011-governed-neural-weight-evolution.md) | Architecture decisions |

## Scope

RuVLLM qualifies adapters for research on the panels it measures; production activation is a separate, explicit step. Results are reported for Qwen3.5 4B on tool-use and arithmetic panels. Earlier exploratory studies, including a direct-edit experiment on Qwen3.5 0.8B that the quality gate rejected, are documented with their evidence in [benchmarks/](benchmarks/README.md) and [research/vast_cross_model/](research/vast_cross_model/results/cross-model-20261103/README.md).

## License and attribution

Python research and adapter material use [Apache 2.0](LICENSE). The native engine retains its [MIT license](native/LICENSE); RVF and upstream model components retain their own terms. HumanEval is MIT licensed. Published intervention data retain CC BY 4.0 attribution. Base model checkpoints are not included.

[Project notices](NOTICE) · [Native notices](native/THIRD_PARTY_NOTICES.md) · [Research attribution](research/vast_cross_model/THIRD_PARTY_NOTICES.md)
