# Paired measurement of the included 4B adapter

This experiment compares the pinned Qwen3.5-4B base model with the repository's
included `weights/coding-4b` adapter on 192 tool tasks and 64 arithmetic tasks.
It measures whether that adapter improves delivered answers on this panel.
It does not isolate the effect of MetaHarness, retrain the adapter, or authorize
model activation. The protocol is described in [ADR 016](../../docs/adr/ADR-016-paired-model-improvement-measurement.md).

**Current status: measured, keep.** The v3 freeze ran once on a rented Vast.ai
RTX 3090 through the [Vast host adapter](../../docs/adr/ADR-017-vast-host-adapter-and-live-paired-run.md)
($0.27). The local re-score of the retrieved raw responses matched the remote
report byte for byte, and the rental was confirmed destroyed. Tool tasks: base
123/192, adapter 173/192 (50 gains, 0 regressions, exact McNemar p = 1.8e-15).
Arithmetic: 63/64 in both arms. The preregistered rule is satisfied
(`research_improvement_confirmed: true`, no reasons). This is a research
qualification on a synthetic panel, not production activation.
Raw outputs, both reports, remote logs, receipts, the
[failure report](results/qwen35-4b-adapter-confirmation-v3/failure-report.json)
and a [dashboard](results/qwen35-4b-adapter-confirmation-v3/dashboard.html) are in
[results/qwen35-4b-adapter-confirmation-v3](results/qwen35-4b-adapter-confirmation-v3/).
Regenerate the breakdown with `tools/eval-report`, which refuses to write if its
counts disagree with `report.json`.

The committed [v3 protocol](frozen/v3/protocol.json) and
[256 cases](frozen/v3/cases.json) were frozen before generation. Protocol SHA256:
`bff9827e1ec8d77bc036519c562a1fb7240f820e64e9fb721c4ddeb2f772d3bc`. The cases are
byte-identical to v1 and v2. Those freezes are retired without output: v1's
shard-name check rejected the real model files, and v2 loaded the FP32 adapter
rounded to BF16 (see [STATUS.json](STATUS.json)). Do not alter a bundle after
generation.

## Freeze the panel before inference

Run from the repository root, with a new absolute output directory:

```bash
mkdir -p /workspace/measurement-runs
python3 experiments/model-improvement/prepare.py \
  --out /workspace/measurement-runs/qwen35-4b-panel-v1
```

The bundle freezes the protocol, 256 cases and their expected answers. Do not
edit a frozen bundle or select tasks after inspecting model outputs. Use a new
run and disclose the change if the protocol needs revision. Tool prompts span
eight families with 24 cases each; arithmetic supplies a separate retained
capability anchor. Exact prompts are checked against the repository's existing
training, development, release and previous cross-model prompt sources. Shared
tool semantics remain: exact-text disjointness does not establish semantic
novelty, absence of pretraining contamination or general capability improvement.

## Prepare a GPU host

Use one CUDA-capable host for both arms, with sufficient memory to load the
entire 4B model in BF16. Freeze its environment before inference:

```bash
python3 -m pip install torch==2.11.0 transformers==5.12.1 \
  peft==0.21.2 safetensors==0.7.0
```

The worker checks these versions and CUDA availability. Choose a compatible
CUDA/PyTorch distribution on the host rather than changing the protocol to
accept an unmeasured environment. Host package installation and weight download
are preparation steps, outside the inference worker. For example, download the
public pinned snapshot using the host's approved network path:

```bash
python3 - <<'PY'
import json
from pathlib import Path
from huggingface_hub import snapshot_download
assets = json.loads(Path("experiments/model-improvement/model-assets.json").read_text())
snapshot_download(
    repo_id="Qwen/Qwen3.5-4B",
    revision="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
    local_dir="/workspace/models/Qwen3.5-4B-851bf6e",
    token=False,
    allow_patterns=[*assets["asset_files"], "model*.safetensors"],
)
PY
```

Transfer the frozen bundle, the repository source and `weights/coding-4b` to
the host. Keep provider credentials at the authenticated host; bundles and
measurement artifacts contain no secret references that need resolution by
model code. The worker uses local files, disables remote model code and checks
model shard and adapter hashes before inference.

## Run both arms and score their saved outputs

After preparing the GPU host, the paired run also has a single host command:

```bash
python3 experiments/model-improvement/run-pair.py \
  --bundle /workspace/ruvllm/experiments/model-improvement/frozen/v3 \
  --model-dir /workspace/models/Qwen3.5-4B-851bf6e \
  --adapter-dir /workspace/ruvllm/weights/coding-4b \
  --out /workspace/measurement-runs/paired-v3 --deadline-seconds 5400
```

It launches fresh sequential processes for both arms, stops on failure, and runs
the scorer only after both arms succeed. It provisions no GPU or credentials.
The GPU loop's trusted host adapter can invoke this command on its allocated
worker. The host must inspect the actual report and executor response before
acknowledging completion; completing a measurement is separate from finding a
positive improvement.

Use distinct, nonexistent absolute output directories. Run the arms sequentially
on the same idle host with unchanged software, device and generation settings:

```bash
python3 experiments/model-improvement/worker.py \
  --bundle /workspace/measurement-runs/qwen35-4b-panel-v1 \
  --model-dir /workspace/models/Qwen3.5-4B-851bf6e \
  --adapter-dir /workspace/ruvllm/weights/coding-4b \
  --out /workspace/measurement-runs/qwen35-4b-base-v1 \
  --arm base --deadline-seconds 1800

python3 experiments/model-improvement/worker.py \
  --bundle /workspace/measurement-runs/qwen35-4b-panel-v1 \
  --model-dir /workspace/models/Qwen3.5-4B-851bf6e \
  --adapter-dir /workspace/ruvllm/weights/coding-4b \
  --out /workspace/measurement-runs/qwen35-4b-adapter-v1 \
  --arm adapter --deadline-seconds 1800

python3 experiments/model-improvement/measurement.py \
  --bundle /workspace/measurement-runs/qwen35-4b-panel-v1 \
  --base-run /workspace/measurement-runs/qwen35-4b-base-v1 \
  --adapter-run /workspace/measurement-runs/qwen35-4b-adapter-v1 \
  --out /workspace/measurement-runs/qwen35-4b-report-v1
```

Both arms load the same BF16 backbone with the same `PeftModel` wrapper and
verify the same 32 adapter tensors at their trained precision.
The base arm uses `disable_adapter()`; the adapter arm enables the adapter.
Greedy decoding, disabled thinking, seed 20261103 and a 512-new-token limit are
identical. An incomplete deadline-limited run cannot produce a qualifying
report. Run a new pair if it must be repeated; never combine selected outputs
from different runs.

Each arm saves `runtime.json`, `responses.jsonl` and a completion record emitted
only after all cases finish. The scorer validates identities, hashes, coverage,
runtime bindings and complete output before writing `report.json`. Preserve
the bundle, full raw responses, both runtime/completion records and the report
together. The report is the source for numerical claims; stdout from a model
or a provider status message is not evidence of improvement.

The primary result is delivered exact tool-call accuracy over 192 paired cases.
Only the expected call and exact typed arguments pass. The 64 arithmetic cases
require the expected boxed integer. Missing, malformed, truncated or incorrect
answers count as incorrect. Model-produced programs and tool calls are scored
as data and are never executed. The report includes paired wins/losses, exact
two-sided McNemar p-values and retained-capability/completion checks.

Research qualification requires a positive tool accuracy difference, primary
McNemar p < 0.05, no arithmetic accuracy regression and no completion decrease.
A positive point estimate alone is insufficient. The small, templated panel and
single checkpoint/seed do not support broad model-quality or production claims.
Latency is descriptive: one sequential pair does not establish a speedup.

## Use with the scheduled GPU loop

The existing [GPU loop](../../integrations/metaharness/GPU-LOOP.md) can hand work
to an operator-configured, pinned host adapter. This experiment does not add a
ruOS or Vast.ai API and does not turn the filesystem spool into a GPU executor.
The trusted host must prepare local files, run both worker arms, invoke the
scorer and retain verified artifacts before deriving a successful structured
completion for the scheduler. An `experiment.configPath`/`planPath` in an
existing loop configuration retains its existing native-experiment semantics;
it is not a new worker command interface.

Keep ruOS secret resolution, provider spending enforcement and allocation
cleanup at that host, under the existing job/run/idempotency bindings. Do not
convert agent-written JSON or mere process submission into verified completion.
No paid job, recurring schedule or provider allocation is enabled by these
commands or the local contract tests. Qualification of this research panel
does not bypass the existing external activation gates.
