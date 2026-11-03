# Descriptive timing analysis

This optional analysis reads a retrieved result snapshot. It does not change the frozen inference protocol, grade answers, select a model, or authorize deployment.

```sh
.venv/bin/python research/vast_cross_model/analysis/timing.py \
  --work runtime/vast-cross-model-retrieved \
  --output runtime/vast-cross-model-timing.json
```

Use `--allow-partial` for a stable snapshot containing only complete batches. A missing completion receipt is then explicitly reported. All present row bindings, runtime and model manifests must still match the frozen protocol. A completed arm must also match its completion receipt and raw output digest. These are operator artifacts, not hardware attestation.

Every response in a batch repeats the same `batch_seconds`. The helper counts that duration once per batch, then divides total output tokens by the sum of those durations. This is aggregate inference API throughput, not individual request latency. It includes synchronous generation overhead and any first call compilation.

The worker's `complete.generation_seconds` covers a broader completion phase that includes final model hashing and protocol verification. It appears separately as `completion_phase_seconds_including_verification`; subtracting inference call time gives the other completion phase time, which also includes tokenization and output writes.

A partial snapshot includes remaining case count, remaining output token cap and a conditional cap time at the observed aggregate rate. That value is neither an ETA nor a guaranteed upper wall time bound. It excludes future loading and verification, and throughput can change substantially between models, thinking modes, and task panels. No cross model time forecast is produced.

Acceptance test:

```sh
.venv/bin/python -m unittest discover -s research/vast_cross_model/analysis -v
```

Eight rows sharing a two second batch and two rows sharing a one second batch must report three inference API seconds, not eighteen.
