# MetaHarness CPU experiment orchestration benchmark

The [recorded run](metaharness-cpu-20261103.json) compares two ways of executing the
same bounded native experiment: launch a fresh Node CLI for each experiment, or
reuse a Node process through the exported runner API. The Claude Code MCP server
uses the second arrangement and serializes its requests. Both arrangements execute
the same native checks; process reuse does not cache authorization decisions or
skip executable, policy, source, checkpoint, RVF or rollback validation.

## Workload and method

Each experiment starts from an 88-byte, four-value BF16 Safetensors checkpoint.
Two distinct candidate IDs use the same one-value edit and precomputed synthetic
evaluation. Repeated identical candidates make the orchestration workloads equal;
this is not evidence of proposal diversity or discovery of a better intervention.

The runner performs native inspection once, then patching, evaluation-receipt
validation, RVF export, RVF verification and exact rollback for each candidate,
followed by actual MetaHarness Darwin development selection over qualified
candidates. Selection output and its explicit lack of independent confirmation
are also asserted.
The benchmark asserts that both candidates qualify under the fixture gate, each
restored digest equals the source digest, the runner's byte comparison passed,
the original source bytes remain unchanged, and saved report bindings match the
current run. Runner and native checks are timed; benchmark-side assertions after
the runner returns are outside the timing interval.

There are two warmup experiments per mode and seven measured experiments per
mode. Measured mode order alternates to reduce systematic filesystem-cache order
effects. This executes 18 experiments and 36 candidate lifecycles in total,
including warmups. Samples are sequential; this does not measure concurrent
serving. Percentiles use nearest rank, so p95 is the largest of these seven
samples. The JSON stores all measured samples, aggregate throughput, environment,
source and executable hashes, dependency-lock hash, and per-sample evidence hashes.
Temporary checkpoints and RVF artifacts are removed after validation; the JSON is
a benchmark summary, not a retained native experiment archive.

## Recorded result

The run used Linux x86-64 and Node 24.22.1. Exact CPU/kernel information and code
bindings are included in the JSON. Timings are wall-clock elapsed time for the
complete two-candidate experiment.

| Execution mode | p50 | p95 | Mean | Candidates/second |
| --- | ---: | ---: | ---: | ---: |
| Fresh Node CLI process | 238.42 ms | 281.32 ms | 236.98 ms | 8.44 |
| Reused Node runner process | 114.58 ms | 176.34 ms | 117.16 ms | 17.07 |

In this sample, reusing Node reduced median orchestration time by 51.94%. The
practical optimization is to retain the MCP process or import `runExperiment`
once for a batch instead of spawning the Node CLI repeatedly. Native operations
still run in separate bounded processes with the same checks. This comparison is
between execution modes of the same implementation, not a claimed speedup over
an earlier software release.

The tiny checkpoint makes Node startup and module loading a substantial portion
of the fresh-process time. These numbers do not predict throughput for large
model checkpoints, model inference, GPU training, or networked evaluation. Warm
filesystem caches, shared-machine scheduling and seven samples limit precision.
All evaluation scores are synthetic contract fixtures; no model quality gain,
generated-code execution, training completion or production activation is
established by this benchmark.

## Reproduce

First build the pinned native binary and install the integration dependencies:

```sh
cd native
cargo build --locked --release --bin weight_evolution
cd ../integrations/metaharness
npm ci --ignore-scripts
node bench/run.mjs --iterations 7 --warmup 2 --candidates 2 \
  --output /absolute/new/path/metaharness-cpu.json
```

The output path must not exist. Without `--output`, JSON is printed on stdout.
`--binary` can select another trusted native build; its digest is recorded.
Iteration, warmup and candidate counts have bounded integer ranges. A failed
native operation, missing fixture qualification, mismatched persisted report or
inexact rollback fails the benchmark with a nonzero exit code. Reconcile failed
experiment artifacts before retrying an interrupted mutation.

The execution sandbox used during development restricted some child-process
integration tests. Use the platform's supported command-execution permission
route when required; retain network policy, filesystem restrictions and native
verification. There is no network request or model download in the benchmark.
