---
description: Run a frozen, policy-bounded RuVLLM weight experiment and review its evidence.
argument-hint: /absolute/path/to/experiment-plan.json
allowed-tools: mcp__plugin_ruflo-metaharness_experiments__experiment_run
---

Run the `weight-experiment` skill with plan path `$ARGUMENTS`. Use only `experiment_run`; never substitute arbitrary shell commands or change runtime policy. Report candidate results, exact rollback verification, evidence mode, and unresolved evaluation limitations. No model activation is authorized.
