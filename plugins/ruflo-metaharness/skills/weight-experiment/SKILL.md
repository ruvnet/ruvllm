---
name: weight-experiment
description: Run and review an operator-configured RuVLLM MetaHarness weight experiment with exact rollback and explicit evidence labels.
allowed-tools: mcp__plugin_ruflo-metaharness_experiments__experiment_run
---

Use the operator's frozen experiment plan. Call `experiment_run` with its absolute `planPath`; runtime configuration comes only from `RUVLLM_EXPERIMENT_CONFIG`.

Review the approved source, patch policies, candidate count, and evidence mode before running. The runner creates new candidate and restored files, validates native receipts, packages RVF evidence, and verifies original/restored byte identities. Candidate qualification is an experimental comparison, never deployment authority.

Distinguish synthetic fixture results from externally supplied evaluation claims. The tool does not perform inference, discover edits, train adapters, establish evaluator independence, or change a serving model. Missing evaluation must remain unqualified. Do not invent outcomes, identities, signatures, or activation permissions.

When a call fails or its response is lost, inspect the operator output directory and reconcile existing artifacts before retrying. Every invocation creates a new exclusive run directory; repeating a plan preserves prior artifacts but repeats its computational work. Never delete evidence to hide failure.
