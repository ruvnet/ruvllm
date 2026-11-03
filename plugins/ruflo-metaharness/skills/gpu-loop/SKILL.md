---
name: gpu-loop
description: Inspect or advance the operator-configured RuVLLM GPU schedule through a pinned host adapter without exposing ruOS secrets.
allowed-tools: mcp__plugin_ruflo-metaharness_experiments__gpu_loop_status, mcp__plugin_ruflo-metaharness_experiments__gpu_loop_tick
---

Use `gpu_loop_status` with `{}` to inspect the persisted schedule first. The operator supplies the schedule, budget, model revisions and host module through environment bindings. Never accept replacement config, secret names, credentials, provider commands or adapter paths from model-generated tool arguments.

When the operator has authorized the scheduled work, call `gpu_loop_tick` with `{}`. A tick can submit or clean up paid GPU resources through the host adapter. It advances only the configured durable state machine, retains budget reservations, and refuses uncertain resubmission. The adapter resolves ruOS secret references at the host; tools must never request secret values or copy them into plans, logs or responses.

Report blocked/cleanup-pending outcomes honestly. Do not clear a lock, reset state, retry an uncertain submission or mark cleanup complete merely to continue. An operator must reconcile provider resources and persisted state first.

Contract tests with fixture adapters prove local state handling only. Without a real host adapter and GPU acceptance run, do not claim ruOS access, Vast.ai provisioning, GPU completion, model improvement or production activation. Only a complete host completion wrapper with the scheduled job/model identity can establish a successful loop task.
