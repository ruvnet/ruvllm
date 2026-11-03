---
description: Inspect and advance the operator-configured durable GPU loop schedule.
allowed-tools: mcp__plugin_ruflo-metaharness_experiments__gpu_loop_status, mcp__plugin_ruflo-metaharness_experiments__gpu_loop_tick
---

Use the `gpu-loop` skill. Read `gpu_loop_status` with `{}`. Advance `gpu_loop_tick` with `{}` only within the operator's authorized scheduled work and configured budget. Never supply arbitrary commands, configuration paths, adapters or secrets through tool arguments. Preserve uncertain submissions and cleanup-pending states for operator reconciliation. Distinguish local contract fixtures from live GPU evidence; no activation is authorized.
