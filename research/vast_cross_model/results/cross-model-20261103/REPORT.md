# Frozen cloud model comparison

Protocol `cf83995f29b5c3f7c28871d87d1295839bd94452dec0b8c352b990ba52a26cb6`. 1851 generations. No candidate selection or promotion.

| Arm | Panel | Delivered correct | Completed | Truncated | Output tokens |
| :--- | :--- | ---: | ---: | ---: | ---: |
| qwen035-08b-base-direct | humaneval164 | 50/164 | 155 | 9 | 53375 |
| qwen035-08b-base-direct | tools27 | 15/27 | 27 | 0 | 1274 |
| qwen035-08b-base-direct | math128 | 114/128 | 123 | 5 | 108970 |
| qwen035-08b-edited-direct | humaneval164 | 49/164 | 154 | 10 | 52992 |
| qwen035-08b-edited-direct | tools27 | 11/27 | 27 | 0 | 1281 |
| qwen035-08b-edited-direct | math128 | 113/128 | 127 | 1 | 88646 |
| qwen035-4b-base-direct | humaneval164 | 133/164 | 161 | 3 | 42222 |
| qwen035-4b-base-direct | tools27 | 24/27 | 27 | 0 | 1446 |
| qwen035-4b-base-direct | math128 | 127/128 | 128 | 0 | 64579 |
| qwen035-4b-lora-direct | humaneval164 | 135/164 | 161 | 3 | 41430 |
| qwen035-4b-lora-direct | tools27 | 27/27 | 27 | 0 | 1381 |
| qwen035-4b-lora-direct | math128 | 127/128 | 128 | 0 | 63906 |
| qwen3-coder-30b-direct | humaneval164 | 151/164 | 162 | 2 | 38084 |
| qwen3-coder-30b-direct | tools27 | 12/27 | 27 | 0 | 1648 |
| qwen3-coder-30b-direct | math128 | 127/128 | 128 | 0 | 38957 |
| qwen035-08b-base-thinking | math128 | 0/128 | 0 | 128 | 1048576 |
| qwen035-08b-edited-thinking | math128 | 29/128 | 29 | 99 | 903435 |

## Limits

HumanEval pretraining contamination unknown; repeated public benchmark
27 tool fixtures are synthetic and too small for production qualification
math corpus is original procedural arithmetic, not MATH500 or advanced reasoning
greedy fixed chat protocol is not an official leaderboard score
one GPU and seed; batch throughput is not individual request latency
prior local 16 case weight repair gate failed; cloud run is diagnostic replication
