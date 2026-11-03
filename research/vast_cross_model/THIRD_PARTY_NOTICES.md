# Third party notices

These notices identify the benchmark inputs and published weight edit used by
this experiment. They do not assign one license to the combined model artifacts,
runtime dependencies, or compiled binaries.

## HumanEval

The 164 HumanEval cases and original tests come from
[OpenAI HumanEval](https://github.com/openai/human-eval/tree/b9453e8f1865df0a0057a1383767caa400acfbcf),
revision `b9453e8f1865df0a0057a1383767caa400acfbcf`.
Copyright (c) OpenAI. The source is MIT licensed. Its exact license text is
included in [licenses/HumanEval.LICENSE](licenses/HumanEval.LICENSE).
The complete function chat prompts and local scoring procedure are project
adaptations; this experiment does not use the official leaderboard protocol.

## Published neural weight edit

The Qwen3.5 0.8B sparse BF16 edit is attributed to Rikiya Takehi, Jayden Teoh,
Alex Zhang, and Omar Khattab, authors of
[LLMs can understand and write neural network weights](https://rikiyat.github.io/blog/weight-space-programming/).
The input is the published p50c intervention in
[experiment logs at revision 4d09ba6d6a7e891dd8063974dfe81b3d78e3494f](https://github.com/RikiyaT/lm_understand-logs/tree/4d09ba6d6a7e891dd8063974dfe81b3d78e3494f/exp8_math).

The authors' notice assigns CC BY 4.0 to their logs, data, and networks, and MIT
to their scripts. The exact upstream notice, including its separate third party
content qualifications, is included in
[licenses/LICENSE-research.txt](licenses/LICENSE-research.txt).
The cloud experiment applies the selected alpha 1.0 sparse change. Conversion
to the native patch format and the experiment harness are local work. The
earlier development compared alpha 0.5 and 1.0. This public export includes the
selected alpha 1.0 release results in the [local benchmark evidence](../../benchmarks/README.md).

## Qwen checkpoints

The following official model cards declare Apache 2.0 licensing. The immutable
revisions and published Safetensors hashes are recorded in
[models.json](models.json).

| Checkpoint | Revision | Official model card |
| :--- | :--- | :--- |
| Qwen3.5 0.8B | `2fc06364715b967f1860aea9cf38778875588b17` | [Qwen/Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B/blob/2fc06364715b967f1860aea9cf38778875588b17/README.md) |
| Qwen3.5 4B | `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` | [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B/blob/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a/README.md) |
| Qwen3 Coder 30B A3B Instruct | `b2cff646eb4bb1d68355c01b18ae02e7cf42d120` | [Qwen/Qwen3-Coder-30B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct/blob/b2cff646eb4bb1d68355c01b18ae02e7cf42d120/README.md) |

The 0.8B native patch and the 4B LoRA merge are separate derived artifacts.
The 30B coding reference remains unedited. Keep upstream license files with
redistributed checkpoints.
