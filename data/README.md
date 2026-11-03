# Data provenance

The 288 training and 36 validation examples were generated from original templates in `ruflo_model/data.py`. They cover Read, Edit, Write, Glob, Grep, and Bash tool schemas, exact replacements, tool results, and a small repository prompt injection fixture. They contain no private source code, user conversations, or commercial teacher model output.

Development and release each contain 27 tool cases and 8 executable Python tasks. Tool vocabulary, paths, and prompts differ by split, while tool semantics and some templates are shared. The code task families differ between development and release and are absent from training. These are narrow synthetic checks, not a broad software engineering benchmark.

`manifest.json` hashes each frozen split. Candidate selection uses development only. Release results may reject a candidate but never select another checkpoint in this experiment. A further optimization cycle requires a new untouched release set.

`external/humaneval.jsonl` contains 164 untouched public HumanEval tasks, derived from the pinned upstream revision in its manifest. Its original MIT license is included. The chat prompt, token budget, Python version, and sandbox differ from some published evaluations, so resulting scores are local results rather than leaderboard submissions. Base model pretraining contamination is unknown. Reference solutions are present in the original archive but are never sent to inference or training.
