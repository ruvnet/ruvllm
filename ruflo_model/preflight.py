"""Check frozen benchmark files and run canonical reference solutions."""
import argparse
import gzip
import json
import platform

from .common import ROOT, digest, now, read_jsonl, write_json
from .scoring import score_code


def run(output):
    data = ROOT / "data"
    manifest = json.loads((data / "manifest.json").read_text())
    for name, expected in manifest["files"].items():
        if digest(data / name) != expected:
            raise ValueError(f"Frozen data changed: {name}")
    external = data / "external"
    manifest = json.loads((external / "manifest.json").read_text())
    original = external / "HumanEval.jsonl.gz"
    suite = external / "humaneval.jsonl"
    if digest(original) != manifest["original_sha256"] or digest(suite) != manifest["sha256"]:
        raise ValueError("External benchmark hash mismatch")
    with gzip.open(original, "rt") as stream:
        references = [json.loads(line) for line in stream if line.strip()]
    rows = read_jsonl(suite)
    if len(rows) != 164 or {r["id"] for r in rows} != {r["task_id"] for r in references}:
        raise ValueError("External task inventory mismatch")
    failures = []
    for row in references:
        passed, reason = score_code({"tests": row["test"] + "\ncheck(" + row["entry_point"] + ")"},
                                    row["prompt"] + row["canonical_solution"])
        if not passed:
            failures.append({"id": row["task_id"], "reason": reason})
    result = {"total": len(references), "passed": len(references) - len(failures), "failures": failures,
              "suite_sha256": digest(suite), "source_archive_sha256": digest(original),
              "completed_at": now(), "platform": platform.platform()}
    write_json(output, result)
    if failures:
        raise RuntimeError("Canonical reference preflight failed; do not evaluate the model yet")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(ROOT / "runtime/humaneval-reference-check.json"))
    print(json.dumps(run(parser.parse_args().output), indent=2))
