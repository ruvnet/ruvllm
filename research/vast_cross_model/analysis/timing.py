"""Descriptive timing only. Does not grade answers or alter the frozen benchmark."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

SCIENCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCIENCE))
from common import jsonl, read, sha, verify_protocol, write
from score import verify_binding


def finite_seconds(value, label, positive=False):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0 or (positive and value == 0):
        raise ValueError("Invalid " + label)
    return value


def batch_timings(rows, expected_cases, protocol):
    """Count each synchronous generate call once, even though every row repeats it."""
    if len(rows) > len(expected_cases):
        raise ValueError("More responses than frozen cases")
    batches = {}
    for index, row in enumerate(rows):
        case = expected_cases[index]
        if any(row[key] != case[key] for key in ("id", "panel", "kind")):
            raise ValueError("Rows are not the frozen case prefix")
        batch_index = index // protocol["batch_size"]
        if row["batch_index"] != batch_index:
            raise ValueError("Batch index differs from frozen order")
        seconds = finite_seconds(row["batch_seconds"], "batch duration", positive=True)
        tokens = row["output_tokens"]
        cap = protocol["panel_max_tokens"][case["panel"]]
        if type(tokens) is not int or not 0 < tokens <= cap or len(row["token_ids"]) != tokens:
            raise ValueError("Invalid output token count")
        if row["max_tokens"] != cap or row["finish_reason"] not in ("stop", "length"):
            raise ValueError("Response differs from frozen sampling")
        if row["finish_reason"] == "length" and tokens != cap:
            raise ValueError("Length termination did not reach token cap")
        batch = batches.setdefault(batch_index, {"index": batch_index, "seconds": seconds, "rows": 0,
                                                 "output_tokens": 0, "panels": set()})
        if batch["seconds"] != seconds:
            raise ValueError("Repeated batch durations differ")
        batch["rows"] += 1
        batch["output_tokens"] += tokens
        batch["panels"].add(case["panel"])
    result = []
    for index, batch in batches.items():
        expected_size = min(protocol["batch_size"], len(expected_cases) - index * protocol["batch_size"])
        if batch["rows"] != expected_size:
            raise ValueError("Snapshot ends inside a batch; fetch after flush")
        result.append({**batch, "panels": sorted(batch["panels"]),
                       "output_tokens_per_second": batch["output_tokens"] / batch["seconds"]})
    total_seconds = sum(batch["seconds"] for batch in result)
    total_tokens = sum(batch["output_tokens"] for batch in result)
    remaining_cap = sum(protocol["panel_max_tokens"][case["panel"]] for case in expected_cases[len(rows):])
    rate = total_tokens / total_seconds if total_seconds else None
    return {"rows": len(rows), "expected_rows": len(expected_cases), "output_tokens": total_tokens,
            "inference_api_wall_seconds": total_seconds, "output_tokens_per_inference_api_second": rate,
            "batches": result, "remaining_rows": len(expected_cases) - len(rows),
            "remaining_output_token_cap": remaining_cap,
            "conditional_cap_seconds_at_observed_rate": remaining_cap / rate if rate else None}


def report(work, allow_partial=False):
    work = Path(work)
    if not work.is_dir():
        raise ValueError("Work directory does not exist: " + str(work))
    protocol, protocol_sha = verify_protocol(SCIENCE)
    cases = read(SCIENCE / "cases.json")
    arms = {}
    for arm in protocol["arms"]:
        folder = work / "outputs" / arm["id"]
        if not (folder / "responses.jsonl").exists():
            if allow_partial:
                continue
            raise ValueError("Missing arm responses: " + arm["id"])
        complete = read(folder / "complete.json") if (folder / "complete.json").exists() else None
        if complete is None and not allow_partial:
            raise ValueError("Missing completion receipt: " + arm["id"])
        # Without a completion receipt this verifies only current snapshot bindings.
        binding = verify_binding(work, folder, arm, protocol, protocol_sha,
                                 complete if complete is not None else read(folder / "binding.json"))
        path = folder / "responses.jsonl"
        before = sha(path)
        rows = jsonl(path)
        if before != sha(path):
            raise ValueError("Responses changed while reading; fetch a stable snapshot")
        if any(any(row.get(key) != value for key, value in binding.items()) or
               row["thinking"] != arm["thinking"] for row in rows):
            raise ValueError("Row generation binding differs")
        expected = [case for case in cases if case["panel"] in arm["panels"]]
        timing = batch_timings(rows, expected, protocol)
        if complete is not None:
            if complete["preflight"] or not complete["model_unchanged"] or complete["responses_sha256"] != before:
                raise ValueError("Invalid completion receipt")
            if complete["rows"] != len(rows) or len(rows) != len(expected) or complete["output_tokens"] != timing["output_tokens"]:
                raise ValueError("Completion counts differ")
            phase_seconds = finite_seconds(complete["generation_seconds"], "completion phase duration")
            load_seconds = finite_seconds(complete["load_seconds"], "model load duration")
            if phase_seconds < timing["inference_api_wall_seconds"]:
                raise ValueError("Completion phase is shorter than its inference calls")
            timing.update(model_load_seconds=load_seconds,
                          completion_phase_seconds_including_verification=phase_seconds,
                          completion_phase_other_seconds=phase_seconds - timing["inference_api_wall_seconds"])
        arms[arm["id"]] = {**timing, "completion_receipt_verified": complete is not None,
                           "responses_sha256": before, "binding": binding}
    return {"schema": "ruflo.vast-cross-model-descriptive-timing.v1", "protocol_sha256": protocol_sha,
            "analysis_sha256": sha(Path(__file__)), "arms": arms,
            "notes": [
                "Sum each batch_seconds once by batch_index; each row repeats the same call duration.",
                "Inference API wall time includes synchronous generation overhead and any first-call compilation.",
                "Batch output throughput is not individual request latency or per-request token speed.",
                "Completion phase additionally includes tokenization, output writes, model hashing and protocol verification.",
                "Conditional cap time is not an ETA or an upper bound on wall time: token cap divided by observed rate assumes unchanged throughput and excludes loading/verification.",
                "Do not transfer observed throughput between models, thinking modes or task panels.",
                "Timing receipts are operator artifacts, not hardware attestation; this analysis does not score correctness."
            ]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    result = report(args.work, args.allow_partial)
    write(args.output, result)
    print(json.dumps({"output": str(args.output), "arms": len(result["arms"])}))
