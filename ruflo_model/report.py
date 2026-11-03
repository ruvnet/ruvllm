import json
import math
from pathlib import Path

from .common import ROOT, now, read_jsonl, write_json


def wilson(passed, total):
    z = 1.959963984540054
    p = passed / total
    center = (p + z*z/(2*total)) / (1+z*z/total)
    half = z * math.sqrt(p*(1-p)/total + z*z/(4*total*total)) / (1+z*z/total)
    return [round(center-half, 4), round(center+half, 4)]


def paired(baseline_file, candidate_file):
    a = {r["id"]: r for r in read_jsonl(baseline_file)}
    b = {r["id"]: r for r in read_jsonl(candidate_file)}
    if set(a) != set(b):
        raise ValueError("Paired comparison requires identical task IDs")
    wins = [k for k in a if not a[k]["passed"] and b[k]["passed"]]
    losses = [k for k in a if a[k]["passed"] and not b[k]["passed"]]
    n = len(wins) + len(losses)
    pvalue = min(1.0, 2 * sum(math.comb(n,k) for k in range(min(len(wins),len(losses))+1)) / (2**n)) if n else 1.0
    return {"improved_ids": wins, "regressed_ids": losses, "exact_mcnemar_p": pvalue,
            "note": "This paired test covers this suite only. Task correlation and pretraining contamination limit broader conclusions"}


def build():
    run = ROOT / "runtime/pilot-v2"
    selection = json.loads((run / "selection.json").read_text())
    state = json.loads((run / "state.json").read_text())
    rows = []
    paths = [("Qwen3.5 4B BF16", ROOT / "runtime/pilot-v1/baseline-dev", "M4 Pro, MLX"),
             ("Qwen3.5 4B 4 bit", run / "baseline-dev", "M4 Pro, MLX"),
             ("Qwen3 Coder 30B BF16", ROOT / "runtime/cloud", "A100 80GB, Transformers")]
    adapter = selection["selected_adapter"]
    if adapter:
        paths.append(("RuFlo Code 4B candidate", run / (Path(adapter).name + "-dev"), "M4 Pro, MLX"))
    for name, directory, hardware in paths:
        summary = json.loads((directory / "summary.json").read_text())
        rows.append({"name": name, "hardware": hardware, **summary,
                     "wilson_95_accuracy": wilson(summary["passed"], summary["total"])})
    report = {"generated_at": now(), "development": rows, "selected_adapter": str(Path(adapter).relative_to(ROOT)) if adapter else None,
              "synthetic_release_gate": state["release_gate_passed"], "sota_verified": False,
              "claims": "Experimental specialist; synthetic pilot and local external validation only"}
    if adapter:
        report["development_paired"] = paired(run / "baseline-dev/responses.jsonl", run / (Path(adapter).name + "-dev") / "responses.jsonl")
        report["release_paired"] = paired(run / "baseline-release/responses.jsonl", run / "selected-release/responses.jsonl")
        report["synthetic_release"] = {name: json.loads((run / f"{name}-release/summary.json").read_text()) for name in ("baseline", "selected")}
        for row in report["synthetic_release"].values():
            row["wilson_95_accuracy"] = wilson(row["passed"], row["total"])
    ext = ROOT / "runtime/release-validation/report.json"
    if ext.exists():
        report["external"] = json.loads(ext.read_text())
        for name in ("baseline", "selected"):
            row = report["external"][name]
            row["wilson_95_accuracy"] = wilson(row["passed"], row["total"])
        if adapter:
            report["external_paired"] = paired(ext.parent / "baseline-humaneval/responses.jsonl", ext.parent / "selected-humaneval/responses.jsonl")
    integration = ROOT / "runtime/integration/selected-coding/receipt.json"
    if integration.exists():
        report["coding_integration"] = json.loads(integration.read_text())
    ledger = json.loads((ROOT / "runtime/cloud-budget.json").read_text())
    report["cloud"] = {"authorized_cap_usd": 200, "reserved_max_usd": sum(r["reserved_micro_usd"] for r in ledger["reservations"]) / 1e6,
                       "actual_invoice_usd": None}
    receipts = json.loads((ROOT / "results/cloud/receipts.json").read_text())
    report["cloud"]["estimated_compute_upper_usd"] = sum(j["estimated_compute_upper_usd"] for j in receipts)
    report["cloud"]["all_recorded_jobs_completed"] = all(j["stage"] == "COMPLETED" for j in receipts)
    # Do not publish workstation absolute paths in the report.
    report = json.loads(json.dumps(report).replace(str(ROOT), "."))
    write_json(ROOT / "results/report.json", report)
    lines = ["# Measured results", "", "SOTA is not established. These results describe the recorded experiment only.", "",
             "| Model | Runtime | Development pass count | Tool pass count | Code pass count | Output tokens | Seconds |",
             "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |"]
    for row in rows:
        g = row["groups"]
        lines.append(f"| {row['name']} | {row['hardware']} | {row['passed']}/{row['total']} | {g['tool']['passed']}/{g['tool']['total']} | {g['code']['passed']}/{g['code']['total']} | {row['output_tokens']} | {row['elapsed_seconds']:.1f} |")
    lines += ["", "A100 and Mac timings are not hardware matched. Tool scoring requires one valid call with the requested arguments and action scope. Code scoring executes separate tests in the sandbox.",
              "", "The development suite contains 35 synthetic cases and shared template families. Candidate selection used this suite. Release and external results are in report.json and were excluded from selection.",
              "", f"Estimated compute upper cost from elapsed job time: USD {report['cloud']['estimated_compute_upper_usd']:.2f}. Reserved maximum: USD {report['cloud']['reserved_max_usd']:.2f} of USD 200. Actual invoice not yet reconciled."]
    if adapter and "external" in report:
        lines += ["", "| Held out evaluation | Base | Candidate | Base seconds | Candidate seconds |",
                  "| :--- | :--- | :--- | :--- | :--- |"]
        for label, key in (("Synthetic release", "synthetic_release"), ("HumanEval local protocol", "external")):
            base, selected = report[key]["baseline"], report[key]["selected"]
            lines.append(f"| {label} | {base['passed']}/{base['total']} | {selected['passed']}/{selected['total']} | {base['elapsed_seconds']:.1f} | {selected['elapsed_seconds']:.1f} |")
        changes = report["external_paired"]
        conclusion = "not statistically conclusive" if changes["exact_mcnemar_p"] >= .05 else "significant under this paired test"
        lines += ["", f"HumanEval gained {len(changes['improved_ids'])} cases and lost {len(changes['regressed_ids'])}. The exact paired McNemar p value is {changes['exact_mcnemar_p']:.3f}. The aggregate increase is {conclusion} and does not mean every task improved."]
        if report.get("coding_integration", {}).get("passed"):
            lines += ["", f"The selected adapter completed an actual Claude Code Read, Edit, Bash fixture in {report['coding_integration']['seconds']:.1f} seconds. Four independent assertions passed. The first fixture run was denied because of an incorrectly scoped path rule; its evidence is retained. This is one integration fixture, not a repository benchmark.",
                      "", "The server explicitly binds the base and adapter and verifies all 32 loaded adapter tensors. Earlier HTTP integration results were invalidated after discovering that MLX LM 0.31.3 dropped its configured adapter during model lookup. The direct offline evaluations were unaffected. Invalidated packages and receipts are retained separately."]
    (ROOT / "results/REPORT.md").write_text("\n".join(lines) + "\n")
    return report


if __name__ == "__main__":
    print(json.dumps(build(), indent=2))
