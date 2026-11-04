//! `three-way`: compare base, a reference adapter and a candidate adapter on one
//! held-out panel scored as two frozen pairs (reference bundle, candidate bundle)
//! with byte-identical cases. Each pair's verdict comes only from its own
//! `report.json`. The adapter-vs-adapter comparison is the preregistered
//! descriptive McNemar test; base reproducibility compares the two base arms.
#[path = "../classify.rs"]
#[allow(dead_code)]
mod classify;

use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::path::Path;
use std::{env, fs, process};

type Result<T> = std::result::Result<T, String>;

fn load(path: &Path) -> Result<Value> {
    serde_json::from_slice(&fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?).map_err(|e| format!("{}: {e}", path.display()))
}

fn responses(path: &Path) -> Result<BTreeMap<String, Value>> {
    let text = fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut rows = BTreeMap::new();
    for line in text.lines().filter(|l| !l.trim().is_empty()) {
        let row: Value = serde_json::from_str(line).map_err(|e| e.to_string())?;
        let id = row["id"].as_str().ok_or("response without id")?.to_string();
        if rows.insert(id.clone(), row).is_some() {
            return Err(format!("duplicate response {id}"));
        }
    }
    Ok(rows)
}

/// Exact two-sided McNemar p, computed as measurement.py does.
fn mcnemar(gains: u64, losses: u64) -> f64 {
    let n = gains + losses;
    if n == 0 {
        return 1.0;
    }
    let (mut term, mut total) = (1.0f64, 0.0f64);
    for k in 0..=gains.min(losses) {
        if k > 0 {
            term *= (n - k + 1) as f64 / k as f64;
        }
        total += term;
    }
    (2.0 * total / 2f64.powi(n as i32)).min(1.0)
}

struct Arm {
    correct: BTreeMap<String, bool>,
    text: BTreeMap<String, String>,
}

fn arm(cases: &[Value], run: &Path) -> Result<Arm> {
    let rows = responses(&run.join("responses.jsonl"))?;
    let (mut correct, mut text) = (BTreeMap::new(), BTreeMap::new());
    for case in cases {
        let id = case["id"].as_str().ok_or("case without id")?;
        let row = rows.get(id).ok_or(format!("missing response {id}"))?;
        correct.insert(id.to_string(), classify::grade(case, row).delivered_correct);
        text.insert(id.to_string(), row["text"].as_str().unwrap_or_default().to_string());
    }
    Ok(Arm { correct, text })
}

fn count(arm: &Arm, ids: &[&str]) -> u64 {
    ids.iter().filter(|id| arm.correct[**id]).count() as u64
}

fn run(args: &[String]) -> Result<()> {
    let flag = |name: &str| args.windows(2).find(|w| w[0] == format!("--{name}")).map(|w| w[1].clone()).ok_or(format!("missing --{name}"));
    let (ref_bundle, cand_bundle) = (flag("reference-bundle")?, flag("candidate-bundle")?);
    let (ref_run, cand_run, out) = (flag("reference-run")?, flag("candidate-run")?, flag("out")?);
    let (rp, cp) = (load(&Path::new(&ref_bundle).join("protocol.json"))?, load(&Path::new(&cand_bundle).join("protocol.json"))?);
    let ref_cases = fs::read(Path::new(&ref_bundle).join("cases.json")).map_err(|e| e.to_string())?;
    if ref_cases != fs::read(Path::new(&cand_bundle).join("cases.json")).map_err(|e| e.to_string())? || rp["cases_sha256"] != cp["cases_sha256"] {
        return Err("the two bundles do not share byte-identical cases".into());
    }
    let cases: Vec<Value> = serde_json::from_slice(&ref_cases).map_err(|e| e.to_string())?;
    let (ref_dir, cand_dir) = (Path::new(&ref_run), Path::new(&cand_run));
    let (base_a, reference) = (arm(&cases, &ref_dir.join("base"))?, arm(&cases, &ref_dir.join("adapter"))?);
    let (base_b, candidate) = (arm(&cases, &cand_dir.join("base"))?, arm(&cases, &cand_dir.join("adapter"))?);
    let (ref_report, cand_report) = (load(&ref_dir.join("scored/report.json"))?, load(&cand_dir.join("scored/report.json"))?);
    for (report, protocol) in [(&ref_report, &rp), (&cand_report, &cp)] {
        if report["run_id"] != protocol["run_id"] {
            return Err("report does not belong to its bundle".into());
        }
    }
    let ids = |kind: &str| cases.iter().filter(|c| c["kind"] == kind).filter_map(|c| c["id"].as_str()).collect::<Vec<_>>();
    let (tool, math) = (ids("tool"), ids("math"));
    // Cross-check the grader against both reports before using it for anything.
    for (report, base, adapter) in [(&ref_report, &base_a, &reference), (&cand_report, &base_b, &candidate)] {
        let panel = &report["panels"]["tool"];
        if panel["base_correct"] != json!(count(base, &tool)) || panel["adapter_correct"] != json!(count(adapter, &tool))
            || report["panels"]["math"]["adapter_correct"] != json!(count(adapter, &math)) {
            return Err(format!("grader disagrees with {}", report["run_id"]));
        }
    }
    let (gains, losses) = tool.iter().fold((0u64, 0u64), |(g, l), id| match (reference.correct[*id], candidate.correct[*id]) {
        (false, true) => (g + 1, l),
        (true, false) => (g, l + 1),
        _ => (g, l),
    });
    let mut families: BTreeMap<String, [u64; 3]> = BTreeMap::new();
    for case in cases.iter().filter(|c| c["kind"] == "tool") {
        let id = case["id"].as_str().expect("id");
        let entry = families.entry(case["family"].as_str().unwrap_or("?").to_string()).or_default();
        for (slot, arm) in [&base_b, &reference, &candidate].into_iter().enumerate() {
            entry[slot] += u64::from(arm.correct[id]);
        }
    }
    let differing: Vec<&str> = cases.iter().filter_map(|c| c["id"].as_str()).filter(|id| base_a.text[*id] != base_b.text[*id]).collect();
    let p = mcnemar(gains, losses);
    let primary = cand_report["research_improvement_confirmed"] == true;
    let replace = primary && gains > losses && p < 0.05 && count(&candidate, &math) >= count(&reference, &math);
    let summary = json!({
        "schema": "ruvllm-three-way/v1", "cases_sha256": rp["cases_sha256"],
        "reference": {"run_id": rp["run_id"], "adapter_dir": rp["adapter_dir"], "research_improvement_confirmed": ref_report["research_improvement_confirmed"], "reasons": ref_report["reasons"]},
        "candidate": {"run_id": cp["run_id"], "adapter_dir": cp["adapter_dir"], "research_improvement_confirmed": cand_report["research_improvement_confirmed"], "reasons": cand_report["reasons"]},
        "tool_correct": {"base": count(&base_b, &tool), "reference": count(&reference, &tool), "candidate": count(&candidate, &tool), "of": tool.len()},
        "math_correct": {"base": count(&base_b, &math), "reference": count(&reference, &math), "candidate": count(&candidate, &math), "of": math.len()},
        "candidate_vs_reference": {"gains": gains, "regressions": losses, "exact_mcnemar_p": p, "scope": "descriptive, preregistered"},
        "families": families.iter().map(|(k, v)| (k.clone(), json!({"base": v[0], "reference": v[1], "candidate": v[2]}))).collect::<serde_json::Map<_, _>>(),
        "base_reproducibility": {"identical_texts": cases.len() - differing.len(), "differing_ids": differing},
        "decision": {"candidate_vs_base": if primary { "keep" } else { "reject" }, "recommend_replacing_reference": replace,
            "rule": "experiments/model-improvement/training/pool-v4/preregistration.json"},
    });
    fs::create_dir_all(&out).map_err(|e| e.to_string())?;
    fs::write(Path::new(&out).join("three-way.json"), serde_json::to_string_pretty(&summary).expect("json") + "\n").map_err(|e| e.to_string())?;
    println!("{}", summary["decision"]);
    Ok(())
}

fn main() {
    if let Err(message) = run(&env::args().skip(1).collect::<Vec<_>>()) {
        eprintln!("three-way: {message}");
        eprintln!("usage: three-way --reference-bundle frozen/v4 --reference-run DIR --candidate-bundle frozen/v5 --candidate-run DIR --out DIR");
        process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::mcnemar;

    #[test]
    fn mcnemar_matches_the_scorer() {
        assert_eq!(mcnemar(0, 0), 1.0);
        assert!((mcnemar(50, 0) - 1.7763568394002505e-15).abs() < 1e-24);
        assert!((mcnemar(3, 1) - 0.625).abs() < 1e-12);
    }
}
