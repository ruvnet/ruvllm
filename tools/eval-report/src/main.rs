//! `eval-report`: turn a scored paired measurement into a failure taxonomy and
//! a static dashboard. Reads the frozen bundle, both arms' raw responses and
//! the scorer's report. Refuses to write anything if its own per-case grading
//! disagrees with the scorer's aggregates, so the taxonomy can never drift from
//! the preregistered verdict.
mod classify;
mod render;

use classify::{grade, Category};
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::{env, fs, process};

type Result<T> = std::result::Result<T, String>;

pub struct CaseResult {
    pub id: String,
    pub kind: String,
    pub family: String,
    pub base: classify::Graded,
    pub adapter: classify::Graded,
    pub base_tokens: u64,
    pub adapter_tokens: u64,
    pub adapter_excerpt: String,
    pub base_excerpt: String,
}

fn read(path: &Path) -> Result<Vec<u8>> {
    fs::read(path).map_err(|e| format!("{}: {e}", path.display()))
}

fn sha(bytes: &[u8]) -> String {
    Sha256::digest(bytes).iter().map(|b| format!("{b:02x}")).collect()
}

fn responses(path: &Path) -> Result<BTreeMap<String, Value>> {
    let text = String::from_utf8(read(path)?).map_err(|_| "responses are not UTF-8".to_string())?;
    let mut rows = BTreeMap::new();
    for line in text.lines().filter(|l| !l.trim().is_empty()) {
        let row: Value = serde_json::from_str(line).map_err(|e| format!("bad response row: {e}"))?;
        let id = row["id"].as_str().ok_or("response without id")?.to_string();
        if rows.insert(id.clone(), row).is_some() {
            return Err(format!("duplicate response {id}"));
        }
    }
    Ok(rows)
}

fn excerpt(text: &str) -> String {
    let flat: String = text.replace("<|im_end|>", "").chars().map(|c| if c.is_control() { ' ' } else { c }).collect();
    let trimmed = flat.trim();
    if trimmed.chars().count() > 240 { format!("{}…", trimmed.chars().take(240).collect::<String>()) } else { trimmed.to_string() }
}

fn tally<'a>(results: impl Iterator<Item = &'a CaseResult>, adapter: bool) -> BTreeMap<&'static str, u64> {
    let mut counts = BTreeMap::new();
    for r in results {
        let g = if adapter { &r.adapter } else { &r.base };
        *counts.entry(g.category.name()).or_insert(0) += 1;
    }
    counts
}

fn check(report: &Value, results: &[CaseResult]) -> Result<()> {
    for kind in ["tool", "math"] {
        let panel = &report["panels"][kind];
        let rows: Vec<_> = results.iter().filter(|r| r.kind == kind).collect();
        let base = rows.iter().filter(|r| r.base.delivered_correct).count() as u64;
        let adapter = rows.iter().filter(|r| r.adapter.delivered_correct).count() as u64;
        let gains = rows.iter().filter(|r| !r.base.delivered_correct && r.adapter.delivered_correct).count() as u64;
        let losses = rows.iter().filter(|r| r.base.delivered_correct && !r.adapter.delivered_correct).count() as u64;
        let expected = [panel["base_correct"].as_u64(), panel["adapter_correct"].as_u64(), panel["gains"].as_u64(), panel["regressions"].as_u64()];
        if expected != [Some(base), Some(adapter), Some(gains), Some(losses)] {
            return Err(format!("taxonomy disagrees with scorer on {kind}: scorer {expected:?}, taxonomy {:?}", [base, adapter, gains, losses]));
        }
    }
    Ok(())
}

fn decision(report: &Value) -> Value {
    let confirmed = report["research_improvement_confirmed"] == true;
    let tool = &report["panels"]["tool"];
    let delta = tool["delta_percentage_points"].as_f64().unwrap_or(0.0);
    let verdict = if confirmed { "keep" } else { "reject" };
    let basis = if confirmed {
        "Preregistered ADR-016 rule satisfied: positive tool delta, McNemar p < 0.05, no arithmetic or completion regression."
    } else if delta > 0.0 {
        "Positive point estimate, but the preregistered rule is not satisfied; treat as inconclusive, not as improvement."
    } else {
        "The adapter did not improve delivered tool accuracy on the frozen panel."
    };
    json!({"verdict": verdict, "basis": basis, "reasons": report["reasons"].clone(),
           "production_activation": false, "rule": "ADR-016 research qualification"})
}

fn run(args: &[String]) -> Result<()> {
    let mut flags: BTreeMap<&str, PathBuf> = BTreeMap::new();
    for pair in args.chunks(2) {
        let [key, value] = pair else { return Err("flags need values".into()) };
        let name = key.strip_prefix("--").ok_or("expected --flag")?;
        if !["bundle", "base-run", "adapter-run", "report", "out"].contains(&name) {
            return Err(format!("unknown flag {key}"));
        }
        flags.insert(["bundle", "base-run", "adapter-run", "report", "out"].into_iter().find(|n| *n == name).unwrap(), PathBuf::from(value));
    }
    let get = |name: &str| flags.get(name).cloned().ok_or(format!("missing --{name}"));
    let (bundle, base_run, adapter_run, report_path, out) = (get("bundle")?, get("base-run")?, get("adapter-run")?, get("report")?, get("out")?);
    let cases_bytes = read(&bundle.join("cases.json"))?;
    let protocol: Value = serde_json::from_slice(&read(&bundle.join("protocol.json"))?).map_err(|e| e.to_string())?;
    if protocol["cases_sha256"] != sha(&cases_bytes) {
        return Err("cases.json does not match the frozen protocol".into());
    }
    let cases: Vec<Value> = serde_json::from_slice(&cases_bytes).map_err(|e| e.to_string())?;
    let report_bytes = read(&report_path)?;
    let report: Value = serde_json::from_slice(&report_bytes).map_err(|e| e.to_string())?;
    if report["run_id"] != protocol["run_id"] {
        return Err("report belongs to a different run".into());
    }
    let (base_rows, adapter_rows) = (responses(&base_run.join("responses.jsonl"))?, responses(&adapter_run.join("responses.jsonl"))?);
    let mut results = Vec::new();
    for case in &cases {
        let id = case["id"].as_str().ok_or("case without id")?;
        let (b, a) = (base_rows.get(id).ok_or(format!("base missing {id}"))?, adapter_rows.get(id).ok_or(format!("adapter missing {id}"))?);
        results.push(CaseResult {
            id: id.to_string(), kind: case["kind"].as_str().unwrap_or_default().to_string(),
            family: case["family"].as_str().unwrap_or_default().to_string(),
            base: grade(case, b), adapter: grade(case, a),
            base_tokens: b["output_tokens"].as_u64().unwrap_or(0), adapter_tokens: a["output_tokens"].as_u64().unwrap_or(0),
            adapter_excerpt: excerpt(a["text"].as_str().unwrap_or_default()), base_excerpt: excerpt(b["text"].as_str().unwrap_or_default()),
        });
    }
    check(&report, &results)?;

    let mut families: BTreeMap<String, Map<String, Value>> = BTreeMap::new();
    for r in &results {
        let entry = families.entry(r.family.clone()).or_default();
        for (key, hit) in [("count", true), ("base_correct", r.base.delivered_correct), ("adapter_correct", r.adapter.delivered_correct),
                           ("gains", !r.base.delivered_correct && r.adapter.delivered_correct),
                           ("regressions", r.base.delivered_correct && !r.adapter.delivered_correct)] {
            let value = entry.get(key).and_then(Value::as_u64).unwrap_or(0) + u64::from(hit);
            entry.insert(key.into(), value.into());
        }
    }
    let mut regression_groups: BTreeMap<&str, u64> = BTreeMap::new();
    let mut regressions = Vec::new();
    let mut gains = Vec::new();
    for r in &results {
        let row = json!({"id": r.id, "family": r.family, "base": r.base.category.name(), "adapter": r.adapter.category.name(),
                         "detail": r.adapter.detail.clone().or(r.base.detail.clone()), "adapter_excerpt": r.adapter_excerpt, "base_excerpt": r.base_excerpt});
        if r.base.delivered_correct && !r.adapter.delivered_correct {
            *regression_groups.entry(r.adapter.category.group()).or_insert(0) += 1;
            regressions.push(row);
        } else if !r.base.delivered_correct && r.adapter.delivered_correct {
            gains.push(row);
        }
    }
    let by_kind = |kind: &str, adapter: bool| json!(tally(results.iter().filter(|r| r.kind == kind), adapter));
    let failure_report = json!({
        "schema": "ruvllm-failure-report/v1", "run_id": protocol["run_id"], "report_sha256": sha(&report_bytes),
        "decision": decision(&report),
        "categories": {"tool": {"base": by_kind("tool", false), "adapter": by_kind("tool", true)},
                       "math": {"base": by_kind("math", false), "adapter": by_kind("math", true)}},
        "category_groups": Category::ALL.iter().map(|c| (c.name(), c.group())).collect::<BTreeMap<_, _>>(),
        "families": families, "regressions_by_group": regression_groups, "regressions": regressions, "gains": gains,
        "mean_output_tokens": {"base": results.iter().map(|r| r.base_tokens).sum::<u64>() as f64 / results.len() as f64,
                               "adapter": results.iter().map(|r| r.adapter_tokens).sum::<u64>() as f64 / results.len() as f64},
        "scope": "Descriptive breakdown of one frozen paired run; only the scorer's primary test supports the decision.",
    });
    fs::create_dir_all(&out).map_err(|e| e.to_string())?;
    let json_text = serde_json::to_string_pretty(&failure_report).map_err(|e| e.to_string())? + "\n";
    fs::write(out.join("failure-report.json"), &json_text).map_err(|e| e.to_string())?;
    fs::write(out.join("dashboard.html"), render::dashboard(&report, &failure_report)).map_err(|e| e.to_string())?;
    println!("{}", json!({"out": out, "decision": failure_report["decision"]["verdict"], "regressions": failure_report["regressions"].as_array().map_or(0, Vec::len)}));
    Ok(())
}

fn main() {
    let args: Vec<String> = env::args().skip(1).collect();
    if let Err(message) = run(&args) {
        eprintln!("eval-report: {message}");
        eprintln!("usage: eval-report --bundle DIR --base-run DIR --adapter-run DIR --report report.json --out DIR");
        process::exit(1);
    }
}
