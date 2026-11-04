//! Static, self-contained HTML dashboard. No scripts or external resources.
use serde_json::Value;
use std::fmt::Write;

fn esc(text: &str) -> String {
    text.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;").replace('"', "&quot;")
}

fn num(value: &Value) -> u64 { value.as_u64().unwrap_or(0) }

fn pct(n: u64, d: u64) -> String { if d == 0 { "–".into() } else { format!("{:.1}%", 100.0 * n as f64 / d as f64) } }

const STYLE: &str = r#"
:root{--bg:#f7f7f5;--panel:#fff;--ink:#1c1d1f;--muted:#60646c;--line:#e2e3e0;--good:#1f7a4d;--bad:#b4332e;--warn:#9a6700;--accent:#2f5bd3;--bar:#c9d3f2}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#141518;--panel:#1c1d21;--ink:#e8e8e6;--muted:#9a9ea6;--line:#2c2e33;--good:#4cc38a;--bad:#f2726b;--warn:#e2b340;--accent:#8aa6ff;--bar:#34406b}}
:root[data-theme="dark"]{--bg:#141518;--panel:#1c1d21;--ink:#e8e8e6;--muted:#9a9ea6;--line:#2c2e33;--good:#4cc38a;--bad:#f2726b;--warn:#e2b340;--accent:#8aa6ff;--bar:#34406b}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1080px;margin:0 auto;padding:24px 16px 64px}h1{font-size:22px;margin:0 0 4px}h2{font-size:16px;margin:32px 0 10px}
.muted{color:var(--muted)}.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px;margin-top:16px}
.tile{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:14px}.tile b{display:block;font-size:24px;font-variant-numeric:tabular-nums}
.tile span{font-size:13px;color:var(--muted)}.verdict{display:inline-block;padding:4px 12px;border-radius:999px;font-weight:700;letter-spacing:.04em;margin:8px 0}
.keep{background:color-mix(in srgb,var(--good) 18%,transparent);color:var(--good)}.reject{background:color-mix(in srgb,var(--bad) 16%,transparent);color:var(--bad)}
.wrap{overflow-x:auto;background:var(--panel);border:1px solid var(--line);border-radius:10px}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}
th,td{padding:7px 10px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}th{font-size:12px;text-transform:uppercase;letter-spacing:.04em;color:var(--muted)}
td.n{text-align:right}.up{color:var(--good)}.down{color:var(--bad)}code{font:12px/1.4 ui-monospace,Menlo,monospace;word-break:break-word}
.bar{height:8px;background:var(--bar);border-radius:4px}details{margin-top:12px}summary{cursor:pointer;color:var(--accent)}
"#;

fn tile(out: &mut String, value: &str, label: &str) {
    let _ = write!(out, "<div class=tile><b>{}</b><span>{}</span></div>", esc(value), esc(label));
}

fn case_table(out: &mut String, rows: &[Value]) {
    out.push_str("<div class=wrap><table><tr><th>Case</th><th>Base</th><th>Adapter</th><th>Detail</th><th>Adapter output (excerpt)</th></tr>");
    for row in rows {
        let _ = write!(out, "<tr><td><code>{}</code><br><span class=muted>{}</span></td><td>{}</td><td>{}</td><td>{}</td><td><code>{}</code></td></tr>",
            esc(row["id"].as_str().unwrap_or("")), esc(row["family"].as_str().unwrap_or("")), esc(row["base"].as_str().unwrap_or("")),
            esc(row["adapter"].as_str().unwrap_or("")), esc(row["detail"].as_str().unwrap_or("")), esc(row["adapter_excerpt"].as_str().unwrap_or("")));
    }
    out.push_str("</table></div>");
}

pub fn dashboard(report: &Value, failures: &Value) -> String {
    let (tool, math) = (&report["panels"]["tool"], &report["panels"]["math"]);
    let decision = &failures["decision"];
    let verdict = decision["verdict"].as_str().unwrap_or("reject");
    let run_id = report["run_id"].as_str().unwrap_or("");
    let mut out = String::new();
    let _ = write!(out, "<!doctype html><html lang=en><head><meta charset=utf-8><meta name=viewport content=\"width=device-width,initial-scale=1\">\
        <title>Adapter Measurement Results</title><style>{STYLE}</style></head><body><main>");
    let _ = write!(out, "<h1>Qwen3.5-4B coding adapter vs base</h1><div class=muted>Run <code>{}</code> · protocol <code>{}</code> · {} paired cases · evidence: {}</div>",
        esc(run_id), esc(&report["protocol_sha256"].as_str().unwrap_or("")[..12.min(report["protocol_sha256"].as_str().unwrap_or("").len())]),
        num(&tool["count"]) + num(&math["count"]), esc(report["evidence_mode"].as_str().unwrap_or("")));
    let _ = write!(out, "<div class=\"verdict {verdict}\">{}</div><div>{}</div>", verdict.to_uppercase(), esc(decision["basis"].as_str().unwrap_or("")));
    if let Some(reasons) = decision["reasons"].as_array().filter(|r| !r.is_empty()) {
        let list: Vec<String> = reasons.iter().filter_map(Value::as_str).map(esc).collect();
        let _ = write!(out, "<div class=muted>Unmet criteria: <code>{}</code></div>", list.join("</code>, <code>"));
    }
    out.push_str("<div class=tiles>");
    let tool_n = num(&tool["count"]);
    tile(&mut out, &format!("{} → {}", pct(num(&tool["base_correct"]), tool_n), pct(num(&tool["adapter_correct"]), tool_n)), "Delivered exact tool calls (base → adapter)");
    tile(&mut out, &format!("{:+.1} pp", tool["delta_percentage_points"].as_f64().unwrap_or(0.0)), "Tool accuracy difference");
    tile(&mut out, &format!("{:.4}", tool["exact_mcnemar_p"].as_f64().unwrap_or(1.0)), "Exact McNemar p (primary, needs < 0.05)");
    tile(&mut out, &format!("+{} / −{}", num(&tool["gains"]), num(&tool["regressions"])), "Paired tool gains / regressions");
    tile(&mut out, &format!("{} → {}", pct(num(&math["base_correct"]), num(&math["count"])), pct(num(&math["adapter_correct"]), num(&math["count"]))), "Arithmetic anchor (must not drop)");
    let totals = &report["totals"];
    tile(&mut out, &format!("{} → {}", pct(num(&totals["base"]["completed"]), num(&totals["base"]["count"])), pct(num(&totals["adapter"]["completed"]), num(&totals["adapter"]["count"]))), "Completion rate (must not drop)");
    out.push_str("</div>");

    out.push_str("<h2>Tool accuracy by family</h2><div class=wrap><table><tr><th>Family</th><th>Base</th><th>Adapter</th><th>Gains</th><th>Regressions</th><th style=\"width:30%\">Adapter accuracy</th></tr>");
    if let Some(families) = failures["families"].as_object() {
        for (name, row) in families {
            let (n, a) = (num(&row["count"]), num(&row["adapter_correct"]));
            let _ = write!(out, "<tr><td>{}</td><td class=n>{}</td><td class=n>{}</td><td class=\"n up\">{}</td><td class=\"n down\">{}</td><td><div class=bar style=\"width:{:.0}%\"></div></td></tr>",
                esc(name), pct(num(&row["base_correct"]), n), pct(a, n), num(&row["gains"]), num(&row["regressions"]), if n == 0 { 0.0 } else { 100.0 * a as f64 / n as f64 });
        }
    }
    out.push_str("</table></div>");

    out.push_str("<h2>Failure categories</h2><div class=wrap><table><tr><th>Category</th><th>Group</th><th>Tool base</th><th>Tool adapter</th><th>Math base</th><th>Math adapter</th></tr>");
    let groups = &failures["category_groups"];
    if let Some(names) = groups.as_object() {
        let cats = &failures["categories"];
        for (name, group) in names {
            let cells = [&cats["tool"]["base"][name], &cats["tool"]["adapter"][name], &cats["math"]["base"][name], &cats["math"]["adapter"][name]];
            if cells.iter().all(|c| num(c) == 0) { continue; }
            let _ = write!(out, "<tr><td><code>{}</code></td><td>{}</td>", esc(name), esc(group.as_str().unwrap_or("")));
            for c in cells { let _ = write!(out, "<td class=n>{}</td>", num(c)); }
            out.push_str("</tr>");
        }
    }
    out.push_str("</table></div>");

    let regressions = failures["regressions"].as_array().cloned().unwrap_or_default();
    let _ = write!(out, "<h2>Regressions ({}): base correct, adapter wrong</h2>", regressions.len());
    if let Some(groups) = failures["regressions_by_group"].as_object().filter(|g| !g.is_empty()) {
        let parts: Vec<String> = groups.iter().map(|(k, v)| format!("{} {}", esc(k), num(v))).collect();
        let _ = write!(out, "<p class=muted>By adapter failure group: {}</p>", parts.join(" · "));
    }
    case_table(&mut out, &regressions);
    let gains = failures["gains"].as_array().cloned().unwrap_or_default();
    let _ = write!(out, "<details><summary>Gains ({}): base wrong, adapter correct</summary>", gains.len());
    case_table(&mut out, &gains);
    out.push_str("</details>");
    let _ = write!(out, "<h2>Scope</h2><p class=muted>One frozen 256-case synthetic panel, one seed, one GPU. The family and category breakdowns are descriptive; \
        only the preregistered primary test decides keep/reject. Research qualification never authorizes production activation.</p>\
        <p class=muted>Report SHA256 <code>{}</code></p></main></body></html>\n", esc(failures["report_sha256"].as_str().unwrap_or("")));
    out
}
