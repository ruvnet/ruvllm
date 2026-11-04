//! `sft-bridge`: connect a Weight-EFT SFT export to the pinned PEFT trainer
//! (`scripts/cloud_train.py`, executed unmodified). Weight-EFT emits OpenAI chat
//! JSONL with `tool_calls` but no tool schemas; the trainer needs
//! `{train, valid, training_sha256}` rows carrying `tools`, delivered gzip+base64
//! in `RUFLO_TRAIN_GZIP`. The bridge re-attaches one reviewed tool schema set,
//! normalizes string tool arguments to objects, splits train/valid
//! deterministically, and refuses any row that overlaps a frozen evaluation
//! panel by case ID or by exact prompt messages. It trains nothing itself.
use flate2::{write::GzEncoder, Compression};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::io::Write;
use std::{env, fs, process};

type Result<T> = std::result::Result<T, String>;

fn sha(bytes: &[u8]) -> String {
    Sha256::digest(bytes).iter().map(|b| format!("{b:02x}")).collect()
}

fn base64(bytes: &[u8]) -> String {
    const TABLE: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let n = (u32::from(chunk[0]) << 16) | (u32::from(*chunk.get(1).unwrap_or(&0)) << 8) | u32::from(*chunk.get(2).unwrap_or(&0));
        for (i, shift) in [18, 12, 6, 0].into_iter().enumerate() {
            out.push(if i <= chunk.len() { TABLE[(n >> shift) as usize & 63] as char } else { '=' });
        }
    }
    out
}

/// Prompt identity: every message before the final assistant target.
fn prompt_key(messages: &[Value]) -> String {
    serde_json::to_string(&messages[..messages.len().saturating_sub(1)]).unwrap_or_default()
}

fn normalize(mut row: Value, tools: &Value, names: &BTreeSet<String>) -> Result<Value> {
    let messages = row["messages"].as_array_mut().ok_or("SFT row without messages")?;
    let last = messages.last().ok_or("empty SFT row")?;
    if last["role"] != "assistant" || messages.len() < 2 {
        return Err("SFT row must end with an assistant target after a prompt".into());
    }
    for message in messages.iter_mut() {
        for call in message["tool_calls"].as_array_mut().into_iter().flatten() {
            let function = &mut call["function"];
            let name = function["name"].as_str().ok_or("tool call without name")?;
            if !names.contains(name) {
                return Err(format!("tool call to unreviewed tool {name}"));
            }
            if let Some(text) = function["arguments"].as_str() {
                function["arguments"] = serde_json::from_str(text).map_err(|_| "tool arguments are not JSON")?;
            }
            if !function["arguments"].is_object() {
                return Err("tool arguments must be an object".into());
            }
        }
    }
    Ok(json!({"messages": row["messages"].take(), "tools": tools}))
}

fn run(args: &[String]) -> Result<()> {
    let flag = |name: &str| args.windows(2).find(|w| w[0] == format!("--{name}")).map(|w| w[1].clone()).ok_or(format!("missing --{name}"));
    let (sft, holdout, out) = (flag("sft")?, flag("holdout-cases")?, flag("out")?);
    let valid_every: usize = flag("valid-every").unwrap_or_else(|_| "9".into()).parse().map_err(|_| "--valid-every must be an integer")?;
    let cases: Vec<Value> = serde_json::from_slice(&fs::read(&holdout).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
    // Tool schemas come from the frozen panel: training and evaluation share the reviewed tool surface.
    let tools = cases.iter().find(|c| c["kind"] == "tool").map(|c| c["tools"].clone()).ok_or("holdout has no tool cases")?;
    let names: BTreeSet<String> = tools.as_array().into_iter().flatten().filter_map(|t| t["function"]["name"].as_str().map(String::from)).collect();
    let held_ids: BTreeSet<&str> = cases.iter().filter_map(|c| c["id"].as_str()).collect();
    let held_prompts: BTreeSet<String> = cases.iter().filter_map(|c| c["messages"].as_array()).map(|m| serde_json::to_string(m).unwrap_or_default()).collect();
    let text = fs::read_to_string(&sft).map_err(|e| e.to_string())?;
    let (mut train, mut valid, mut seen) = (Vec::new(), Vec::new(), BTreeSet::new());
    for (index, line) in text.lines().filter(|l| !l.trim().is_empty()).enumerate() {
        let row: Value = serde_json::from_str(line).map_err(|e| format!("row {index}: {e}"))?;
        if row["instance_id"].as_str().is_some_and(|id| held_ids.contains(id)) {
            return Err(format!("row {index} reuses a held-out evaluation case ID"));
        }
        let row = normalize(row, &tools, &names).map_err(|e| format!("row {index}: {e}"))?;
        let key = prompt_key(row["messages"].as_array().expect("normalized"));
        if held_prompts.contains(&key) {
            return Err(format!("row {index} repeats a held-out evaluation prompt"));
        }
        if !seen.insert(key) {
            continue; // Duplicate prompt: keep the first target only.
        }
        if valid_every > 0 && index % valid_every == valid_every - 1 { valid.push(row) } else { train.push(row) }
    }
    if train.is_empty() || valid.is_empty() {
        return Err("need at least one training and one validation row".into());
    }
    let train_jsonl: String = train.iter().map(|r| serde_json::to_string(r).expect("json") + "\n").collect();
    let payload = json!({"train": train, "valid": valid, "training_sha256": sha(train_jsonl.as_bytes())});
    let mut gz = GzEncoder::new(Vec::new(), Compression::best());
    gz.write_all(serde_json::to_string(&payload).map_err(|e| e.to_string())?.as_bytes()).map_err(|e| e.to_string())?;
    let encoded = base64(&gz.finish().map_err(|e| e.to_string())?);
    fs::create_dir_all(&out).map_err(|e| e.to_string())?;
    let out = std::path::Path::new(&out);
    fs::write(out.join("train.jsonl"), &train_jsonl).map_err(|e| e.to_string())?;
    fs::write(out.join("RUFLO_TRAIN_GZIP"), &encoded).map_err(|e| e.to_string())?;
    let receipt = json!({"schema": "ruvllm-sft-bridge/v1", "train_rows": train.len(), "valid_rows": valid.len(),
        "training_sha256": payload["training_sha256"], "sft_sha256": sha(text.as_bytes()), "holdout_cases": held_ids.len(),
        "trainer": "scripts/cloud_train.py (unmodified)", "payload_bytes": encoded.len(), "training_executed": false});
    fs::write(out.join("bridge.json"), serde_json::to_string_pretty(&receipt).expect("json") + "\n").map_err(|e| e.to_string())?;
    println!("{receipt}");
    Ok(())
}

fn main() {
    if let Err(message) = run(&env::args().skip(1).collect::<Vec<_>>()) {
        eprintln!("sft-bridge: {message}");
        eprintln!("usage: sft-bridge --sft sft.jsonl --holdout-cases frozen/v3/cases.json --out DIR [--valid-every 9]");
        process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base64_matches_rfc4648() {
        assert_eq!(base64(b""), "");
        assert_eq!(base64(b"f"), "Zg==");
        assert_eq!(base64(b"fo"), "Zm8=");
        assert_eq!(base64(b"foobar"), "Zm9vYmFy");
    }

    #[test]
    fn normalizes_string_arguments_and_rejects_unreviewed_tools() {
        let tools = json!([{"function": {"name": "Read"}}]);
        let names: BTreeSet<String> = ["Read".to_string()].into();
        let row = json!({"messages": [{"role": "user", "content": "x"}, {"role": "assistant", "content": "",
            "tool_calls": [{"function": {"name": "Read", "arguments": "{\"file_path\":\"/a\"}"}}]}]});
        let out = normalize(row, &tools, &names).unwrap();
        assert_eq!(out["messages"][1]["tool_calls"][0]["function"]["arguments"]["file_path"], "/a");
        assert_eq!(out["tools"], tools);
        let bad = json!({"messages": [{"role": "user", "content": "x"}, {"role": "assistant", "tool_calls": [{"function": {"name": "Bash", "arguments": {}}}]}]});
        assert!(normalize(bad, &tools, &names).is_err());
        assert!(normalize(json!({"messages": [{"role": "user", "content": "x"}]}), &tools, &names).is_err());
    }
}
