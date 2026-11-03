//! JSON CLI for the default-off, offline weight evolution engine.
use std::io::Read;
fn main() {
    let result = (|| -> ruvllm::weight_evolution::Result<serde_json::Value> {
        let bytes = if let Some(path) = std::env::args().nth(1) {
            ruvllm::weight_evolution::read_bounded(std::path::Path::new(&path), 1024 * 1024)?
        } else {
            let mut bytes = Vec::new();
            std::io::stdin()
                .take(1024 * 1024 + 1)
                .read_to_end(&mut bytes)?;
            if bytes.len() > 1024 * 1024 {
                return Err("request exceeds 1 MiB".into());
            }
            bytes
        };
        let request = serde_json::from_slice(&bytes)?;
        Ok(ruvllm::weight_evolution::execute(request)?)
    })();
    match result {
        Ok(value) => println!("{}", serde_json::to_string(&value).unwrap()),
        Err(error) => {
            eprintln!(
                "{}",
                serde_json::json!({"ok": false, "error": error.to_string()})
            );
            std::process::exit(1);
        }
    }
}
