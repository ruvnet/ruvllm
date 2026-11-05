#!/usr/bin/env bash
# Runs on a rented GPU host from a checkout pinned to an exact commit.
# It receives no provider credential. It installs the protocol's pinned
# packages (into the image's system Python, which carries torch), downloads the public pinned model snapshot, runs both arms with
# run-pair.py and leaves raw evidence in /workspace/out for the control host.
# /workspace/run/status.json is a progress hint only; the control host
# re-scores the retrieved raw outputs before accepting any result.
set -Eeuo pipefail

COMMIT=${1:?commit}
ARM_DEADLINE=${2:?per-arm deadline seconds}
BUNDLE_NAME=${3:-v3}
ADAPTER_DIR=${4:-coding-4b}
REPO=/workspace/ruvllm
RUN=/workspace/run
OUT=/workspace/out
MODEL=/workspace/models/Qwen3.5-4B-851bf6e
REVISION=851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a
STEP=start
mkdir -p "$RUN"

status() {
  printf '{"schema":"ruvllm-vast-remote-status/v1","state":"%s","step":"%s","commit":"%s","at":%s}\n' \
    "$1" "$2" "$COMMIT" "$(date +%s)" > "$RUN/status.json.tmp"
  mv "$RUN/status.json.tmp" "$RUN/status.json"
}
step() { STEP=$1; status running "$1"; }
trap 'status failed "$STEP"' ERR

step verify-checkout
test "$(git -C "$REPO" rev-parse HEAD)" = "$COMMIT"
test -z "$(git -C "$REPO" status --porcelain)"
[[ $ADAPTER_DIR =~ ^[a-z0-9][a-z0-9._-]{0,63}$ && $ADAPTER_DIR != *..* ]]

step packages
python3 -m pip install --break-system-packages --no-cache-dir -q torch==2.11.0 transformers==5.12.1 peft==0.21.2 \
  safetensors==0.7.0 accelerate==1.12.0 huggingface_hub
python3 -m pip freeze > "$RUN/pip-freeze.txt"
nvidia-smi --query-gpu=name,uuid,driver_version,memory.total --format=csv,noheader > "$RUN/gpu.txt"

step model-download
includes=()
for name in chat_template.jinja config.json merges.txt model.safetensors.index.json \
    preprocessor_config.json tokenizer.json tokenizer_config.json \
    video_preprocessor_config.json vocab.json 'model.safetensors-*'; do
  includes+=(--include "$name")
done
HF_HUB_DISABLE_TELEMETRY=1 hf download Qwen/Qwen3.5-4B --revision "$REVISION" \
  --local-dir "$MODEL" "${includes[@]}" > "$RUN/download.log" 2>&1
# The worker rejects any unpinned file; drop the downloader's lock/metadata cache.
rm -rf "$MODEL/.cache"

step paired-inference
cd "$REPO"
python3 experiments/model-improvement/run-pair.py \
  --bundle "$REPO/experiments/model-improvement/frozen/$BUNDLE_NAME" \
  --model-dir "$MODEL" --adapter-dir "$REPO/weights/$ADAPTER_DIR" \
  --out "$OUT" --deadline-seconds "$ARM_DEADLINE" > "$RUN/run-pair.log" 2>&1

test -f "$OUT/scored/report.json"
status done scored
