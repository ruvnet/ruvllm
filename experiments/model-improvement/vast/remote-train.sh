#!/usr/bin/env bash
# Runs on a rented GPU host from a checkout pinned to an exact commit. It
# receives no provider credential. It installs the pinned packages, caches the
# public pinned model snapshot, runs scripts/cloud_train.py unmodified on the
# committed RUFLO_TRAIN_GZIP payload, and leaves the checkpoints and the
# trainer's event log in /workspace/out and /workspace/run for the control host,
# which verifies the manifest and payload hash before accepting the result.
set -Eeuo pipefail

COMMIT=${1:?commit}
DEADLINE=${2:?training deadline seconds}
PAYLOAD=${3:?payload directory relative to the repository}
REPO=/workspace/ruvllm
RUN=/workspace/run
OUT=/workspace/out
REVISION=851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a
STEP=start
mkdir -p "$RUN" "$OUT"

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
case "$PAYLOAD" in experiments/model-improvement/training/*) ;; *) exit 1 ;; esac
test -f "$REPO/$PAYLOAD/RUFLO_TRAIN_GZIP"

step packages
python3 -m pip install --break-system-packages --no-cache-dir -q torch==2.11.0 transformers==5.12.1 peft==0.21.2 \
  safetensors==0.7.0 accelerate==1.12.0 huggingface_hub
python3 -m pip freeze > "$RUN/pip-freeze.txt"
nvidia-smi --query-gpu=name,uuid,driver_version,memory.total --format=csv,noheader > "$RUN/gpu.txt"

step model-download
# cloud_train.py loads by repository name and revision, so populate the hub cache.
HF_HUB_DISABLE_TELEMETRY=1 hf download Qwen/Qwen3.5-4B --revision "$REVISION" \
  --exclude '*.md' > "$RUN/download.log" 2>&1

step training
mkdir -p /checkpoints
cd "$REPO"
RUFLO_TRAIN_GZIP=$(cat "$PAYLOAD/RUFLO_TRAIN_GZIP") HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  timeout "$DEADLINE" python3 scripts/cloud_train.py > "$RUN/train.log" 2>&1
grep -q '^RUFLO_TRAIN {"kind": "completed"' "$RUN/train.log"
cp -r /checkpoints/training-v1 "$OUT/training-v1"
status done trained
