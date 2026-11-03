#!/usr/bin/env bash
set -euo pipefail
plugin_root="$(cd "$(dirname "$0")/.." && pwd)"
node "$plugin_root/scripts/structure.mjs"
node --check "$plugin_root/scripts/server.mjs"
