#!/usr/bin/env bash
# Project-local installation. No global Claude/Ruflo settings or credentials are changed.
set -euo pipefail
umask 077
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
usage() {
  cat <<'HELP'
Usage: bash scripts/install-metaharness.sh --config-dir ABS --read-root ABS \
  --policy ABS --output-root ABS [--read-root ABS ...] [--policy ABS ...] \
  [--configure-only]

Requires Python >=3.12, Node >=22 and the repository-pinned Rust toolchain.
Builds and smoke-tests the native engine, installs locked MetaHarness packages
with lifecycle scripts disabled, then creates runtime.json and launch-claude.sh
in a private config directory. Existing files are never replaced. Explicit policy
paths are SHA256-approved; inputs must be under explicit read roots. Output must
be a private directory. No model download, training, inference, global install,
credential entry, or Claude/Ruflo configuration mutation is performed.

--configure-only validates an already prepared checkout and creates configuration;
it does not install packages or rebuild. Use this for additional operator profiles.
HELP
}
if [[ $# == 0 ]]; then usage; exit 2; fi
config_dir='' output_root='' configure_only=0
read_roots=() policies=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --help|-h) usage; exit 0 ;;
    --configure-only) configure_only=1; shift ;;
    --config-dir|--output-root|--read-root|--policy)
      [[ $# -ge 2 && "$2" = /* ]] || { echo "Expected an absolute path after $1" >&2; exit 2; }
      case "$1" in
        --config-dir) config_dir="$2" ;;
        --output-root) output_root="$2" ;;
        --read-root) read_roots+=("$2") ;;
        --policy) policies+=("$2") ;;
      esac
      shift 2 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done
[[ -n "$config_dir" && -n "$output_root" && ${#read_roots[@]} -gt 0 && ${#policies[@]} -gt 0 ]] || { usage >&2; exit 2; }
command -v python3 >/dev/null
command -v node >/dev/null
python3 -c 'import sys; assert sys.version_info >= (3,12), "Python 3.12 or newer is required"'
node -e 'if(Number(process.versions.node.split(".")[0])<22) throw Error("Node 22 or newer is required")'
case "$(uname -s)" in Linux|Darwin) ;; *) echo 'A supported Unix platform is required' >&2; exit 2 ;; esac

# Validate all operator paths before an installation can perform network/build work.
python3 - "$config_dir" "$output_root" "${#read_roots[@]}" "${read_roots[@]}" "${policies[@]}" <<'PY'
import os, pathlib, stat, sys
config, output = map(pathlib.Path, sys.argv[1:3]); count=int(sys.argv[3])
roots=list(map(pathlib.Path,sys.argv[4:4+count])); policies=list(map(pathlib.Path,sys.argv[4+count:]))
if len(roots)>32 or len(policies)>128: raise SystemExit('Too many roots or policies for the bounded runtime')
def check(path):
    if any(p.is_symlink() for p in (path,*path.parents)): raise SystemExit('Symlink paths are refused')
    if '..' in path.parts: raise SystemExit('Parent traversal is refused')
for p in [config,output,*roots,*policies]: check(p)
if config == output or output in config.parents: raise SystemExit('Configuration must be outside output scope')
for p in roots:
    if not p.is_dir(): raise SystemExit('Read roots must be existing directories')
for p in policies:
    if not p.is_file() or p.stat().st_size > 1048576: raise SystemExit('Policy must be a bounded regular file')
    if not any(p.is_relative_to(r) for r in roots): raise SystemExit('Policies must be under explicit read roots')
for p in [config,output]:
    if p.exists() and (not p.is_dir() or p.stat().st_mode & 0o077 or p.stat().st_uid != os.getuid()): raise SystemExit('Existing config/output directories must be owner-only')
for name in ['runtime.json','launch-claude.sh']:
    if (config/name).exists() or (config/name).is_symlink(): raise SystemExit('Installation refuses to replace existing profile files')
PY

if [[ "$configure_only" == 0 ]]; then
  command -v cargo >/dev/null
  command -v npm >/dev/null
  (cd "$repo_root/integrations/metaharness" && npm ci --ignore-scripts)
  (cd "$repo_root/native" && cargo build --locked --release --bin weight_evolution)
fi
[[ -x "$repo_root/native/target/release/weight_evolution" ]] || { echo 'Build the native binary first' >&2; exit 1; }
[[ -d "$repo_root/integrations/metaharness/node_modules" ]] || { echo 'Install the locked integration dependencies first' >&2; exit 1; }
(cd "$repo_root" && python3 native/verify_source.py && python3 native/smoke.py)
python3 - "$repo_root" "$config_dir" "$output_root" "${#read_roots[@]}" "${read_roots[@]}" "${policies[@]}" <<'PY'
import hashlib,json,os,pathlib,shlex,stat,sys
repo,config,output=map(pathlib.Path,sys.argv[1:4]); count=int(sys.argv[4]); roots=list(map(pathlib.Path,sys.argv[5:5+count])); policies=list(map(pathlib.Path,sys.argv[5+count:]))
binary=repo/'native/target/release/weight_evolution'
def digest(p):
    with p.open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()
if binary == output or output in binary.parents: raise SystemExit('Binary must be outside output scope')
for p in [config,output]:
    p.mkdir(mode=0o700,parents=True,exist_ok=True)
    if p.is_symlink() or p.stat().st_mode & 0o077 or p.stat().st_uid != os.getuid(): raise SystemExit('Private owner-controlled directories are required')
runtime={'schema':'ruvllm-experiment-runtime/v1','binaryPath':str(binary),'binarySha256':digest(binary),'readRoots':list(map(str,roots)),'writeRoot':str(output),'policySha256s':[digest(p) for p in policies],'maxCandidates':8,'timeoutMs':120000}
runtime_path=config/'runtime.json'
fd=os.open(runtime_path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
with os.fdopen(fd,'w') as f: json.dump(runtime,f,indent=2); f.write('\n')
launcher=config/'launch-claude.sh'
fd=os.open(launcher,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o700)
with os.fdopen(fd,'w') as f:
    f.write('#!/usr/bin/env bash\nset -euo pipefail\nexport RUVLLM_EXPERIMENT_CONFIG='+shlex.quote(str(runtime_path))+'\nexec claude --plugin-dir '+shlex.quote(str(repo/'plugins/ruflo-metaharness'))+' "$@"\n')
print(json.dumps({'installed':True,'runtimeConfig':str(runtime_path),'claudeLauncher':str(launcher),'productionActivation':False}))
PY
