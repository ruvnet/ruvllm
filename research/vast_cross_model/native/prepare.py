#!/usr/bin/env python3
"""Public source projection and Linux native proof. No provisioning or credentials."""
import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import platform
import re
import shutil
import stat
import subprocess
import tarfile
import time
import tomllib
import urllib.request

BASE = '6a879a726b0d1d63e70a1ddb656263d8e6bab8a0'
COMMIT = 'f5c3444968ede44e43744460192198962ae94da2'
TREE = 'f0a37eeeaf7c515ca6724057536e308112d4ea75'
RUST = '1.97.1'
BASE_MODEL = '04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696'
CANDIDATE_MODEL = 'bd5c223dc7d0ed48a6857f798a94da6ec46c40b9ff4e7bc31872dba10f943ae5'
PATCH = 'c77336ccbabbbde998bf5341c61afef922cd78f08b1bb2364b177e3bb52ef786'
POLICY = '9745cf81af36ea7f6c41bd923163d23ea19c8e08c5ce182bfc84e8e84c1b50b4'
MAX_FILE = 32 * 1024 * 1024
MAX_SOURCE = 384 * 1024 * 1024
MAX_ARCHIVE = 256 * 1024 * 1024
MANIFEST = 'native-source-manifest.json'
REQUIREMENTS_SHA = '37ddb481023ffd0614367fda31ae75d6e1d07592dd7a34086e2f349590640ac9'
PUBLIC_BUILD_ENV = {'PATH', 'HOME', 'USER', 'LOGNAME', 'LANG', 'LC_ALL', 'LC_CTYPE', 'TZ',
                    'TMPDIR', 'TMP', 'TEMP', 'LD_LIBRARY_PATH', 'LIBRARY_PATH', 'CPATH',
                    'C_INCLUDE_PATH', 'CPLUS_INCLUDE_PATH', 'PKG_CONFIG_PATH', 'PKG_CONFIG_LIBDIR',
                    'CC', 'CXX', 'AR', 'CFLAGS', 'CPPFLAGS', 'CXXFLAGS', 'LDFLAGS',
                    'SSL_CERT_FILE', 'SSL_CERT_DIR', 'CARGO_HOME', 'RUSTUP_HOME'}


def clean_environment():
    """Public source/build children receive no cloud or model-provider credentials."""
    return {name: value for name, value in os.environ.items() if name in PUBLIC_BUILD_ENV}


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def git(repo, *args):
    return subprocess.check_output(['git', '-C', str(repo), *args], env=clean_environment())


def relative(name):
    p = PurePosixPath(name)
    if p.is_absolute() or '\\' in name or not p.parts or any(x in ('.', '..') for x in p.parts):
        raise ValueError(f'Unsafe source path: {name}')
    return p


def requirements():
    path = Path(__file__).with_name('bootstrap-requirements.json')
    if sha(path) != REQUIREMENTS_SHA:
        raise ValueError('Bootstrap requirement digest mismatch')
    return read_json(path)


def linux_worker():
    if platform.system() != 'Linux' or platform.machine() not in ('x86_64', 'amd64'):
        raise ValueError('Requires a Linux x86_64 worker; local preparation does not prove portability')


def compiler_preflight():
    required = requirements()
    missing = [name for name in required['required_programs'] if not shutil.which(name)]
    if missing:
        raise ValueError('Missing compiler tools: ' + ', '.join(missing) +
                         '; Ubuntu packages: ' + ' '.join(required['ubuntu_packages_if_missing']))
    return {name: shutil.which(name) for name in required['required_programs']}


def download_verified(url, expected, path, maximum):
    if not url.startswith('https://static.rust-lang.org/dist/'):
        raise ValueError('Toolchain download host is not allowed')
    if path.exists():
        if path.is_symlink() or not path.is_file() or path.stat().st_size > maximum or sha(path) != expected:
            raise ValueError('Existing download differs from pinned artifact')
        return
    temporary = path.with_suffix(path.suffix + '.partial')
    total = 0
    with urllib.request.urlopen(url, timeout=60) as response, temporary.open('xb') as output:
        if not response.geturl().startswith('https://static.rust-lang.org/dist/'):
            raise ValueError('Toolchain download redirected outside official host')
        for block in iter(lambda: response.read(1024 * 1024), b''):
            total += len(block)
            if total > maximum:
                raise ValueError('Toolchain download exceeds resource bound')
            output.write(block)
    if sha(temporary) != expected:
        raise ValueError('Official toolchain artifact SHA256 mismatch')
    os.link(temporary, path)
    temporary.unlink()


def extract_component(archive, destination, prefix):
    """Only directories and regular files from a separately hash-verified tarball."""
    destination.mkdir(exist_ok=False)
    seen, total = set(), 0
    with tarfile.open(archive, 'r:xz') as tar:
        for member in tar:
            name = relative(member.name)
            if name.parts[0] != prefix or str(name) in seen or not (member.isfile() or member.isdir()):
                raise ValueError('Unsafe toolchain archive entry')
            seen.add(str(name))
            total += member.size
            if len(seen) > 100000 or member.size > 512 * 1024 * 1024 or total > 2 * 1024 * 1024 * 1024:
                raise ValueError('Toolchain archive exceeds resource bound')
            path = destination.joinpath(*name.parts)
            if member.isdir():
                path.mkdir(parents=True, exist_ok=True)
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open('xb') as output:
                    shutil.copyfileobj(tar.extractfile(member), output)
                path.chmod(0o755 if member.mode & 0o111 else 0o644)
    return destination / prefix


def bootstrap(args):
    linux_worker()
    compilers = compiler_preflight()
    required = requirements()
    work = args.work_root.resolve() / 'native/toolchain'
    work.mkdir(parents=True, exist_ok=True)
    prefix = args.rust_prefix.absolute()
    if prefix.is_symlink() or prefix.exists():
        raise FileExistsError('Rust prefix must be new; preserve existing toolchain installations')
    downloads = work / 'downloads'
    downloads.mkdir(exist_ok=True)
    manifest = downloads / 'channel-rust-1.97.1.toml'
    download_verified(required['manifest_url'], required['manifest_sha256'], manifest, 2 * 1024 * 1024)
    official = tomllib.loads(manifest.read_text())
    if official['date'] != required['manifest_date']:
        raise ValueError('Official release date mismatch')
    records = []
    for component in ('rustc', 'rust-std', 'cargo'):
        artifact = required['components'][component]
        published = official['pkg'][component]['target'][required['target']]
        if (published['xz_url'], published['xz_hash']) != (artifact['url'], artifact['sha256']):
            raise ValueError('Component differs from pinned official release manifest')
        archive = downloads / artifact['url'].rsplit('/', 1)[1]
        download_verified(artifact['url'], artifact['sha256'], archive, 512 * 1024 * 1024)
        source = extract_component(archive, work / ('extract-' + component), archive.name.removesuffix('.tar.xz'))
        command = ['sh', str(source / 'install.sh'), '--prefix=' + str(prefix), '--disable-ldconfig']
        log = work / (component + '-install.log')
        started = time.monotonic()
        with log.open('xb') as output:
            result = subprocess.run(command, cwd=source, env=clean_environment(), stdout=output,
                                    stderr=subprocess.STDOUT, timeout=300)
        record = {'component': component, 'archive_sha256': artifact['sha256'], 'archive_bytes': archive.stat().st_size,
                  'installer_sha256': sha(source / 'install.sh'), 'argv': command, 'exit_code': result.returncode,
                  'seconds': time.monotonic() - started, 'log_sha256': sha(log)}
        records.append(record)
        if result.returncode:
            write_json(work / 'failure.json', {'requirements_sha256': REQUIREMENTS_SHA, 'components': records})
            raise RuntimeError(f'Official {component} installer failed; inspect {log}')
    installed = {}
    for name in ('rustc', 'cargo'):
        path = prefix / 'bin' / name
        version = subprocess.check_output([str(path), '--version'], text=True, env=clean_environment()).strip()
        if version.split()[1] != RUST:
            raise ValueError('Installed Rust version mismatch')
        installed[name] = {'path': str(path), 'version': version, 'sha256': sha(path)}
    receipt = {'schema': 'ruvllm-linux-native-bootstrap/v1', 'requirements_sha256': REQUIREMENTS_SHA,
               'manifest_sha256': sha(manifest), 'rust_prefix': str(prefix), 'components': records,
               'installed': installed, 'compiler_preflight': compilers, 'linux_bootstrap_verified': True,
               'production_authorized': False}
    write_json(work / 'bootstrap-receipt.json', receipt)
    print(json.dumps(receipt, indent=2))


def artifact_manifest(work):
    """Small proof artifacts and executable only; Cargo cache and checkpoints stay out."""
    names = ['source-verification.json', 'build-receipt.json', 'patch-receipt.json', 'patch-run.json',
             'rollback-receipt.json', 'rollback-run.json', 'target/release/examples/weight_evolution']
    names += ['build-logs/unit.log', 'build-logs/release.log', 'toolchain/bootstrap-receipt.json',
              'toolchain/cargo-install.log', 'toolchain/rustc-install.log', 'toolchain/rust-std-install.log']
    records = {}
    for name in names:
        path = work / name
        if path.exists():
            if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_FILE:
                raise ValueError('Native evidence artifact is not a bounded regular file')
            records[name] = {'sha256': sha(path), 'bytes': path.stat().st_size}
    result = {'schema': 'ruvllm-linux-native-artifacts/v1', 'source_commit': COMMIT,
              'bootstrap_requirements_sha256': REQUIREMENTS_SHA, 'files': records,
              'production_authorized': False}
    return result


def focused_test_counts(log):
    matches = re.findall(r'^test result: ok\. (\d+) passed; (\d+) failed; (\d+) ignored;',
                         log, flags=re.MULTILINE)
    if matches != [('17', '0', '0')]:
        raise ValueError('Expected exactly 17 passing native tests with no ignored tests')
    return {'passed': 17, 'failed': 0, 'ignored': 0}


def add_file(tar, name, data, mode=0o644):
    info = tarfile.TarInfo(name)
    info.size = len(data)
    info.mode = mode
    info.mtime = 0
    tar.addfile(info, io.BytesIO(data))


def package(args):
    repo, bundle = args.source_repo.resolve(), args.bundle.resolve()
    if git(repo, 'rev-parse', COMMIT + '^{tree}').decode().strip() != TREE:
        raise ValueError('Native source tree is not the frozen implementation')
    pins = read_json(bundle / 'source-pins.json')
    if pins['native_base'] != BASE or pins['native_commit'] != COMMIT:
        raise ValueError('Wrong research source bundle')
    changes = (bundle / 'source/native-changes.patch').read_bytes()
    if changes != git(repo, 'diff', '--binary', BASE, COMMIT):
        raise ValueError('Research bundle patch differs from frozen committed changes')
    cargo = tomllib.loads(git(repo, 'show', COMMIT + ':Cargo.toml').decode())
    paths = ['Cargo.toml', 'Cargo.lock', 'LICENSE', 'README.md', '.cargo', 'crates', 'patches']
    paths += [name for name in cargo['workspace']['members'] if not name.startswith('crates/')]
    paths = sorted(set(paths))
    output = args.output.absolute()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() or output.with_suffix(output.suffix + '.json').exists():
        raise FileExistsError('Native source archive or receipt already exists')
    files = {}
    total = 0
    command = ['git', '-C', str(repo), 'archive', '--format=tar', COMMIT, *paths]
    child = subprocess.Popen(command, stdout=subprocess.PIPE, env=clean_environment())
    try:
        with output.open('xb') as raw, gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0) as zipped:
            with tarfile.open(fileobj=zipped, mode='w', format=tarfile.PAX_FORMAT) as target:
                with tarfile.open(fileobj=child.stdout, mode='r|') as source:
                    for member in source:
                        if member.isdir():
                            continue
                        name = str(relative(member.name))
                        if not member.isfile() or member.size > MAX_FILE or name in files:
                            raise ValueError(f'Unsupported source entry: {name}')
                        total += member.size
                        if total > MAX_SOURCE or len(files) >= 20000:
                            raise ValueError('Source projection exceeds resource bound')
                        data = source.extractfile(member).read()
                        mode = member.mode & 0o777
                        if mode not in (0o644, 0o664, 0o755, 0o775):
                            raise ValueError(f'Unexpected source mode: {name}')
                        mode = 0o755 if mode & 0o111 else 0o644
                        add_file(target, name, data, mode)
                        files[name] = {'sha256': sha_bytes(data), 'bytes': len(data), 'mode': mode}
                if child.wait() != 0:
                    raise RuntimeError('git archive failed')
                name = 'provenance/native-changes.patch'
                add_file(target, name, changes)
                files[name] = {'sha256': sha_bytes(changes), 'bytes': len(changes), 'mode': 0o644}
                manifest = {'schema': 'ruvllm-linux-native-source/v1', 'base_commit': BASE, 'source_commit': COMMIT,
                            'source_tree': TREE, 'source_projection': paths, 'files': files,
                            'patch_sha256': sha_bytes(changes), 'rust_toolchain': RUST,
                            'workspace_declared_rust_version': cargo['workspace']['package']['rust-version'],
                            'observed_selected_dependency_minimum': '1.85',
                            'linux_minimum_rust_tested': False,
                            'build_features': ['async-runtime', 'weight-evolution'],
                            'default_features': False, 'native_inference_included': False,
                            'model_scope': 'Qwen3.5-0.8B published sparse patch only; no cross-model weight transfer'}
                manifest_bytes = (json.dumps(manifest, sort_keys=True, indent=2) + '\n').encode()
                add_file(target, MANIFEST, manifest_bytes)
    finally:
        if child.poll() is None:
            child.terminate()
            child.wait()
    receipt = {'schema': 'ruvllm-linux-native-archive/v1', 'archive': str(output), 'sha256': sha(output),
               'bytes': output.stat().st_size, 'manifest_sha256': sha_bytes(manifest_bytes),
               'source_commit': COMMIT, 'source_tree': TREE, 'source_files': len(files),
               'source_bytes': sum(record['bytes'] for record in files.values()),
               'linux_build_verified': False, 'public_git_objects_only': True}
    write_json(output.with_suffix(output.suffix + '.json'), receipt)
    print(json.dumps(receipt, indent=2))


def source_root(work):
    return work.resolve() / 'native/source'


def verify_source(root):
    manifest = read_json(root / MANIFEST)
    if (manifest.get('source_commit'), manifest.get('source_tree'), manifest.get('base_commit')) != (COMMIT, TREE, BASE):
        raise ValueError('Source provenance mismatch')
    files = manifest['files']
    actual = set()
    total = 0
    for path in root.rglob('*'):
        if path.is_symlink():
            raise ValueError('Source projection contains a symlink')
        if not (path.is_file() or path.is_dir()):
            raise ValueError('Source projection contains a nonregular entry')
        if path.is_file():
            name = path.relative_to(root).as_posix()
            actual.add(name)
            if name == MANIFEST:
                continue
            expected = files.get(name)
            if not expected or not stat.S_ISREG(path.stat().st_mode) or path.stat().st_size != expected['bytes']:
                raise ValueError(f'Unexpected source file: {name}')
            total += expected['bytes']
            if total > MAX_SOURCE or expected['bytes'] > MAX_FILE:
                raise ValueError('Source byte bound exceeded')
            if sha(path) != expected['sha256'] or path.stat().st_mode & 0o7777 != expected['mode']:
                raise ValueError(f'Source bytes or mode differ: {name}')
    if actual != set(files) | {MANIFEST}:
        raise ValueError('Source projection is incomplete')
    return {'source_commit': COMMIT, 'source_tree': TREE, 'manifest_sha256': sha(root / MANIFEST),
            'verified_files': len(files), 'verified_bytes': total}


def extract(args):
    archive = args.archive.resolve()
    if archive.stat().st_size > MAX_ARCHIVE or sha(archive) != args.expected_sha256:
        raise ValueError('Native source archive digest or size mismatch')
    root = source_root(args.work_root)
    root.mkdir(parents=True, exist_ok=False)
    seen = set()
    total = 0
    with tarfile.open(archive, 'r:gz') as tar:
        for member in tar:
            name = str(relative(member.name))
            if not member.isfile() or name in seen or member.size > MAX_FILE:
                raise ValueError(f'Unsafe source archive member: {name}')
            seen.add(name)
            total += member.size
            if len(seen) > 20001 or total > MAX_SOURCE + MAX_FILE:
                raise ValueError('Source archive resource bound exceeded')
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('xb') as stream:
                shutil.copyfileobj(tar.extractfile(member), stream)
            path.chmod(member.mode & 0o777)
    result = verify_source(root)
    result.update(schema='ruvllm-linux-native-source-verification/v1', archive_sha256=args.expected_sha256)
    write_json(args.work_root.resolve() / 'native/source-verification.json', result)
    print(json.dumps(result, indent=2))


def build(args):
    linux_worker()
    compilers = compiler_preflight()
    root = source_root(args.work_root)
    binding = verify_source(root)
    if binding['manifest_sha256'] != requirements()['source_manifest_sha256']:
        raise ValueError('Source manifest differs from verified preparation')
    env = clean_environment()
    if args.rust_prefix:
        prefix = args.rust_prefix.absolute()
        env['PATH'] = str(prefix / 'bin') + os.pathsep + env.get('PATH', '')
        env['RUSTC'] = str(prefix / 'bin/rustc')
    rustc = subprocess.check_output(['rustc', '--version'], text=True, env=env).strip()
    cargo = subprocess.check_output(['cargo', '--version'], text=True, env=env).strip()
    if rustc.split()[1] != RUST or cargo.split()[1] != RUST:
        raise ValueError(f'Install pinned Rust {RUST} first, for example rustup toolchain install {RUST} --profile minimal')
    work = args.work_root.resolve() / 'native'
    logs = work / 'build-logs'
    logs.mkdir(exist_ok=False)
    env['CARGO_TARGET_DIR'] = str(work / 'target')
    env['CARGO_BUILD_JOBS'] = str(args.jobs)
    env['RUST_MIN_STACK'] = '16777216'
    common = ['-p', 'ruvllm', '--locked', '--config', 'registries.crates-io.protocol="sparse"',
              '--no-default-features', '--features', 'async-runtime,weight-evolution']
    commands = [('unit', ['cargo', 'test', *common, '--lib', 'weight_evolution::tests']),
                ('release', ['cargo', 'build', '--release', *common, '--example', 'weight_evolution'])]
    records = []
    for label, command in commands:
        path = logs / (label + '.log')
        started = time.monotonic()
        with path.open('xb') as output:
            result = subprocess.run(command, cwd=root, env=env, stdout=output, stderr=subprocess.STDOUT, timeout=args.timeout_seconds)
        records.append({'name': label, 'argv': command, 'exit_code': result.returncode,
                        'seconds': time.monotonic() - started, 'log_sha256': sha(path)})
        if result.returncode:
            write_json(work / 'build-failure.json', {'source': binding, 'commands': records, 'linux_build_verified': False})
            raise RuntimeError(f'Native {label} failed; inspect {path}')
        if label == 'unit':
            try:
                records[-1]['tests'] = focused_test_counts(path.read_text(errors='replace'))
            except ValueError:
                write_json(work / 'build-failure.json', {'source': binding, 'commands': records,
                           'linux_build_verified': False})
                raise
    verify_source(root)
    binary = work / 'target/release/examples/weight_evolution'
    receipt = {'schema': 'ruvllm-linux-native-build/v1', 'source': binding, 'commands': records,
               'rustc': rustc, 'cargo': cargo, 'platform': platform.platform(),
               'binary': str(binary), 'binary_sha256': sha(binary), 'jobs': args.jobs,
               'compiler_preflight': compilers, 'focused_tests_passed': 17,
               'linux_build_verified': True, 'production_authorized': False}
    write_json(work / 'build-receipt.json', receipt)
    write_json(work / 'artifacts-build.json', artifact_manifest(work))
    print(json.dumps(receipt, indent=2))


def native(work, request):
    receipt = read_json(work / 'native/build-receipt.json')
    binary = Path(receipt['binary'])
    if not receipt['linux_build_verified'] or sha(binary) != receipt['binary_sha256']:
        raise ValueError('Linux native executable differs from build receipt')
    started = time.monotonic()
    result = subprocess.run([str(binary)], input=json.dumps(request), text=True, capture_output=True,
                            env=clean_environment(), timeout=300)
    if result.returncode:
        raise RuntimeError(result.stderr[:8192])
    if sha(binary) != receipt['binary_sha256']:
        raise ValueError('Linux native executable changed during operation')
    return json.loads(result.stdout), time.monotonic() - started, receipt['binary_sha256']


def weights(args, rollback=False):
    work = args.work_root.resolve()
    base = args.source_checkpoint.resolve()
    patch, policy = work / 'inputs/patch.json', work / 'inputs/policy.json'
    if sha(patch) != PATCH or sha(policy) != POLICY or sha(base) != BASE_MODEL:
        raise ValueError('Only the exact published Qwen3.5-0.8B source, patch and policy are allowed')
    label = 'rollback' if rollback else 'patch'
    output = work / 'models' / ('qwen035-08b-restored' if rollback else 'qwen035-08b-edited') / base.name
    output.parent.mkdir(parents=True, exist_ok=True)
    source = work / 'models/qwen035-08b-edited' / base.name if rollback else base
    request = {'op': label, 'source': str(source), 'output': str(output), 'patch': str(patch), 'policy': str(policy)}
    if rollback:
        request['receipt'] = str(work / 'native/patch-receipt.json')
    if output.exists() or (work / 'native' / (label + '-receipt.json')).exists():
        raise FileExistsError('Refusing to overwrite native experiment evidence')
    result, seconds, binary_hash = native(work, request)
    expected = BASE_MODEL if rollback else CANDIDATE_MODEL
    if result['candidate_sha256'] != expected or sha(output) != expected or sha(base) != BASE_MODEL:
        raise ValueError('Native candidate, rollback or preserved source SHA mismatch')
    if rollback and subprocess.run(['cmp', '-s', str(base), str(output)], env=clean_environment()).returncode != 0:
        raise ValueError('Rollback byte comparison failed')
    write_json(work / 'native' / (label + '-receipt.json'), result)
    write_json(work / 'native' / (label + '-run.json'), {'schema': 'ruvllm-linux-native-operation/v1',
               'request': request, 'seconds': seconds, 'native_binary_sha256': binary_hash,
               'source_sha256': BASE_MODEL, 'output_sha256': expected, 'source_unchanged': True,
               'byte_exact_rollback': rollback, 'production_authorized': False})
    write_json(work / 'native' / ('artifacts-' + label + '.json'), artifact_manifest(work / 'native'))
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='operation', required=True)
    p = commands.add_parser('package')
    p.add_argument('--source-repo', type=Path, required=True)
    p.add_argument('--bundle', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p = commands.add_parser('extract')
    p.add_argument('--archive', type=Path, required=True)
    p.add_argument('--expected-sha256', required=True)
    p.add_argument('--work-root', type=Path, required=True)
    p = commands.add_parser('requirements')
    p = commands.add_parser('bootstrap')
    p.add_argument('--work-root', type=Path, required=True)
    p.add_argument('--rust-prefix', type=Path, default=Path('/workspace/rust'))
    for name in ['verify-source', 'build', 'apply', 'rollback']:
        p = commands.add_parser(name)
        p.add_argument('--work-root', type=Path, required=True)
        if name in ('apply', 'rollback'):
            p.add_argument('--source-checkpoint', type=Path, required=True)
        if name == 'build':
            p.add_argument('--rust-prefix', type=Path, default=Path('/workspace/rust'))
            p.add_argument('--jobs', type=int, default=4, choices=range(1, 33))
            p.add_argument('--timeout-seconds', type=int, default=1200, choices=range(60, 3601))
    args = parser.parse_args()
    if args.operation == 'package': package(args)
    elif args.operation == 'extract': extract(args)
    elif args.operation == 'verify-source': print(json.dumps(verify_source(source_root(args.work_root)), indent=2))
    elif args.operation == 'build': build(args)
    elif args.operation == 'bootstrap': bootstrap(args)
    elif args.operation == 'requirements': print(json.dumps({'sha256': REQUIREMENTS_SHA, 'requirements': requirements()}, indent=2))
    else: weights(args, rollback=args.operation == 'rollback')


if __name__ == '__main__':
    main()
