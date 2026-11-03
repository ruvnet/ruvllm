#!/usr/bin/env python3
"""Verify a completed cloud publication offline without executing archive code.

Requires Python 3.12 and the project's jsonschema==4.26.0 dependency for importing
the trusted timing helper. No inference, generated code, provider or network calls.
Pin a trusted repository commit; unsigned manifests do not authenticate themselves.
"""
import argparse
from datetime import datetime
import gzip
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat
import sys
import tarfile
import tempfile


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = 'cf83995f29b5c3f7c28871d87d1295839bd94452dec0b8c352b990ba52a26cb6'
FILES = {'summary.json', 'scored.json', 'REPORT.md', 'timing.json', 'closeout.json'}
MAX_FILE = 256 * 1024 * 1024
MAX_TOTAL = 1024 * 1024 * 1024
MAX_MEMBERS = 5000


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(value):
    require(isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value), 'Invalid SHA256')
    return value


def unique(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, 'Duplicate JSON field')
        result[key] = value
    return result


def loads(data):
    return json.loads(data, object_pairs_hook=unique,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError('Nonfinite JSON')))


def regular(path, limit=MAX_FILE):
    require(not Path(path).is_symlink(), 'Linked file is forbidden')
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        info = os.fstat(descriptor)
        require(stat.S_ISREG(info.st_mode) and 0 < info.st_size <= limit, 'Unbounded or nonregular file')
    except BaseException:
        os.close(descriptor)
        raise
    return os.fdopen(descriptor, 'rb'), info


def read_bytes(path, limit=MAX_FILE):
    stream, info = regular(path, limit)
    with stream:
        data = stream.read(limit + 1)
    require(len(data) == info.st_size, 'File changed while reading')
    return data


def sha(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def record(value, limit=MAX_FILE):
    require(isinstance(value, dict) and set(value) == {'bytes', 'sha256'}
            and type(value['bytes']) is int and 0 < value['bytes'] <= limit, 'Invalid file record')
    digest(value['sha256'])


class BoundedReader:
    """Count expanded gzip bytes, including tar metadata and padding."""
    def __init__(self, stream):
        self.stream, self.total = stream, 0

    def read(self, size):
        require(type(size) is int and 0 <= size <= MAX_TOTAL, 'Unbounded decompression read')
        data = self.stream.read(size)
        self.total += len(data)
        require(self.total <= MAX_TOTAL + MAX_MEMBERS * 4096, 'Expanded archive exceeds bound')
        return data


def extract_verified(archive, declaration, destination):
    require(Path(archive).name == declaration['filename'], 'Archive filename differs')
    stream, info = regular(archive, MAX_TOTAL)
    with stream:
        require(info.st_size == declaration['bytes'], 'Archive size differs')
        require(hashlib.file_digest(stream, 'sha256').hexdigest() == declaration['sha256'], 'Archive digest differs')
        stream.seek(0)
        seen, total = set(), 0
        with gzip.GzipFile(fileobj=stream, mode='rb') as compressed:
            expanded = BoundedReader(compressed)
            with tarfile.open(fileobj=expanded, mode='r|') as bundle:
                for member in bundle:
                    name = member.name
                    path = PurePosixPath(name)
                    require(name and '\\' not in name and '\x00' not in name and not path.is_absolute()
                            and str(path) == name and all(part not in ('.', '..') for part in path.parts)
                            and path.parts[0] == declaration['top_directory'], 'Unsafe archive path')
                    require(name not in seen and len(seen) < MAX_MEMBERS, 'Duplicate or excessive archive members')
                    seen.add(name)
                    require(member.isfile() and not member.issparse() and member.size >= 0
                            and member.size <= MAX_FILE and member.mode in (0o444, 0o555),
                            'Archive member must be a bounded regular evidence file')
                    total += member.size
                    require(total <= MAX_TOTAL, 'Extracted archive exceeds bound')
                    target = destination.joinpath(*path.parts)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    source = bundle.extractfile(member)
                    require(source is not None, 'Missing archive payload')
                    remaining = member.size
                    with target.open('xb') as output:
                        while remaining:
                            chunk = source.read(min(1024 * 1024, remaining))
                            require(chunk, 'Truncated archive member')
                            output.write(chunk)
                            remaining -= len(chunk)
                    target.chmod(member.mode)
            # Force gzip trailer validation and bound trailing expanded data too.
            while expanded.read(1024 * 1024):
                pass
        require(seen, 'Empty archive')
        after = os.fstat(stream.fileno())
        require((info.st_size, info.st_mtime_ns, info.st_ino) == (after.st_size, after.st_mtime_ns, after.st_ino),
                'Archive changed during verification')
    return destination / declaration['top_directory']


def trusted_bundle_check(root):
    if __package__:
        from .package_vast_evidence import verify_bundle
    else:
        from package_vast_evidence import verify_bundle
    return verify_bundle(root)


def trusted_timing(root):
    """Use local reviewed code with extracted data, restoring all altered globals."""
    science = ROOT / 'research/vast_cross_model'
    helper = science / 'analysis/timing.py'
    extracted = root / 'source/project/research/vast_cross_model'
    require(read_bytes(helper) == read_bytes(extracted / 'analysis/timing.py'), 'Timing implementation differs')
    previous_path = list(sys.path)
    previous = {key: sys.modules.get(key) for key in ('common', 'score')}
    module = score = None
    old_science = old_here = None
    try:
        sys.path.insert(0, str(ROOT))
        sys.path.insert(0, str(science))
        for key in previous:
            sys.modules.pop(key, None)
        spec = importlib.util.spec_from_file_location('_trusted_cloud_timing', helper)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        score = sys.modules['score']
        require(Path(score.__file__).resolve() == science / 'score.py'
                and Path(sys.modules['common'].__file__).resolve() == science / 'common.py', 'Untrusted helper import')
        old_science, old_here = module.SCIENCE, score.HERE
        module.SCIENCE, score.HERE = extracted, extracted
        return module.report(root / 'work', allow_partial=False)
    finally:
        if module is not None and old_science is not None:
            module.SCIENCE = old_science
        if score is not None and old_here is not None:
            score.HERE = old_here
        sys.path[:] = previous_path
        for key, value in previous.items():
            if value is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = value


def finite(value):
    require(type(value) in (int, float) and math.isfinite(value) and value >= 0, 'Invalid finite scalar')


def timestamp(value):
    require(isinstance(value, str) and len(value) <= 40, 'Invalid timestamp')
    parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
    require(parsed.tzinfo is not None, 'Timestamp lacks timezone')
    return parsed.timestamp()


def validate_closeout(value, manifest, history):
    """Narrow public checks adapted from project_vast_closeout.py at 1ae6edc.

    Private cleanup observations cannot be authenticated from public digests.
    """
    require(set(value) == {'schema', 'status', 'protocol_sha256', 'scientific_bundle_manifest_sha256', 'generated_at',
            'resource_binding_sha256', 'budget', 'estimate', 'cleanup', 'private_source_sha256', 'authority'}, 'Unexpected closeout field')
    require(value['schema'] == 'ruvllm.cloud-closeout.v1' and value['status'] == 'closed'
            and value['protocol_sha256'] == PROTOCOL and value['scientific_bundle_manifest_sha256'] == manifest,
            'Closeout does not bind the complete scientific bundle')
    digest(value['resource_binding_sha256'])
    budget, cost, cleanup = value['budget'], value['estimate'], value['cleanup']
    require(set(budget) == {'authorized_cap_micro_usd', 'total_reserved_micro_usd', 'this_run_reserved_micro_usd', 'reservations_are_invoices'}
            and all(type(budget[k]) is int for k in ('authorized_cap_micro_usd', 'total_reserved_micro_usd', 'this_run_reserved_micro_usd'))
            and 0 < budget['this_run_reserved_micro_usd'] <= budget['total_reserved_micro_usd'] <= budget['authorized_cap_micro_usd'] <= 200000000
            and budget['reservations_are_invoices'] is False, 'Invalid budget projection')
    numbers = {'elapsed_seconds', 'gpu_storage_hourly_usd', 'gpu_storage_elapsed_usd', 'download_allowance_gb',
        'download_rate_usd_per_gb', 'download_allowance_usd', 'upload_allowance_gb', 'upload_rate_usd_per_gb',
        'upload_allowance_usd', 'guard_reserved_usd'}
    require(set(cost) == numbers | {'rental_started_at', 'rental_absence_confirmed_at', 'transfer_allowances_are_enforced_caps',
        'transfer_actual_bytes', 'guard_actual_usd', 'invoice_usd', 'final_invoice_reconciled'}, 'Unexpected cost field')
    for key in numbers:
        finite(cost[key])
    require(all(cost[key] is None for key in ('transfer_actual_bytes', 'guard_actual_usd', 'invoice_usd'))
            and cost['final_invoice_reconciled'] is False and cost['transfer_allowances_are_enforced_caps'] is False,
            'Unknown costs must remain null; allowances are not enforced caps')
    seconds = timestamp(cost['rental_absence_confirmed_at']) - timestamp(cost['rental_started_at'])
    require(seconds > 0 and cost['gpu_storage_hourly_usd'] > 0 and math.isclose(seconds, cost['elapsed_seconds'], abs_tol=1e-5)
            and math.isclose(seconds / 3600 * cost['gpu_storage_hourly_usd'], cost['gpu_storage_elapsed_usd'], rel_tol=1e-9),
            'Quoted elapsed cost arithmetic differs')
    for direction in ('download', 'upload'):
        require(math.isclose(cost[direction + '_allowance_usd'], cost[direction + '_allowance_gb'] * cost[direction + '_rate_usd_per_gb']),
                'Transfer allowance arithmetic differs')
    flags = {'cleanup_verified', 'gpu_absence_verified', 'job_absent', 'service_account_absent', 'secret_binding_absent'}
    require(set(cleanup) == flags | {'independent_absence_reads', 'absence_observed_at', 'matching_owned_resources_remaining', 'completed_at'}
            and all(cleanup[k] is True for k in flags) and type(cleanup['independent_absence_reads']) is int
            and cleanup['independent_absence_reads'] == 2 and type(cleanup['matching_owned_resources_remaining']) is int
            and cleanup['matching_owned_resources_remaining'] == 0 and len(cleanup['absence_observed_at']) == 2,
            'Cleanup is incomplete')
    first, second = map(timestamp, cleanup['absence_observed_at'])
    require(timestamp(cost['rental_absence_confirmed_at']) <= first < second <= timestamp(cleanup['completed_at'])
            <= timestamp(value['generated_at']) and second - first >= 4.9, 'Cleanup timeline differs')
    required = {'destruction', 'cleanup_summary', 'run_spec', 'rental', 'guard_config', 'guard_deployment', 'guard_identity',
        'guard_readiness', 'guard_execution', 'cleanup_complete', 'cleanup_intent', 'absence_proof', 'absence_observation_1',
        'absence_observation_2', 'job_absence', 'identity_absence', 'grant_absence', 'grant_before', 'guard_stop',
        'continuation_intent', 'effective_quote', 'creation_intent', 'rental_observation', 'shared_budget', 'scientific_run_receipt'}
    allowed = required | {'quote_replan', 'guard_terminal'} | {f'execution_readback_{i}' for i in range(13)} | {
        f'historical_execution_{i}' for i in range(20)}
    require(isinstance(value['private_source_sha256'], dict) and required <= set(value['private_source_sha256']) <= allowed,
            'Missing or nonpublic source labels')
    for item in value['private_source_sha256'].values():
        digest(item)
    current = history['attempts'][-1]
    require(current['status'] == 'completed' and current['protocol_sha256'] == PROTOCOL
            and current['origin_receipt_sha256'] == value['private_source_sha256']['scientific_run_receipt']
            and timestamp(cost['rental_started_at']) <= timestamp(current['completed_at'])
            <= timestamp(cost['rental_absence_confirmed_at']), 'Scientific run receipt or rental timeline differs')
    require(value['authority'] == 'Operator receipt consistency only; no provider signature, live API check, invoice reconciliation or production activation.',
            'Unsupported closeout authority claim')


def verify(publication, archive):
    publication = Path(publication)
    if publication.is_dir():
        publication /= 'publication.json'
    metadata = loads(read_bytes(publication, 1024 * 1024))
    require(set(metadata) == {'schema', 'protocol_sha256', 'archive', 'scientific_bundle_manifest_sha256', 'files'}
            and metadata['schema'] == 'ruvllm.cloud-publication.v1' and metadata['protocol_sha256'] == PROTOCOL,
            'Unexpected publication schema or protocol')
    declaration = metadata['archive']
    require(set(declaration) == {'filename', 'bytes', 'sha256', 'top_directory'}, 'Unexpected archive metadata')
    require(isinstance(declaration['filename'], str) and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,120}\.tar\.gz', declaration['filename'])
            and isinstance(declaration['top_directory'], str) and re.fullmatch('[A-Za-z0-9][A-Za-z0-9_.-]{0,100}', declaration['top_directory']),
            'Invalid archive name')
    record({k: declaration[k] for k in ('bytes', 'sha256')}, MAX_TOTAL)
    digest(metadata['scientific_bundle_manifest_sha256'])
    require(set(metadata['files']) == FILES, 'Exactly five checked in cloud evidence files are required')
    files = {}
    for name in sorted(FILES):
        receipt = metadata['files'][name]
        record(receipt)
        data = read_bytes(publication.parent / name)
        require(len(data) == receipt['bytes'] and sha(data) == receipt['sha256'], 'Checked in file differs: ' + name)
        files[name] = data
    with tempfile.TemporaryDirectory(prefix='ruvllm-cloud-verify-') as temporary:
        root = extract_verified(archive, declaration, Path(temporary).resolve())
        proof = trusted_bundle_check(root)
        require(proof['read_only_verification'] is True and proof['protocol_sha256'] == PROTOCOL
                and proof['generations'] == 1851 and proof['manifest_sha256'] == metadata['scientific_bundle_manifest_sha256'],
                'Archive is not the declared complete scientific bundle')
        for name in ('summary.json', 'scored.json', 'REPORT.md'):
            require(read_bytes(root / 'results' / name) == files[name], 'Published result differs from archive: ' + name)
        timing = loads(files['timing.json'])
        require(canonical(timing) == canonical(trusted_timing(root)), 'Published timing differs from recomputed complete timing')
        validate_closeout(loads(files['closeout.json']), proof['manifest_sha256'],
                          loads(read_bytes(root / 'history/runtime-attempts.json')))
    return {'verified': True, 'protocol_sha256': PROTOCOL, 'generations': 1851, 'checked_in_files': 5,
        'archive_sha256': declaration['sha256'], 'scientific_bundle_manifest_sha256': proof['manifest_sha256'],
        'archive_code_executed': False, 'generated_code_executed': False, 'network_access': False,
        'private_cleanup_receipts_authenticated': False, 'invoice_reconciled': False,
        'trust': 'Consistency of locally supplied artifacts; pin the trusted repository and published archive digest independently.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--publication', type=Path, required=True)
    parser.add_argument('--archive', type=Path, required=True)
    args = parser.parse_args()
    try:
        print(json.dumps(verify(args.publication, args.archive), indent=2))
    except (OSError, EOFError, ValueError, KeyError, TypeError, AttributeError, IndexError, tarfile.TarError, ImportError) as error:
        print(json.dumps({'verified': False, 'error': type(error).__name__, 'reason': str(error)}), file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
