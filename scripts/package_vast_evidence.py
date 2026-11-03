#!/usr/bin/env python3
"""Build or verify an immutable public research evidence bundle. Standard library only."""
import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import tarfile

NATIVE_COMMIT = 'f5c3444968ede44e43744460192198962ae94da2'
NATIVE_BASE = '6a879a726b0d1d63e70a1ddb656263d8e6bab8a0'
NATIVE_TREE = 'f0a37eeeaf7c515ca6724057536e308112d4ea75'
NATIVE_ARCHIVE = 'f5d093c47c68501ccd76b477e2dd8133fc868cd37dbc2dd9dbfda8ff82b66caf'
NATIVE_MANIFEST = '731b39197b09c1f606b2dd53ecb3f888e51573e47e576fe8513c2dd45cf2af40'
LINUX_BINARY = '964165fa58154463991b6b7bfe08a5b4d5484cb6967b4fe2a9e7bfbe443f1d5e'
BASE_MODEL = '04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696'
EDITED_MODEL = 'bd5c223dc7d0ed48a6857f798a94da6ec46c40b9ff4e7bc31872dba10f943ae5'
MAX_FILE = 256 * 1024 * 1024
MAX_TOTAL = 1024 * 1024 * 1024
MAX_FILES = 5000
RESEARCH = 'research/vast_cross_model/'
ARM_IDS = ('qwen035-08b-base-direct', 'qwen035-08b-edited-direct', 'qwen035-4b-base-direct',
           'qwen035-4b-lora-direct', 'qwen3-coder-30b-direct', 'qwen035-08b-base-thinking',
           'qwen035-08b-edited-thinking')
VARIANTS = ('qwen035-08b-base', 'qwen035-08b-edited', 'qwen035-4b-base', 'qwen035-4b-lora', 'qwen3-coder-30b')
PREFLIGHT = ('qwen035-08b-base-direct', 'qwen3-coder-30b-direct')
RESEARCH_FILES = ('protocol.json', 'freeze.json', 'models.json', 'assets.json', 'cases.json', 'prompts.json',
                  'common.py', 'prepare.py', 'models_prepare.py', 'worker.py', 'run_all.py', 'score.py',
                  'test_protocol.py', 'README.md', 'THIRD_PARTY_NOTICES.md',
                  'licenses/HumanEval.LICENSE', 'licenses/LICENSE-research.txt',
                  'native/prepare.py', 'native/bootstrap-requirements.json', 'native/test_prepare.py',
                  'analysis/timing.py', 'analysis/test_timing.py', 'analysis/README.md')
ABANDONED_FILES = ('abandonment.json', 'protocol.json', 'freeze.json', 'worker.py', 'test_protocol.py', 'prepare.py',
                   'tokenizer-signature.json', 'eos-runtime-evidence.json', 'preflight-failure.log')
SOURCE_FILES = tuple(RESEARCH + name for name in RESEARCH_FILES) + tuple(
    RESEARCH + 'archive/unused-runtime-attempt-01/' + name for name in ABANDONED_FILES) + tuple(
    RESEARCH + 'archive/unused-tokenizer-only-freeze/' + name for name in
    ('abandonment.json', 'protocol.json', 'freeze.json', 'worker.py', 'test_protocol.py', 'prepare.py')) + tuple(
    RESEARCH + 'archive/unused-freeze-01/' + name for name in ('abandonment.json', 'protocol.json', 'freeze.json')) + (
    'ruflo_model/__init__.py', 'ruflo_model/scoring.py', 'pyproject.toml', 'LICENSE',
    'scripts/vast_worker_pipeline.py', 'scripts/vast_native_prepare.sh', 'scripts/package_vast_evidence.py',
    'scripts/native_dependency_notices.py', 'tests/test_package_vast_evidence.py', 'tests/test_vast_worker_pipeline.py')
NATIVE_FILES = ('source-verification.json', 'build-receipt.json', 'patch-receipt.json', 'rollback-receipt.json',
                'patch-run.json', 'rollback-run.json', 'build-logs/unit.log', 'build-logs/release.log',
                'target/release/examples/weight_evolution')
ARM_FILES = ('binding.json', 'runtime.json', 'complete.json', 'state.json', 'responses.jsonl')
RESULT_FILES = ('summary.json', 'scored.json', 'REPORT.md')
SECRET_MARKERS = re.compile(rb'-----BEGIN (?:[A-Z ]+ )?PRIVATE KEY-----|https?://[^/\s"\x27]+:[^/@\s"\x27]+@')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite_number(value, *, positive=False):
    return type(value) in (int, float) and math.isfinite(value) and (value > 0 if positive else value >= 0)


def relative(name):
    require(isinstance(name, str) and name and '\\' not in name and '\x00' not in name, 'Invalid relative path')
    path = PurePosixPath(name)
    require(not path.is_absolute() and str(path) == name and all(part not in ('.', '..') for part in path.parts),
            'Path traversal or normalization is forbidden')
    return path


def safe_path(root, name):
    path = root
    for component in relative(name).parts:
        path = path / component
        require(not path.is_symlink(), 'Symlink in selected public path: ' + name)
    return path


def open_regular(path):
    descriptor = os.open(path, os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_NONBLOCK', 0))
    info = os.fstat(descriptor)
    if not stat.S_ISREG(info.st_mode) or not 0 <= info.st_size <= MAX_FILE:
        os.close(descriptor)
        raise ValueError('Expected a bounded regular file: ' + str(path))
    os.set_blocking(descriptor, True)
    return os.fdopen(descriptor, 'rb'), info


def sha(path):
    stream, _ = open_regular(path)
    with stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def unique(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, 'Duplicate JSON object key')
        result[key] = value
    return result


def reject_constant(value):
    raise ValueError('Nonfinite JSON constant: ' + value)


def read(path):
    stream, _ = open_regular(path)
    with stream:
        return json.load(stream, object_pairs_hook=unique, parse_constant=reject_constant)


def rows(path):
    stream, _ = open_regular(path)
    with stream:
        result = []
        for line in stream:
            require(len(line) <= 2 * 1024 * 1024 and line.strip(), 'Invalid or oversized response row')
            result.append(json.loads(line, object_pairs_hook=unique, parse_constant=reject_constant))
            require(len(result) <= 2000, 'Too many generation rows')
        return result


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def object_sha(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as output:
        json.dump(value, output, sort_keys=True, indent=2, allow_nan=False)
        output.write('\n')


def protocol_inputs(project):
    research = project / RESEARCH
    protocol = read(research / 'protocol.json')
    digest = sha(research / 'protocol.json')
    require(read(research / 'freeze.json')['protocol_sha256'] == digest, 'Active freeze differs from protocol')
    require(protocol['total_generations'] == 1851 and [a['id'] for a in protocol['arms']] == list(ARM_IDS),
            'This package requires the complete seven arm, 1851 generation experiment')
    for group in ('input_sha256', 'implementation_sha256'):
        for name, expected in protocol[group].items():
            path = (research / name).resolve()
            require(path.is_relative_to(project), 'Frozen source escapes project')
            require(sha(path) == expected, 'Frozen source or input changed: ' + name)
    require(sha(project / 'ruflo_model/scoring.py') == protocol['implementation_sha256']['../../ruflo_model/scoring.py'],
            'Sandbox scorer differs from frozen source')
    return protocol, digest


def verify_native(work, assets):
    native = work / 'native'
    build = read(native / 'build-receipt.json')
    require(build['linux_build_verified'] is True and build['production_authorized'] is False,
            'Native Linux build was not verified')
    require(build['source']['source_commit'] == NATIVE_COMMIT and build['source']['source_tree'] == NATIVE_TREE
            and build['source']['manifest_sha256'] == NATIVE_MANIFEST, 'Native source differs')
    require(build['binary_sha256'] == LINUX_BINARY and sha(native / 'target/release/examples/weight_evolution') == LINUX_BINARY,
            'Native Linux executable differs')
    require(build['focused_tests_passed'] == 17 and len(build['commands']) == 2, 'Missing 17 focused native tests')
    for command, expected in zip(build['commands'], ('unit', 'release'), strict=True):
        require(command['name'] == expected and command['exit_code'] == 0, 'Native build command failed')
        require(command['log_sha256'] == sha(native / 'build-logs' / (expected + '.log')), 'Native log digest differs')
        if expected == 'unit':
            require(command['tests'] == {'passed': 17, 'failed': 0, 'ignored': 0}, 'Native unit counts differ')
            log = (native / 'build-logs/unit.log').read_text()
            require(re.findall(r'^test result: ok\. (\d+) passed; (\d+) failed; (\d+) ignored;', log, re.M)
                    == [('17', '0', '0')], 'Native unit log does not show exactly 17 passing tests')
    source = read(native / 'source-verification.json')
    require(source['archive_sha256'] == NATIVE_ARCHIVE and source['manifest_sha256'] == NATIVE_MANIFEST,
            'Native extraction binding differs')
    for operation, original, candidate in (('patch', BASE_MODEL, EDITED_MODEL), ('rollback', EDITED_MODEL, BASE_MODEL)):
        receipt = read(native / (operation + '-receipt.json'))
        require(receipt['source_sha256'] == original and receipt['candidate_sha256'] == candidate
                and receipt['patch_sha256'] == assets['patch.json'] and receipt['policy_sha256'] == assets['policy.json'],
                'Native patch or restoration binding differs')
        require(receipt['activation_authorized'] is False and receipt['changed_values'] == 130188,
                'Unexpected native authority or patch size')
        run = read(native / (operation + '-run.json'))
        require(run['source_sha256'] == BASE_MODEL and run['output_sha256'] == candidate and run['source_unchanged'] is True
                and run['native_binary_sha256'] == LINUX_BINARY and run['production_authorized'] is False,
                'Native operation proof differs')
        require(run['byte_exact_rollback'] is (operation == 'rollback'), 'Missing byte comparison proof')
    return {'source_commit': NATIVE_COMMIT, 'binary_sha256': LINUX_BINARY, 'focused_tests_passed': 17,
            'candidate_sha256': EDITED_MODEL, 'restored_sha256': BASE_MODEL}


def manifests(work, project, assets):
    pins = {item['key']: item for item in read(project / RESEARCH / 'models.json')}
    result = {key: read(work / 'models' / (key + '.manifest.json')) for key in VARIANTS}
    for key, manifest in result.items():
        require(manifest['variant'] == key and object_sha(manifest['files']) == manifest['sha256'], 'Model tree binding differs')
        for name, digest in manifest['files'].items():
            relative(name)
            require(re.fullmatch('[0-9a-f]{64}', digest), 'Invalid model file digest')
        if key in pins:
            require(manifest['source'] == pins[key], 'Original model revision differs')
            require(all(manifest['files'].get(name) == digest for name, digest in pins[key]['weights_sha256'].items()),
                    'Original weight digest differs')
    base, edited = result['qwen035-08b-base'], result['qwen035-08b-edited']
    changed = {name for name in set(base['files']) | set(edited['files']) if base['files'].get(name) != edited['files'].get(name)}
    weight = 'model.safetensors-00001-of-00001.safetensors'
    require(changed == {weight} and edited['files'][weight] == EDITED_MODEL and edited['base_sha256'] == base['sha256'],
            'Native model changed unexpected files')
    require(edited['patch_sha256'] == assets['patch.json'] and edited['policy_sha256'] == assets['policy.json'],
            'Native manifest input differs')
    base, edited = result['qwen035-4b-base'], result['qwen035-4b-lora']
    require(edited['base_sha256'] == base['sha256'] and edited['adapter_sha256'] == assets['adapter_model.safetensors'],
            'LoRA lineage differs')
    targets = {f'model.language_model.layers.{n}.mlp.down_proj.weight' for n in range(16, 32)}
    require(len(edited['changed_tensors']) == 16 and {item['tensor'] for item in edited['changed_tensors']} == targets
            and edited['untouched_tensor_hashes_verified'] > 0, 'LoRA tensor audit differs')
    require(set(base['files']) == set(edited['files']) and all(edited['files'][name] == digest
            for name, digest in base['files'].items() if not name.endswith('.safetensors')), 'LoRA modified configuration')
    return result


def arm_evidence(work, arm, protocol, protocol_sha, model_manifests, preflight=False):
    folder = work / ('preflight' if preflight else 'outputs') / arm['id']
    complete, binding, runtime, state = [read(folder / name) for name in ('complete.json', 'binding.json', 'runtime.json', 'state.json')]
    expected_binding = {'protocol_sha256': protocol_sha, 'variant': arm['variant'], 'arm': arm['id'],
                        'model_sha256': model_manifests[arm['variant']]['sha256'],
                        'model_manifest_sha256': sha(work / 'models' / (arm['variant'] + '.manifest.json')),
                        'runtime_sha256': object_sha(runtime)}
    require(binding == expected_binding and all(complete.get(k) == v for k, v in binding.items()), 'Arm binding differs')
    require(set(runtime) == {'versions', 'packages', 'python', 'platform', 'cuda', 'gpu', 'gpu_bytes', 'driver'},
            'Runtime contains unexpected metadata')
    require(all(runtime['versions'][key].split('+')[0] == value for key, value in protocol['runtime_versions'].items()),
            'Runtime package version differs')
    require(complete['preflight'] is preflight and complete['model_unchanged'] is True and state['status'] == 'completed',
            'Incomplete or misclassified generation arm')
    require(sha(folder / 'responses.jsonl') == complete['responses_sha256'], 'Raw output digest differs')
    data = rows(folder / 'responses.jsonl')
    require(len(data) == complete['rows'] == state['completed'], 'Generation counts differ')
    for row in data:
        require(all(row.get(k) == v for k, v in binding.items()) and row['thinking'] is arm['thinking'], 'Response binding differs')
        cap = 32 if preflight else protocol['panel_max_tokens'][row['panel']]
        require(row['finish_reason'] in ('stop', 'length') and row['max_tokens'] == cap
                and type(row['output_tokens']) is int and 0 < row['output_tokens'] <= cap
                and len(row['token_ids']) == row['output_tokens'], 'Output token accounting differs')
        require(all(type(token) is int and token >= 0 for token in row['token_ids']), 'Invalid token IDs')
        require(row['finish_reason'] != 'length' or row['output_tokens'] == cap, 'Truncation count differs')
    require(sum(row['output_tokens'] for row in data) == complete['output_tokens'], 'Arm token total differs')
    for name in ('load_seconds', 'generation_seconds'):
        require(type(complete[name]) in (int, float) and math.isfinite(complete[name]) and complete[name] >= 0,
                'Invalid measured timing')
    return data, complete, binding


def aggregate(scored):
    return {'count': len(scored), **{key: sum(row[key] for row in scored) for key in
            ('answer_correct', 'completed', 'delivered_correct', 'truncated', 'output_tokens')},
            'delivered_accuracy': sum(row['delivered_correct'] for row in scored) / len(scored)}


def paired(left, right):
    gains = sum(not a['delivered_correct'] and b['delivered_correct'] for a, b in zip(left, right, strict=True))
    losses = sum(a['delivered_correct'] and not b['delivered_correct'] for a, b in zip(left, right, strict=True))
    total = gains + losses
    probability = min(1.0, 2 * sum(math.comb(total, n) for n in range(min(gains, losses) + 1)) / 2 ** total) if total else 1.0
    return {'gains': gains, 'regressions': losses, 'net': gains - losses, 'exact_mcnemar_two_sided_p': probability,
            'multiple_testing_adjusted': False, 'exploratory': True}


def report_text(summary):
    lines = ['# Frozen cloud model comparison', '',
             f"Protocol `{summary['protocol_sha256']}`. {summary['generations']} generations. No candidate selection or promotion.", '',
             '| Arm | Panel | Delivered correct | Completed | Truncated | Output tokens |',
             '| :--- | :--- | ---: | ---: | ---: | ---: |']
    # The scorer iterates frozen protocol panel order, not JSON key order.
    panels = {name: ('math128',) if name.endswith('-thinking') else ('humaneval164', 'tools27', 'math128') for name in ARM_IDS}
    for name in ARM_IDS:
        for panel in panels[name]:
            row = summary['arms'][name]['panels'][panel]
            lines.append(f"| {name} | {panel} | {row['delivered_correct']}/{row['count']} | {row['completed']} | {row['truncated']} | {row['output_tokens']} |")
    lines += ['', '## Limits', ''] + summary['limitations']
    return '\n'.join(lines) + '\n'


def verify_evidence(project, work, results, inputs):
    protocol, protocol_sha = protocol_inputs(project)
    summary, scored = read(results / 'summary.json'), read(results / 'scored.json')
    require(summary['protocol_sha256'] == protocol_sha and summary['generations'] == 1851, 'Incomplete or wrong scored experiment')
    require(all(summary[key] is False for key in ('selection_use', 'sota_verified', 'production_qualified')), 'Unsupported research claim')
    require(summary['scorer_sha256'] == sha(project / RESEARCH / 'score.py')
            and summary['sandbox_scorer_sha256'] == sha(project / 'ruflo_model/scoring.py'), 'Scorer binding differs')
    require(set(summary['arms']) == set(scored) == set(ARM_IDS), 'Missing or unexpected scored arms')
    assets = read(project / RESEARCH / 'assets.json')
    require(set(assets) == {'patch.json', 'policy.json', 'adapter_config.json', 'adapter_model.safetensors'}, 'Unexpected model inputs')
    for name, expected in assets.items():
        require(sha(safe_path(inputs, name)) == expected, 'Public model input digest differs')
    native = verify_native(work, assets)
    model_manifests = manifests(work, project, assets)
    cases = read(project / RESEARCH / 'cases.json')
    require(len({case['id'] for case in cases}) == len(cases), 'Duplicate case IDs')
    count, tokens = 0, 0
    for arm in protocol['arms']:
        data, complete, binding = arm_evidence(work, arm, protocol, protocol_sha, model_manifests)
        expected = [case for case in cases if case['panel'] in arm['panels']]
        require([row['id'] for row in data] == [case['id'] for case in expected], 'Incomplete or reordered raw generations')
        grades = scored[arm['id']]
        require([row['id'] for row in grades] == [case['id'] for case in expected], 'Incomplete or reordered scoring')
        for row, grade, case in zip(data, grades, expected, strict=True):
            require(row['kind'] == case['kind'] and row['panel'] == grade['panel'] == case['panel'], 'Case classification differs')
            require(all(type(grade[key]) is bool for key in ('answer_correct', 'completed', 'delivered_correct', 'truncated')),
                    'Score flags must be Boolean')
            closed = (not row['thinking'] or '</think>' in row['text']) and row['text'].rfind('<think>') <= row['text'].rfind('</think>')
            require(grade['completed'] is (row['finish_reason'] == 'stop' and closed)
                    and grade['truncated'] is (row['finish_reason'] == 'length')
                    and grade['output_tokens'] == row['output_tokens'], 'Scored completion or token scalar differs')
            require(not grade['delivered_correct'] or (grade['completed'] and grade['answer_correct']), 'Impossible delivered score')
            if case['kind'] in ('code', 'tool'):
                require(grade['delivered_correct'] is (grade['completed'] and grade['answer_correct']),
                        'Delivered code or tool scalar differs')
        arm_summary = summary['arms'][arm['id']]
        require(arm_summary['binding'] == binding and arm_summary['responses_sha256'] == complete['responses_sha256'],
                'Summary does not bind actual raw responses')
        require(arm_summary['panels'] == {panel: aggregate([row for row in grades if row['panel'] == panel])
                for panel in arm['panels']}, 'Score aggregate differs')
        require(all(arm_summary[key] == complete[key] for key in ('load_seconds', 'generation_seconds')), 'Summary timing differs')
        count += len(data)
        tokens += complete['output_tokens']
    comparisons = {}
    for baseline, candidate in protocol['comparisons']:
        panels = next(arm['panels'] for arm in protocol['arms'] if arm['id'] == baseline)
        comparisons[candidate] = {panel: paired([row for row in scored[baseline] if row['panel'] == panel],
                                                [row for row in scored[candidate] if row['panel'] == panel]) for panel in panels}
    require(summary['paired'] == comparisons and summary['limitations'] == protocol['limitations'], 'Comparison or limitations differ')
    require((results / 'REPORT.md').read_text() == report_text(summary), 'Published report differs from scored summary')
    require(count == 1851 and tokens <= protocol['maximum_output_tokens'], 'Experiment row or token budget differs')
    for ident in PREFLIGHT:
        arm = next(arm for arm in protocol['arms'] if arm['id'] == ident)
        data, _, _ = arm_evidence(work, arm, protocol, protocol_sha, model_manifests, True)
        require(len(data) == 1 and data[0]['id'] == 'runtime-smoke-unscored', 'Preflight is not the unscored fixture')
    state = read(work / 'run-state.json')
    require(state['status'] == 'completed' and state['protocol_sha256'] == protocol_sha
            and state['generations'] == count and state['output_tokens'] == tokens, 'Supervisor completion differs')
    abandoned = project / RESEARCH / 'archive/unused-runtime-attempt-01'
    note = read(abandoned / 'abandonment.json')
    old_sha = sha(abandoned / 'protocol.json')
    require(note['status'] == 'abandoned_before_scored_generation' and note['protocol_sha256'] == old_sha
            and read(abandoned / 'freeze.json')['protocol_sha256'] == old_sha and old_sha != protocol_sha,
            'Missing abandoned runtime attempt provenance')
    return {'protocol_sha256': protocol_sha, 'generations': count, 'output_tokens': tokens, 'native': native,
            'production_qualified': False, 'sota_verified': False, 'selection_use': False,
            'answer_correctness_reexecuted': False, 'model_weights_included': False}


def selected_work_files():
    names = ['run-state.json'] + ['native/' + name for name in NATIVE_FILES]
    names += ['models/' + name + '.manifest.json' for name in VARIANTS]
    names += ['outputs/' + arm + '/' + name for arm in ARM_IDS for name in ARM_FILES]
    names += ['preflight/' + arm + '/' + name for arm in PREFLIGHT for name in ARM_FILES]
    names += ['preflight-abandoned-01/qwen035-08b-base-direct/' + name for name in ('binding.json', 'runtime.json', 'state.json')]
    return names


def history_projection(work, project):
    attempts = []
    for directory in ('pipeline', 'pipeline-continuation'):
        path = work / directory / 'receipt.json'
        receipt = read(path)
        attempt = {key: receipt[key] for key in ('protocol_sha256', 'pipeline_sha256', 'status', 'deadline_epoch',
                                                'completed_at', 'generated_code_execution', 'cloud_api_calls')}
        attempt.update(origin_receipt_sha256=sha(path), origin_receipt=directory + '/receipt.json', stages=[])
        for stage in receipt['stages']:
            attempt['stages'].append({key: stage[key] for key in
                                     ('stage', 'status', 'started_at', 'return_code', 'seconds', 'log_sha256')})
        if directory == 'pipeline-continuation':
            reused = receipt['reused_preparation']
            attempt['previous_receipt_sha256'] = reused['previous_receipt_sha256']
            attempt['scored_generations_before_repair'] = reused['scored_generations_before_repair']
        attempts.append(attempt)
    result = {'schema': 'ruflo.vast-public-runtime-history.v1', 'attempts': attempts,
              'projection': 'Selected public scalar fields from private pipeline receipts; argv, errors and pipeline logs excluded',
              'authority': 'Unsigned process evidence. Origin receipt digests identify private source records; those records are not distributed.'}
    verify_history(result, project, work)
    return result


def verify_history(history, project, work):
    require(history['schema'] == 'ruflo.vast-public-runtime-history.v1' and len(history['attempts']) == 2,
            'Incomplete runtime repair history')
    previous, current = history['attempts']
    protocol_sha = sha(project / RESEARCH / 'protocol.json')
    old_sha = sha(project / RESEARCH / 'archive/unused-runtime-attempt-01/protocol.json')
    require(previous['status'] == 'failed' and previous['protocol_sha256'] == old_sha
            and current['status'] == 'completed' and current['protocol_sha256'] == protocol_sha,
            'Runtime attempt status or protocol binding differs')
    require(all(isinstance(attempt['pipeline_sha256'], str) and re.fullmatch('[0-9a-f]{64}', attempt['pipeline_sha256'])
                and finite_number(attempt['deadline_epoch'], positive=True) for attempt in (previous, current)),
            'Invalid public history digest or deadline scalar')
    require(current['pipeline_sha256'] == sha(project / 'scripts/vast_worker_pipeline.py')
            and current['previous_receipt_sha256'] == previous['origin_receipt_sha256']
            and type(current['scored_generations_before_repair']) is int
            and current['scored_generations_before_repair'] == 0, 'Continuation provenance differs')
    require(current['deadline_epoch'] <= previous['deadline_epoch'], 'Continuation extended the experiment deadline')
    base_fields = {'origin_receipt', 'origin_receipt_sha256', 'protocol_sha256', 'pipeline_sha256', 'status',
                   'deadline_epoch', 'completed_at', 'generated_code_execution', 'cloud_api_calls', 'stages'}
    for index, attempt in enumerate((previous, current)):
        expected = base_fields | ({'previous_receipt_sha256', 'scored_generations_before_repair'} if index else set())
        require(set(attempt) == expected and attempt['generated_code_execution'] is False and attempt['cloud_api_calls'] is False,
                'Unexpected runtime history field or authority')
        require(attempt['origin_receipt'] == ('pipeline-continuation' if index else 'pipeline') + '/receipt.json',
                'Unexpected private history origin')
        require(re.fullmatch('[0-9a-f]{64}', attempt['origin_receipt_sha256']), 'Invalid history source digest')
        datetime.fromisoformat(attempt['completed_at'])
        for stage in attempt['stages']:
            require(set(stage) == {'stage', 'status', 'started_at', 'return_code', 'seconds', 'log_sha256'},
                    'Nonpublic runtime stage fields')
            require(re.fullmatch('[a-z0-9-]{1,80}', stage['stage']) and stage['status'] in ('failed', 'completed')
                    and type(stage['return_code']) is int and re.fullmatch('[0-9a-f]{64}', stage['log_sha256'])
                    and finite_number(stage['seconds']),
                    'Invalid stage history scalar')
            datetime.fromisoformat(stage['started_at'])
    require([stage['stage'] for stage in current['stages']] == ['preflight-' + arm for arm in PREFLIGHT] + ['benchmark']
            and all(stage['status'] == 'completed' and stage['return_code'] == 0 for stage in current['stages']),
            'Continuation did not complete both preflights and benchmark')
    abandoned = work / 'preflight-abandoned-01/qwen035-08b-base-direct'
    require(not any((abandoned / name).exists() or (abandoned / name).is_symlink()
                    for name in ('responses.jsonl', 'complete.json')), 'Abandoned preflight unexpectedly contains generations')
    binding, runtime, state = [read(abandoned / name) for name in ('binding.json', 'runtime.json', 'state.json')]
    require(binding['protocol_sha256'] == old_sha and binding['arm'] == 'qwen035-08b-base-direct'
            and binding['runtime_sha256'] == object_sha(runtime) and state['status'] == 'loading',
            'Abandoned preflight evidence differs')
    diagnosis = project / RESEARCH / 'archive/unused-runtime-attempt-01/eos-runtime-evidence.json'
    require(read(diagnosis)['failure_log_sha256'] == sha(diagnosis.with_name('preflight-failure.log')),
            'Public runtime failure log differs')


def native_license_files(root):
    index, provenance = read(root / 'index.json'), read(root / 'provenance.json')
    require(provenance['target'] == 'x86_64-unknown-linux-gnu' and provenance['native_source_commit'] == NATIVE_COMMIT
            and provenance['package_count'] == len(index) == 167, 'Incorrect native notice closure')
    require(provenance['dependency_tree_sha256'] == sha(root / 'dependency-licenses.txt')
            and provenance['index_sha256'] == sha(root / 'index.json'), 'Native notice provenance differs')
    names = ['index.json', 'provenance.json', 'dependency-licenses.txt']
    identities = set()
    for item in index:
        identity = item['package'] + '-' + item['version']
        require(identity not in identities and re.fullmatch(r'[A-Za-z0-9_.+-]+', identity), 'Duplicate or invalid dependency identity')
        identities.add(identity)
        require(set(item['notice_files']) == set(item['notice_sha256']), 'Incomplete notice file hashes')
        for name in item['notice_files']:
            require(relative(name).parts[0] == identity and len(relative(name).parts) == 2, 'Notice escapes its dependency directory')
            require(sha(safe_path(root, name)) == item['notice_sha256'][name], 'Dependency notice differs')
            names.append(name)
    require(len(names) == len(set(names)), 'Duplicate notice file')
    return names


def model_license_files(root, project):
    provenance = read(root / 'provenance.json')
    pins = {item['key']: item for item in read(project / RESEARCH / 'models.json')}
    records = provenance['models']
    require(len(records) == 3 and {record['key'] for record in records} == set(pins), 'Missing pinned model notices')
    names = ['provenance.json']
    for record in records:
        key = record['key']
        require(record['model'] == pins[key]['model'] and record['revision'] == pins[key]['revision']
                and record['license'] == 'Apache-2.0' and record['path'] == key + '.LICENSE', 'Model license provenance differs')
        require(record['sha256'] == sha(safe_path(root, record['path'])), 'Model notice digest differs')
        names.append(record['path'])
    return names


def copy_plan(project, work, results, inputs, archive, native_licenses, model_licenses):
    plan = [(safe_path(project, name), 'source/project/' + name) for name in SOURCE_FILES]
    plan += [(safe_path(work, name), 'work/' + name) for name in selected_work_files()]
    plan += [(safe_path(results, name), 'results/' + name) for name in RESULT_FILES]
    plan += [(safe_path(inputs, name), 'inputs/' + name) for name in
             ('patch.json', 'policy.json', 'adapter_config.json', 'adapter_model.safetensors')]
    plan += [(archive, 'source/native-source.tar.gz'), (project / 'LICENSE', 'licenses/project-LICENSE')]
    plan += [(safe_path(native_licenses, name), 'licenses/native-dependencies/' + name) for name in native_license_files(native_licenses)]
    plan += [(safe_path(model_licenses, name), 'licenses/models/' + name) for name in model_license_files(model_licenses, project)]
    destinations = [name for _, name in plan]
    require(len(destinations) == len(set(destinations)) and len(plan) < MAX_FILES, 'Duplicate or excessive copy plan')
    return plan


def copy_public(source, destination, budget):
    stream, info = open_regular(source)
    budget[0] += info.st_size
    if budget[0] > MAX_TOTAL:
        stream.close()
        raise ValueError('Public bundle exceeds total byte bound')
    destination.parent.mkdir(parents=True, exist_ok=True)
    fingerprint = hashlib.sha256()
    tail = b''
    text_file = source.suffix in ('.json', '.jsonl', '.py', '.sh', '.md', '.toml', '.txt', '.log')
    with stream, destination.open('xb') as target:
        total = 0
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            total += len(block)
            require(total <= info.st_size, 'Selected source changed while being copied')
            if text_file:
                require(not SECRET_MARKERS.search(tail + block), 'Private key or credential URL in public evidence')
                tail = block[-4096:]
            fingerprint.update(block)
            target.write(block)
        require(total == info.st_size, 'Selected source truncated during copy')
    mode = 0o555 if destination.name == 'weight_evolution' else 0o444
    destination.chmod(mode)
    return {'sha256': fingerprint.hexdigest(), 'bytes': info.st_size, 'mode': mode}


def file_inventory(root):
    records, total = {}, 0
    for path in sorted(root.rglob('*')):
        require(not path.is_symlink(), 'Symlink in evidence bundle')
        info = path.lstat()
        if stat.S_ISDIR(info.st_mode):
            continue
        require(stat.S_ISREG(info.st_mode), 'Nonregular evidence entry')
        name = path.relative_to(root).as_posix()
        relative(name)
        if name == 'manifest.json':
            continue
        total += info.st_size
        require(info.st_size <= MAX_FILE and total <= MAX_TOTAL and len(records) < MAX_FILES,
                'Evidence bundle exceeds resource bound')
        records[name] = {'sha256': sha(path), 'bytes': info.st_size, 'mode': info.st_mode & 0o7777}
    return records


def verify_inventory(root):
    require((root / 'manifest.json').stat().st_size <= 4 * 1024 * 1024, 'Manifest exceeds resource bound')
    require((root / 'manifest.json').stat().st_mode & 0o7777 & ~0o200 == 0o444, 'Manifest permissions differ')
    manifest = read(root / 'manifest.json')
    require(manifest['schema'] == 'ruflo.vast-public-evidence.v1', 'Unknown bundle schema')
    records = manifest['files']
    require(isinstance(records, dict) and 0 < len(records) <= MAX_FILES, 'Invalid file manifest')
    expected_dirs, total = set(), 0
    for name, record in records.items():
        path = relative(name)
        require(type(record['bytes']) is int and 0 <= record['bytes'] <= MAX_FILE, 'Invalid manifest byte count')
        total += record['bytes']
        require(total <= MAX_TOTAL, 'Manifest total exceeds byte bound')
        require(record['mode'] in (0o444, 0o555), 'Invalid public file permissions')
        for parent in path.parents:
            if str(parent) != '.': expected_dirs.add(str(parent))
    actual = file_inventory(root)
    require(set(actual) == set(records), 'Missing or unmanifested bundle file')
    for name, record in actual.items():
        wanted = records[name]
        require(record['sha256'] == wanted['sha256'] and record['bytes'] == wanted['bytes'], 'Bundle file digest differs: ' + name)
        # Python safe tar extraction may add owner write. No other mode changes pass.
        require(record['mode'] & ~0o200 == wanted['mode'], 'Bundle file permissions differ: ' + name)
    actual_dirs = {path.relative_to(root).as_posix() for path in root.rglob('*') if path.is_dir()}
    require(actual_dirs == expected_dirs, 'Unlisted bundle directory')
    require(manifest['total_bytes'] == total, 'Manifest total bytes differ')
    return manifest


def expected_bundle_files(root):
    names = {'README.md', 'source-pins.json', 'verify.py', 'source/native-source.tar.gz', 'licenses/project-LICENSE',
             'history/runtime-attempts.json'}
    names.update('source/project/' + name for name in SOURCE_FILES)
    names.update('work/' + name for name in selected_work_files())
    names.update('results/' + name for name in RESULT_FILES)
    names.update('inputs/' + name for name in ('patch.json', 'policy.json', 'adapter_config.json', 'adapter_model.safetensors'))
    names.update('licenses/native-dependencies/' + name for name in native_license_files(root / 'licenses/native-dependencies'))
    names.update('licenses/models/' + name for name in model_license_files(root / 'licenses/models', root / 'source/project'))
    return names


def verify_bundle(root):
    manifest = verify_inventory(root)
    require(set(manifest['files']) == expected_bundle_files(root), 'Bundle is outside the explicit public allowlist')
    require(sha(root / 'source/native-source.tar.gz') == NATIVE_ARCHIVE, 'Pinned native source archive differs')
    require(sha(root / 'verify.py') == sha(root / 'source/project/scripts/package_vast_evidence.py'), 'Verifier source differs')
    require(read(root / 'licenses/native-dependencies/provenance.json')['generator_sha256']
            == sha(root / 'source/project/scripts/native_dependency_notices.py'), 'Dependency notice generator differs')
    result = verify_evidence(root / 'source/project', root / 'work', root / 'results', root / 'inputs')
    verify_history(read(root / 'history/runtime-attempts.json'), root / 'source/project', root / 'work')
    require(manifest['evidence'] == result, 'Manifest scientific evidence differs')
    pins = read(root / 'source-pins.json')
    require(pins['protocol_sha256'] == result['protocol_sha256'] and pins['native_commit'] == NATIVE_COMMIT
            and pins['native_base'] == NATIVE_BASE and pins['native_tree'] == NATIVE_TREE
            and pins['linux_native_binary_sha256'] == LINUX_BINARY, 'Source pin binding differs')
    require(all(manifest[key] is False for key in ('production_qualified', 'sota_verified', 'selection_use')),
            'Manifest makes unsupported claims')
    return {**result, 'files_verified': len(manifest['files']), 'file_bytes_verified': manifest['total_bytes'],
            'manifest_sha256': sha(root / 'manifest.json'), 'read_only_verification': True}


def git(project, *args):
    return subprocess.check_output(['git', '-C', str(project), *args], text=True, timeout=20).strip()


def build(args):
    project, work, results = args.project_root.resolve(), args.work.resolve(), args.results.resolve()
    output = args.output.absolute()
    archive = output.with_name(output.name + '.tar.gz')
    checksum = archive.with_name(archive.name + '.sha256')
    require(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,100}', output.name), 'Use a simple artifact directory name')
    for path in (output, archive, checksum):
        require(not path.exists() and not path.is_symlink(), 'Immutable output already exists')
    inputs = (args.inputs or project / 'runtime/vast-cross-model/inputs').resolve()
    native_archive = (args.native_archive or project / 'runtime/vast-cross-model/native-source.tar.gz').absolute()
    native_licenses = (args.native_licenses or project / RESEARCH / 'licenses/native-dependencies').resolve()
    model_licenses = (args.model_licenses or project / RESEARCH / 'licenses/models').resolve()
    require(not native_archive.is_symlink() and sha(native_archive) == NATIVE_ARCHIVE, 'Wrong native source archive')
    proof = verify_evidence(project, work, results, inputs)
    history = history_projection(work, project)
    plan = copy_plan(project, work, results, inputs, native_archive, native_licenses, model_licenses)
    require(not git(project, 'status', '--porcelain', '--', *SOURCE_FILES), 'Commit selected source files before packaging')
    source_pins = {'research_commit': git(project, 'rev-parse', 'HEAD'),
                   'protocol_commit': git(project, 'log', '-1', '--format=%H', '--', RESEARCH + 'protocol.json'),
                   'pipeline_commit': git(project, 'log', '-1', '--format=%H', '--', 'scripts/vast_worker_pipeline.py'),
                   'protocol_sha256': proof['protocol_sha256'], 'native_commit': NATIVE_COMMIT, 'native_base': NATIVE_BASE,
                   'native_tree': NATIVE_TREE, 'native_source_archive_sha256': NATIVE_ARCHIVE,
                   'linux_native_binary_sha256': LINUX_BINARY,
                   'models': read(project / RESEARCH / 'models.json')}
    output.mkdir(parents=True, exist_ok=False)
    budget = [0]
    for source, name in plan:
        copy_public(source, safe_path(output, name), budget)
    copy_public(project / 'scripts/package_vast_evidence.py', output / 'verify.py', budget)
    write(output / 'source-pins.json', source_pins)
    write(output / 'history/runtime-attempts.json', history)
    (output / 'history/runtime-attempts.json').chmod(0o444)
    (output / 'README.md').write_text('''# Frozen cloud research evidence

This bundle contains 1,851 generations from seven fixed arms, complete public prompts and test fixtures, model revision and file digests, native patch and byte exact rollback receipts, Linux native source and executable, and the scored results. Model checkpoints, provider account records, SSH credentials, pipeline commands and logs, toolchain archives, compiler caches and cloud control code are excluded.

The prior runtime compatibility attempt was abandoned before scored inference. Its public source, old freeze, unscored failure log and explanatory notes are retained. The history file projects only public status, time and digest fields from the original and continuation pipeline receipts; it includes the origin receipt digests but excludes their argv and logs. The old preflight binding, runtime and loading state are included without generated responses. Origin digests identify undistributed private records and do not grant authority. No benchmark result selects a candidate or authorizes deployment. Production qualification and state of the art remain false. Read results/REPORT.md and the frozen protocol limitations before interpreting scores.

Verify on Python 3.12 or newer, from this directory:

```sh
python3 verify.py verify --bundle .
```

This command reads files, validates the exact inventory and hashes, checks all raw response bindings and 1,851 row counts, recomputes scalar score aggregates and paired comparisons, and verifies native build, patch and rollback evidence. It runs no bundled executable or generated code. It does not rerun HumanEval correctness, download model weights, prove remote hardware attestation, or authenticate a fully rewritten bundle. Compare the archive SHA256 against an independent trusted publication before extraction. An attacker replacing the complete archive and its unsigned manifest can replace their claims together.

Independent scoring reproduction requires macOS with /usr/bin/sandbox-exec, Python 3.12 and jsonschema 4.26.0. From this directory, install jsonschema in a separate environment, then run:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 source/project/research/vast_cross_model/score.py --work work --output /absolute/new/scored-results
```

The frozen scorer runs generated HumanEval code only in its macOS Seatbelt sandbox. Its unsupported-platform result is not valid HumanEval reproduction. Compare the new scored.json and summary.json with results/. Keep the output outside this immutable bundle.

Reproduce descriptive timing directly from the bundled raw batches:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 source/project/research/vast_cross_model/analysis/timing.py --work work --output /absolute/new/timing.json
```

The timing helper counts each repeated batch duration once. Aggregate batch throughput is not individual request latency; completion phase timing also includes model hashing and protocol verification. This optional analysis does not change scores or qualify a model. See source/project/research/vast_cross_model/analysis/README.md for the timing definitions and limitations.

To reproduce native editing, first obtain the exact original checkpoint from the pinned model revision. The public native source archive, verified Rust bootstrap helper and four intervention inputs are included; native/prepare.py documents its CLI. The Linux executable is provided as evidence and is never executed by this verifier. Full GPU reproduction requires the pinned container, downloaded original model weights, and a separately budgeted worker. No provider credential is included.

Files are read only. Safe tar extraction may add owner write permission; the verifier permits only that normalization. This is an evidence bundle, not a filesystem security boundary. Dependency notices cover the distributed native executable's 167 package normal/build closure; they do not claim to license an entire Linux container or Python environment, which are not distributed here.
''')
    for path in (output / 'README.md', output / 'source-pins.json'):
        path.chmod(0o444)
    copied_proof = verify_evidence(output / 'source/project', output / 'work', output / 'results', output / 'inputs')
    require(copied_proof == proof, 'Source evidence changed while copying')
    inventory = file_inventory(output)
    manifest = {'schema': 'ruflo.vast-public-evidence.v1', 'created_at': datetime.now(timezone.utc).isoformat(),
                'files': inventory, 'total_bytes': sum(item['bytes'] for item in inventory.values()),
                'evidence': proof, 'production_qualified': False, 'sota_verified': False, 'selection_use': False,
                'trust': 'Externally verified archive digest required; unsigned manifest alone is not an independent trust anchor'}
    write(output / 'manifest.json', manifest)
    (output / 'manifest.json').chmod(0o444)
    verified = verify_bundle(output)
    with archive.open('xb') as raw, gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0) as compressed:
        with tarfile.open(fileobj=compressed, mode='w', format=tarfile.PAX_FORMAT) as tar:
            for path in sorted(output.rglob('*')):
                if path.is_file():
                    info = tar.gettarinfo(str(path), arcname=output.name + '/' + path.relative_to(output).as_posix())
                    info.uid = info.gid = 0
                    info.uname = info.gname = ''
                    info.mtime = 0
                    with path.open('rb') as stream: tar.addfile(info, stream)
    archive_sha = sha(archive)
    with checksum.open('x') as stream: stream.write(archive_sha + '  ' + archive.name + '\n')
    archive.chmod(0o444)
    checksum.chmod(0o444)
    for path in sorted([p for p in output.rglob('*') if p.is_dir()], key=lambda p: len(p.parts), reverse=True): path.chmod(0o555)
    output.chmod(0o555)
    print(json.dumps({'bundle': str(output), 'archive': str(archive), 'archive_sha256': archive_sha,
                      'archive_bytes': archive.stat().st_size, **verified}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='operation', required=True)
    command = commands.add_parser('build')
    command.add_argument('--work', type=Path, required=True)
    command.add_argument('--results', type=Path, required=True)
    command.add_argument('--output', type=Path, required=True)
    command.add_argument('--project-root', type=Path, default=Path(__file__).resolve().parents[1])
    command.add_argument('--inputs', type=Path)
    command.add_argument('--native-archive', type=Path)
    command.add_argument('--native-licenses', type=Path)
    command.add_argument('--model-licenses', type=Path)
    command = commands.add_parser('verify')
    command.add_argument('--bundle', type=Path, required=True)
    args = parser.parse_args()
    if args.operation == 'build': build(args)
    else: print(json.dumps(verify_bundle(args.bundle.resolve()), indent=2))


if __name__ == '__main__':
    main()
