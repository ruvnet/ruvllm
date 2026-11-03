#!/usr/bin/env python3
"""Post hoc grader stability audit. Run only after final scoring and GPU teardown.

This repeats saved code grading, never model inference, model selection or primary
score replacement. Exactly 30 additional calls cover four fixed cases in five
direct arms. HumanEval/32 is a seeded control; /38, /50 and /53 are unseeded.
Use the original Python 3.12 macOS environment and unchanged frozen scorer.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import sys
import tempfile
import time


PROJECT = Path(__file__).resolve().parents[3]
SCIENCE = PROJECT / 'research/vast_cross_model'
PROTOCOL = 'cf83995f29b5c3f7c28871d87d1295839bd94452dec0b8c352b990ba52a26cb6'
VERIFIER_SHA = '9383d713cfdc3c0bd2f3d34da599b6d20949df7f65fb897fbca3dccfe3658660'
SCORER_SHA = '4bfd12082fb8fc9cf65c069d9ed28be9979553c834d2afc4664ec352c279da6a'
SANDBOX_SHA = 'e462267be43484accd153f2d1f126977b2394a26505a4cd2153489ff46ec3055'
CASE_IDS = ('HumanEval/32', 'HumanEval/38', 'HumanEval/50', 'HumanEval/53')
ARM_IDS = ('qwen035-08b-base-direct', 'qwen035-08b-edited-direct', 'qwen035-4b-base-direct',
           'qwen035-4b-lora-direct', 'qwen3-coder-30b-direct')
REPEATS = 30
REASONS = {'passed', 'syntax error', 'unsupported import', 'unsupported dynamic operation',
           'unsupported introspection', 'sandbox unavailable', 'execution or assertion failure',
           'execution timeout', 'strict parse: ValueError', 'strict parse: TypeError',
           'strict parse: KeyError', 'strict parse: StopIteration'}
FLAGS = ('answer_correct', 'completed', 'delivered_correct', 'truncated')


def require(value, message):
    if not value:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def object_sha(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def file_sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def trusted_verifier():
    path = PROJECT / 'scripts/package_vast_evidence.py'
    require(file_sha(path) == VERIFIER_SHA, 'Reviewed evidence verifier changed')
    return load_module('_grading_stability_evidence', path)


def source_gate(evidence):
    protocol, digest = evidence.protocol_inputs(PROJECT)
    require(digest == PROTOCOL and file_sha(SCIENCE / 'score.py') == SCORER_SHA
            and file_sha(PROJECT / 'ruflo_model/scoring.py') == SANDBOX_SHA,
            'Audit requires the original frozen protocol and scorers')
    require(protocol['total_generations'] == 1851, 'Audit requires 1851 primary generations')
    return protocol


def snapshot(evidence, protocol, work, primary, inputs):
    """Public logical labels and hashes only; never serialize local absolute paths."""
    selected = {'research/vast_cross_model/protocol.json', 'research/vast_cross_model/freeze.json',
                'research/vast_cross_model/analysis/grading_stability.py',
                'scripts/package_vast_evidence.py', 'ruflo_model/__init__.py'}
    for group in ('input_sha256', 'implementation_sha256'):
        selected.update(str((SCIENCE / name).resolve().relative_to(PROJECT)) for name in protocol[group])
    paths = [('source/' + name, evidence.safe_path(PROJECT, name)) for name in sorted(selected)]
    paths += [('work/' + name, evidence.safe_path(work, name)) for name in evidence.selected_work_files()]
    paths += [('primary/' + name, evidence.safe_path(primary, name)) for name in evidence.RESULT_FILES]
    paths += [('inputs/' + name, evidence.safe_path(inputs, name)) for name in
              ('patch.json', 'policy.json', 'adapter_model.safetensors', 'adapter_config.json')]
    return {label: evidence.sha(path) for label, path in paths}


def validate_grade(grade, case, raw):
    require(set(grade) == {'id', 'panel', *FLAGS, 'reason', 'output_tokens'}
            and grade['id'] == case['id'] and grade['panel'] == case['panel']
            and type(grade['output_tokens']) is int and grade['output_tokens'] == raw['output_tokens']
            and all(type(grade[key]) is bool for key in FLAGS)
            and grade['reason'] in REASONS, 'Unexpected code grade or nonpublic reason')
    closed = (not raw['thinking'] or '</think>' in raw['text']) and raw['text'].rfind('<think>') <= raw['text'].rfind('</think>')
    require(grade['completed'] is (raw['finish_reason'] == 'stop' and closed)
            and grade['truncated'] is (raw['finish_reason'] == 'length')
            and grade['delivered_correct'] is (grade['completed'] and grade['answer_correct']),
            'Code grade completion semantics changed')


def prepare(work, primary, inputs):
    evidence = trusted_verifier()
    protocol = source_gate(evidence)
    before = snapshot(evidence, protocol, work, primary, inputs)
    proof = evidence.verify_evidence(PROJECT, work, primary, inputs)
    require(proof['protocol_sha256'] == PROTOCOL and proof['generations'] == 1851
            and proof['answer_correctness_reexecuted'] is False,
            'Incomplete primary evidence; no repeat grading is allowed')
    cases = {case['id']: case for case in evidence.read(SCIENCE / 'cases.json')}
    primary_scores = evidence.read(primary / 'scored.json')
    selected = []
    require(tuple(arm['id'] for arm in protocol['arms'] if 'humaneval164' in arm['panels']) == ARM_IDS,
            'The five fixed direct arms changed')
    for arm in ARM_IDS:
        raw = {row['id']: row for row in evidence.rows(work / 'outputs' / arm / 'responses.jsonl')}
        grades = {row['id']: row for row in primary_scores[arm]}
        for ident in CASE_IDS:
            case, row, grade = cases[ident], raw[ident], grades[ident]
            require(case['kind'] == row['kind'] == 'code' and not row['thinking'], 'Unexpected audit case')
            validate_grade(grade, case, row)
            selected.append({'arm': arm, 'case': case, 'raw': row, 'primary': grade})
    require(len(selected) == 20 and snapshot(evidence, protocol, work, primary, inputs) == before,
            'Selected evidence changed while preparing the audit')
    provenance = {'protocol_sha256': PROTOCOL, 'complete_primary_generations': 1851,
                  'source_and_evidence_sha256': before, 'inventory_sha256': object_sha(before)}

    def revalidate():
        source_gate(evidence)
        require(snapshot(evidence, protocol, work, primary, inputs) == before,
                'Source or primary evidence changed during the audit')

    return selected, provenance, revalidate


def frozen_grade():
    require(sys.platform == 'darwin' and Path('/usr/bin/sandbox-exec').is_file()
            and sys.version_info[:2] == (3, 12), 'Original macOS Python 3.12 sandbox is required')
    require(importlib.metadata.version('jsonschema') == '4.26.0', 'Original jsonschema version is required')
    require(file_sha(SCIENCE / 'score.py') == SCORER_SHA
            and file_sha(PROJECT / 'ruflo_model/scoring.py') == SANDBOX_SHA, 'Frozen scorer changed')
    # All imported code is from the trusted checkout; work contains data only.
    previous_path = list(sys.path)
    sys.path.insert(0, str(PROJECT))
    sys.path.insert(0, str(SCIENCE))
    common = load_module('_grading_stability_common', SCIENCE / 'common.py')
    previous = sys.modules.get('common')
    sys.modules['common'] = common
    try:
        module = load_module('_grading_stability_score', SCIENCE / 'score.py')
    finally:
        if previous is None:
            sys.modules.pop('common', None)
        else:
            sys.modules['common'] = previous
        sys.path[:] = previous_path
    sandbox = sys.modules['ruflo_model.scoring']
    require(Path(sandbox.__file__).resolve() == PROJECT / 'ruflo_model/scoring.py'
            and module.score_code is sandbox.score_code, 'Unexpected sandbox scorer import')
    return module.grade


def atomic_publish(value, output):
    descriptor, temporary = tempfile.mkstemp(prefix='.grading-stability-', dir=output.parent)
    try:
        with os.fdopen(descriptor, 'wb') as stream:
            os.fchmod(stream.fileno(), 0o444)
            stream.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode() + b'\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, output)
        descriptor = os.open(output.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        os.unlink(temporary)


def execute(selected, provenance, revalidate, grade, output):
    """Separate runner permits synthetic tests without executing any model output."""
    output = Path(output)
    journal = output.with_name(output.name + '.partial.jsonl')
    require(output.parent.is_dir() and not output.exists() and not output.is_symlink()
            and not journal.exists() and not journal.is_symlink(), 'Audit output and journal must both be new')
    require([(row['arm'], row['case']['id']) for row in selected]
            == [(arm, ident) for arm in ARM_IDS for ident in CASE_IDS], 'Audit target inventory differs')
    for row in selected:
        validate_grade(row['primary'], row['case'], row['raw'])
    revalidate()
    started = datetime.now(timezone.utc).isoformat()
    wall_start = time.monotonic()
    records = []
    # Durable partial records survive interruption without claiming a finished audit.
    with journal.open('x', encoding='utf-8') as stream:
        stream.write(canonical({'status': 'in_progress', 'purpose': 'post_hoc_grader_stability',
                                'started_at': started, 'provenance': provenance}).decode() + '\n')
        stream.flush()
        os.fsync(stream.fileno())
        for target in selected:
            case, raw, primary = target['case'], target['raw'], target['primary']
            for repetition in range(1, REPEATS + 1):
                begin = time.monotonic()
                event = {'arm': target['arm'], 'id': case['id'], 'repeat': repetition}
                try:
                    result = grade(case, raw)
                    validate_grade(result, case, raw)
                    event.update(status='completed', grade=result,
                                 answer_disagrees_with_primary=result['answer_correct'] != primary['answer_correct'],
                                 delivered_disagrees_with_primary=result['delivered_correct'] != primary['delivered_correct'],
                                 reason_differs_from_primary=result['reason'] != primary['reason'])
                except BaseException as error:
                    event.update(status='error' if isinstance(error, Exception) else 'interrupted',
                                 error_class=type(error).__name__)
                    if isinstance(error, Exception):
                        raise ValueError('Repeat grading failed; partial journal retained') from None
                    raise
                finally:
                    event['seconds'] = time.monotonic() - begin
                    require(math.isfinite(event['seconds']) and event['seconds'] >= 0, 'Invalid monotonic duration')
                    stream.write(canonical(event).decode() + '\n')
                    stream.flush()
                    os.fsync(stream.fileno())
                records.append(event)
        revalidate()
    summaries = []
    for target in selected:
        rows = [row for row in records if row['arm'] == target['arm'] and row['id'] == target['case']['id']]
        summaries.append({'arm': target['arm'], 'id': target['case']['id'],
            'fixture_randomness': 'local_random_seed_42_control' if target['case']['id'] == 'HumanEval/32' else 'unseeded_fixture_draws',
            'case_sha256': object_sha(target['case']),
            'fixture_sha256': hashlib.sha256(target['case']['tests'].encode()).hexdigest(),
            'raw_response_sha256': object_sha(target['raw']), 'primary_verdict': target['primary'],
            'repeat_count': len(rows), 'repeat_answer_passes': sum(row['grade']['answer_correct'] for row in rows),
            'repeat_delivered_passes': sum(row['grade']['delivered_correct'] for row in rows),
            'answer_disagreements_with_primary': sum(row['answer_disagrees_with_primary'] for row in rows),
            'delivered_disagreements_with_primary': sum(row['delivered_disagrees_with_primary'] for row in rows),
            'repeat_verdicts_vary': len({row['grade']['answer_correct'] for row in rows}) > 1,
            'reason_counts': dict(sorted(Counter(row['grade']['reason'] for row in rows).items()))})
    result = {'schema': 'ruvllm.post-hoc-grader-stability.v1', 'status': 'completed',
        'purpose': 'post_hoc_grader_stability', 'model_selection': False, 'primary_scores_replaced': False,
        'new_model_generations': 0, 'cloud_calls': False, 'started_at': started,
        'completed_at': datetime.now(timezone.utc).isoformat(), 'seconds': time.monotonic() - wall_start,
        'provenance': provenance, 'planned_repeats_per_output': REPEATS, 'audited_saved_outputs': 20,
        'completed_repeat_calls': len(records), 'cases': summaries, 'repeats': records,
        'limits': [
            'This post hoc audit does not alter primary comparisons, select models or repair frozen tests.',
            'HumanEval/32 has a local seed of 42; the other three fixtures have unseeded random draws.',
            'Thirty repeats detect a 10 percent disagreement rate with probability 1 - 0.9**30 under independent identical trials, per output.',
            'No observed disagreement does not prove deterministic grading or correctness on all possible inputs.',
            'Reasons are the unchanged scorer categories; private paths and subprocess stderr are not published.',
            'Run sequencing after GPU teardown is an operator requirement, not authenticated by this receipt.'
        ]}
    require(len(records) == 600, 'Missing planned repeat calls')
    atomic_publish(result, output)
    journal.unlink()
    return result


def audit(work, primary, inputs, output):
    work, primary, inputs = (Path(path).resolve(strict=True) for path in (work, primary, inputs))
    output = Path(output).absolute()
    require(not any(output.resolve().is_relative_to(path) for path in (work, primary, inputs)),
            'Audit output must be separate from immutable inputs and primary scores')
    require(output.parent.is_dir() and not output.exists() and not output.is_symlink()
            and not output.with_name(output.name + '.partial.jsonl').exists(), 'Audit output must be new')
    selected, provenance, revalidate = prepare(work, primary, inputs)
    grader = frozen_grade()
    provenance['local_grading_runtime'] = {'python': platform.python_version(), 'platform': sys.platform,
        'python_executable_sha256': file_sha(Path(sys.executable).resolve()),
        'jsonschema': importlib.metadata.version('jsonschema'), 'sandbox': 'macOS Seatbelt'}
    return execute(selected, provenance, revalidate, grader, output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--primary', type=Path, required=True)
    parser.add_argument('--inputs', type=Path, default=PROJECT / 'runtime/vast-cross-model/inputs')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        result = audit(args.work, args.primary, args.inputs, args.output)
    except (OSError, ValueError, KeyError, TypeError, IndexError, ImportError) as error:
        print(json.dumps({'audit_completed': False, 'error_class': type(error).__name__,
                          'reason': 'Audit refused or interrupted; preserve any partial journal'}), file=sys.stderr)
        return 1
    print(json.dumps({'audit_completed': True, 'repeat_calls': result['completed_repeat_calls'],
                     'outputs_with_answer_disagreements': sum(row['answer_disagreements_with_primary'] > 0 for row in result['cases'])}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
