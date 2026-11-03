#!/usr/bin/env python3
"""Read-only local evidence verification. Never executes saved model output."""
import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat

FOUR_ARMS = ('baseline-release', 'selected-release', 'baseline-humaneval', 'selected-humaneval')
MATH_ARMS = ('base_direct', 'base_thinking', 'selected_direct', 'selected_thinking')
BASE = '04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696'
EDITED = 'bd5c223dc7d0ed48a6857f798a94da6ec46c40b9ff4e7bc31872dba10f943ae5'
PATCH = 'c77336ccbabbbde998bf5341c61afef922cd78f08b1bb2364b177e3bb52ef786'
ADAPTER = '52e5adefc83cc77357158ab4d4ea667287b3eaef5f87611b40e7a3397fabb0a4'
MATH_FILES = ('cases.json', 'plan.json', 'plan.sha256.json', 'protocol.json', 'runtime.json', 'toolchain.json',
              'topology.json', 'selection.json', 'release-binding.json', 'release-generations.jsonl',
              'release-summary.json', 'verification.json', 'rollback-verification.json',
              'native-release-alpha-100.json', 'LICENSE-research.txt')
FOUR_FILES = ('release-cases.jsonl', 'humaneval-cases.jsonl', 'humaneval-provenance.json', 'HumanEval.LICENSE',
              'selection.json', 'paired-verdicts.json', 'coding-integration.json')
FILES = frozenset(['README.md', 'tests/test_verify_benchmarks.py'] + ['local-08b/' + f for f in MATH_FILES]
                 + ['local-4b/' + f for f in FOUR_FILES]
                 + ['local-4b/' + arm + '/' + f for arm in FOUR_ARMS
                    for f in ('protocol.json', 'summary.json', 'responses.jsonl')])
MAX_FILE = 16 * 1024 * 1024
MAX_TOTAL = 64 * 1024 * 1024


def require(condition, message):
    if not condition:
        raise ValueError(message)


def relative(name):
    require(isinstance(name, str) and '\\' not in name and '\x00' not in name, 'Invalid relative path')
    p = PurePosixPath(name)
    require(name and not p.is_absolute() and p.as_posix() == name and all(s not in ('.', '..') for s in p.parts),
            'Unsafe or noncanonical relative path')
    return p


def bounded_bytes(path):
    fd = os.open(path, os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_NONBLOCK', 0))
    try:
        info = os.fstat(fd)
        require(stat.S_ISREG(info.st_mode) and 0 <= info.st_size <= MAX_FILE, 'Not a bounded regular file: ' + str(path))
        os.set_blocking(fd, True)
        with os.fdopen(fd, 'rb') as stream:
            fd = None
            data = stream.read(MAX_FILE + 1)
            require(len(data) <= MAX_FILE, 'File grew beyond size limit')
            return data
    finally:
        if fd is not None:
            os.close(fd)


def sha(path):
    return hashlib.sha256(bounded_bytes(path)).hexdigest()


def unique(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, 'Duplicate JSON key')
        result[key] = value
    return result


def nonfinite(value):
    raise ValueError('Nonfinite JSON: ' + value)


def decode(data):
    return json.loads(data, object_pairs_hook=unique, parse_constant=nonfinite)


def read(path):
    return decode(bounded_bytes(path))


def rows(path):
    lines = bounded_bytes(path).splitlines()
    require(0 < len(lines) <= 1000 and all(0 < len(line) <= 1024 * 1024 for line in lines), 'Invalid row inventory')
    return [decode(line) for line in lines]


def finite(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def paired(left, right, flag):
    require([r['id'] for r in left] == [r['id'] for r in right], 'Paired IDs differ')
    gains = [a['id'] for a, b in zip(left, right) if not a[flag] and b[flag]]
    losses = [a['id'] for a, b in zip(left, right) if a[flag] and not b[flag]]
    n = len(gains) + len(losses)
    p = min(1.0, 2 * sum(math.comb(n, k) for k in range(min(len(gains), len(losses)) + 1)) / 2**n) if n else 1.0
    return {'gains': gains, 'regressions': losses, 'two_sided_exact_mcnemar_p': p}


def verify_inventory(root):
    require(not root.is_symlink() and root.is_dir(), 'Evidence root must be a real directory')
    manifest = read(root / 'manifest.json')
    require(manifest['schema'] == 'ruvllm.local-benchmarks.v1', 'Wrong manifest schema')
    require(set(manifest['files']) == FILES, 'Manifest does not match exact public allowlist')
    allowed_dirs = {str(parent) for name in FILES for parent in PurePosixPath(name).parents if str(parent) != '.'}
    found = set()
    total = 0
    for path in root.rglob('*'):
        name = path.relative_to(root).as_posix()
        require(not path.is_symlink(), 'Symlink in evidence inventory')
        info = path.lstat()
        if stat.S_ISDIR(info.st_mode):
            require(name in allowed_dirs, 'Unexpected evidence directory: ' + name)
            continue
        require(stat.S_ISREG(info.st_mode), 'Nonregular evidence entry')
        require(name in FILES or name == 'manifest.json', 'Unexpected evidence file: ' + name)
        total += info.st_size
        require(total <= MAX_TOTAL, 'Evidence exceeds total byte limit')
        found.add(name)
    require(found == FILES | {'manifest.json'}, 'Missing evidence files')
    for name, record in manifest['files'].items():
        relative(name)
        require(set(record) in ({'bytes', 'sha256', 'origin'}, {'bytes', 'sha256'}), 'Unexpected manifest record')
        data = bounded_bytes(root / name)
        require(type(record['bytes']) is int and len(data) == record['bytes'], 'Byte count differs: ' + name)
        require(hashlib.sha256(data).hexdigest() == record['sha256'], 'Digest differs: ' + name)
        if 'origin' in record:
            origin = record['origin']
            relative(origin['path'])
            require(origin['project'] == 'ruflo-code-model' and origin['sha256'] == record['sha256'], 'Source bytes differ')
    require(sha(root.parent / 'scripts/verify_benchmarks.py') == manifest['verifier_sha256'], 'Verifier digest differs')
    require(re.fullmatch('[0-9a-f]{40}', manifest['source_commit']), 'Invalid source commit')
    require(all(manifest[key] is False for key in ('sota_verified', 'production_authorized', 'recursive_self_improvement_verified')),
            'Unsupported qualification claim')
    projection = manifest['derived_report']
    require(set(projection) == {'source_path', 'source_sha256', 'copied_fields', 'output_path'}, 'Invalid report projection')
    require(projection['source_path'] == 'results/report.json' and re.fullmatch('[0-9a-f]{64}', projection['source_sha256'])
            and projection['copied_fields'] == ['release_paired', 'external_paired']
            and projection['output_path'] == 'local-4b/paired-verdicts.json', 'Report projection scope differs')
    return manifest


def verify_4b(root):
    folder = root / 'local-4b'
    suites = {'release': rows(folder / 'release-cases.jsonl'), 'humaneval': rows(folder / 'humaneval-cases.jsonl')}
    require(len(suites['release']) == 35 and len(suites['humaneval']) == 164, 'Wrong case counts')
    for cases in suites.values():
        require(len({c['id'] for c in cases}) == len(cases), 'Duplicate case ID')
    require([c['id'] for c in suites['humaneval']] == ['HumanEval/' + str(i) for i in range(164)], 'HumanEval inventory differs')
    provenance = read(folder / 'humaneval-provenance.json')
    require(provenance['sha256'] == sha(folder / 'humaneval-cases.jsonl') and provenance['cases'] == 164
            and provenance['selection_use'] is False and provenance['revision'] == 'b9453e8f1865df0a0057a1383767caa400acfbcf',
            'HumanEval provenance differs')
    selection = read(folder / 'selection.json')
    require(selection['release_used_for_selection'] is False
            and selection['selected_adapter'] == './runtime/pilot-v2/step-064', 'Selection scope differs')
    require([t['adapter_sha256'] for t in selection['trials'] if t['accepted']] == [ADAPTER], 'Selected adapter differs')
    output, saved, protocols = {}, {}, {}
    for arm in FOUR_ARMS:
        suite = arm.split('-', 1)[1]
        cases, data = suites[suite], rows(folder / arm / 'responses.jsonl')
        protocol, summary = read(folder / arm / 'protocol.json'), read(folder / arm / 'summary.json')
        protocols[arm] = protocol
        require([r['id'] for r in data] == [c['id'] for c in cases], 'Wrong or reordered 4B row inventory')
        require(protocol['suite_sha256'] == sha(folder / (suite + '-cases.jsonl')) and protocol['cases'] == len(cases),
                'Suite binding differs')
        require(protocol['max_tokens'] == 768 and protocol['temperature'] == 0 and protocol['seed'] == 20261102
                and protocol['enable_thinking'] is False and protocol['mlx'] == '0.32.4', '4B inference protocol differs')
        require(protocol['model_pin']['repo_id'] == 'Qwen/Qwen3.5-4B'
                and protocol['model_pin']['revision'] == '851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a'
                and protocol['model_path'] == './models/base-4bit', '4B model revision or quantization path differs')
        selected = arm.startswith('selected')
        require(protocol['adapter_sha256'] == (ADAPTER if selected else None)
                and protocol['adapter'] == ('./runtime/pilot-v2/step-064' if selected else None), '4B adapter binding differs')
        if selected:
            require(selection['selected_at'] < protocol['started_at'], 'Selection was not fixed before selected evaluation')
        for row, case in zip(data, cases):
            require(row['kind'] == case['kind'] and row['family'] == case['family'], 'Case type differs')
            require(type(row['passed']) is bool and row['finish_reason'] in ('stop', 'length', 'error'), 'Invalid saved verdict')
            require(not row['passed'] or row['finish_reason'] == 'stop', 'Truncated or errored answer marked passed')
            require(type(row['output_tokens']) is int and 0 <= row['output_tokens'] <= 768, 'Invalid token count')
            require(row['finish_reason'] != 'length' or row['output_tokens'] == 768, 'Length termination differs')
            require(finite(row['seconds']) and finite(row['peak_memory_gb']), 'Invalid timing or memory scalar')
        result = {'total': len(data), 'passed': sum(r['passed'] for r in data),
                  'completed': sum(r['finish_reason'] == 'stop' for r in data),
                  'truncated': sum(r['finish_reason'] == 'length' for r in data),
                  'output_tokens': sum(r['output_tokens'] for r in data),
                  'elapsed_seconds': sum(r['seconds'] for r in data),
                  'groups': {kind: {'total': sum(r['kind'] == kind for r in data),
                                   'passed': sum(r['passed'] for r in data if r['kind'] == kind)}
                             for kind in sorted({r['kind'] for r in data})}}
        for key in ('total', 'passed', 'truncated', 'output_tokens', 'groups'):
            require(summary[key] == result[key], 'Saved summary differs: ' + arm + '/' + key)
        require(math.isclose(summary['elapsed_seconds'], result['elapsed_seconds'], abs_tol=1e-8)
                and summary['accuracy'] == result['passed'] / result['total'], 'Summary rate or timing differs')
        require(summary['response_sha256'] == sha(folder / arm / 'responses.jsonl'), 'Raw response digest differs')
        output[arm], saved[arm] = result, data
    paired_claims = read(folder / 'paired-verdicts.json')
    require(set(paired_claims) == {'release_paired', 'external_paired'}, 'Unexpected paired report scope')
    comparisons = {}
    for suite, report_key in (('release', 'release_paired'), ('humaneval', 'external_paired')):
        baseline, selected = protocols['baseline-' + suite], protocols['selected-' + suite]
        keys = set(baseline) - {'adapter', 'adapter_sha256', 'started_at'}
        require(all(baseline[k] == selected[k] for k in keys), 'Paired inference settings differ')
        result = paired(saved['baseline-' + suite], saved['selected-' + suite], 'passed')
        claim = paired_claims[report_key]
        require(result['gains'] == claim['improved_ids'] and result['regressions'] == claim['regressed_ids']
                and result['two_sided_exact_mcnemar_p'] == claim['exact_mcnemar_p'], 'Paired verdict report differs')
        comparisons[suite] = result
    return {'arms': output, 'paired': comparisons, 'saved_verdicts_verified': True, 'generated_code_reexecuted': False,
            'token_accounting': 'Saved output token counts; tokenizer replay is not performed'}


def verify_math(root):
    folder = root / 'local-08b'
    cases, plan = read(folder / 'cases.json'), read(folder / 'plan.json')
    require(sha(folder / 'plan.json') == read(folder / 'plan.sha256.json')['sha256'], 'Plan digest differs')
    require(sha(folder / 'cases.json') == plan['input_sha256']['release_cases'], 'Frozen cases differ')
    require([c['id'] for c in cases] == [f'we-release-{i:02}' for i in range(1, 17)], 'Math case inventory differs')
    answers = [23*47, 12*(62//2-12), 240*85*75//10000, 9*12*13//2, 100*21//35, 5+24*6,
               82-3*23, pow(3,40,17), math.gcd(252,198), 5*24-12-18-24-30, 1000*11*11//100,
               6*7*7, math.comb(9,2), math.lcm(18,24), 15-2*((45-41)//2), 3*168//4+2*90//3]
    require([c['answer'] for c in cases] == answers, 'Expected arithmetic answers differ')
    settings = plan['protocol']
    require(tuple(settings['release_arms']) == MATH_ARMS and settings['max_new_tokens'] == 2048
            and settings['release_seed_start'] == 2026110301 and settings['eos_token_ids'] == [248044,248046],
            'Math protocol differs')
    require(plan['model']['source_sha256'] == BASE
            and plan['model']['revision'] == '2fc06364715b967f1860aea9cf38778875588b17', 'Math source revision differs')
    for name, expected in plan['execution_sha256'].items():
        require(name in ('runtime','toolchain','protocol','topology') and sha(folder / (name+'.json')) == expected,
                'Execution artifact digest differs')
    protocol = read(folder / 'protocol.json')
    require(protocol['protocol'] == settings and protocol['release_cases_sha256'] == sha(folder / 'cases.json'),
            'Protocol artifact differs')
    binding, selection = read(folder / 'release-binding.json'), read(folder / 'selection.json')
    require(selection['release_seen'] is False and selection['selected'] == 'alpha-100'
            and binding['selection_sha256'] == sha(folder / 'selection.json')
            and binding['plan_sha256'] == sha(folder / 'plan.json')
            and binding['candidate_sha256'] == EDITED, 'Frozen release selection differs')
    data = rows(folder / 'release-generations.jsonl')
    require(Counter((r['id'],r['arm']) for r in data) == Counter((c['id'],a) for c in cases for a in MATH_ARMS),
            'Missing, duplicate or unexpected math generation')
    summaries, by_arm = {}, {}
    for arm in MATH_ARMS:
        arm_rows = [r for r in data if r['arm'] == arm]
        require([r['id'] for r in arm_rows] == [c['id'] for c in cases], 'Reordered math rows')
        result = {k:0 for k in ('n','answer_correct','completed','delivered_correct','truncated','output_tokens')}
        for index, (row, case) in enumerate(zip(arm_rows,cases)):
            selected, thinking = arm.startswith('selected'), arm.endswith('thinking')
            require(row['question'] == case['question'] and row['expected'] == case['answer']
                    and row['seed'] == settings['release_seed_start'] + index, 'Prompt or seed differs')
            model = EDITED if selected else BASE
            require(row['model_sha256'] == model, 'Row model differs')
            execution = row['execution']
            require(execution['modelSha256'] == model and execution['modelGeneration'] == int(selected)
                    and execution['patchSha256'] == (PATCH if selected else '0'*64), 'Generation identity differs')
            for key, name in (('runtimeSha256','runtime'),('toolchainSha256','toolchain'),
                              ('protocolSha256','protocol'),('topologySha256','topology')):
                require(execution[key] == plan['execution_sha256'][name], 'Row execution binding differs')
            tokens = row['generated_token_ids']
            require(isinstance(tokens,list) and all(type(t) is int and 0 <= t < 2**32 for t in tokens)
                    and len(tokens) == row['output_tokens'] and 0 < len(tokens) <= 2048, 'Generated token inventory differs')
            finished = tokens[-1] in settings['eos_token_ids']
            require(row['finish_reason'] == ('stop' if finished else 'length') and (finished or len(tokens) == 2048),
                    'Completion or cap differs')
            text = row['text']
            has_close = '</think>' in text
            closed = has_close and text.rfind('<think>') < text.rfind('</think>')
            final = text[text.rfind('</think>') + len('</think>'):] if has_close else text
            values = re.findall(r'\\boxed\{\s*([+-]?\d[\d,]*)\s*\}', final)
            answer = int(values[-1].replace(',','')) if values else None
            correct = answer == case['answer']; delivered = correct and finished and (not thinking or closed)
            expected = {'answer_correct':correct,'completed':finished,'delivered_correct':delivered,'truncated':not finished}
            require(row['answer'] == answer and all(type(row[k]) is bool and row[k] == v for k,v in expected.items()),
                    'Independent math rescore differs')
            require(row['thinking_closed'] is (closed if thinking else None), 'Thinking completion differs')
            for key,value in {**expected,'n':1,'output_tokens':len(tokens)}.items():result[key] += value
        for peer in (read(folder/'release-summary.json')[arm], read(folder/'verification.json')['summary'][arm]):
            require(all(peer[k] == v for k,v in result.items()), 'Math summary differs')
        summaries[arm], by_arm[arm] = result, arm_rows
    for mode in ('direct','thinking'):
        for a,b in zip(by_arm['base_'+mode],by_arm['selected_'+mode]):
            require(a['prompt_token_ids'] == b['prompt_token_ids'], 'Paired tokenized prompts differ')
    comparisons = {mode:paired(by_arm['base_'+mode],by_arm['selected_'+mode],'delivered_correct') for mode in ('direct','thinking')}
    verification = read(folder/'verification.json')
    require(comparisons == verification['paired'] and verification['plan_sha256'] == sha(folder/'plan.json'), 'Math pairing differs')
    gate = summaries['selected_direct']['delivered_correct'] > summaries['base_direct']['delivered_correct'] and \
           summaries['selected_direct']['completed'] >= summaries['base_direct']['completed']
    require(verification['behavioral_gate_passed'] is gate and not gate
            and all(verification[k] is False for k in ('production_authorized','sota_verified','recursive_self_improvement_verified')),
            'Unsupported math qualification claim')
    rollback = read(folder/'rollback-verification.json')
    require(rollback['original_sha256'] == rollback['restored_sha256'] == BASE and rollback['byte_exact'] is True
            and rollback['source_unchanged'] is True and rollback['production_activation'] is False
            and rollback['bytes'] == 1746942600, 'Rollback receipt differs')
    return {'arms':summaries,'paired':comparisons,'generations':len(data),'math_text_independently_rescored':True,
            'behavioral_gate_passed':gate,'rollback_receipt_consistent':True,'checkpoint_bytes_rehashed':False}


def verify_historical_receipts(root):
    native = read(root/'local-08b/native-release-alpha-100.json')
    receipt = native['receipt']
    plan = read(root/'local-08b/plan.json')
    selected = next(c for c in plan['candidates'] if c['id'] == 'alpha-100')
    require(native['operation'] == 'registered_final_patch'
            and native['native_source_commit'] == 'f5c3444968ede44e43744460192198962ae94da2'
            and native['binary_sha256'] == '4a799f3eb18f6f169ec947438a7f4cf244421a483ef637095f5ac64cfaec78c0'
            and native['binary_unchanged_after_application'] is True, 'Historical native execution binding differs')
    require(finite(native['wall_seconds']) and round(native['wall_seconds'],3) == 6.692,
            'Historical native timing differs')
    require(receipt['schema'] == 'ruvllm.weight_receipt.v1' and receipt['activation_authorized'] is False
            and receipt['source_sha256'] == BASE and receipt['candidate_sha256'] == EDITED
            and receipt['patch_sha256'] == PATCH and receipt['policy_sha256'] == selected['policy_sha256']
            and receipt['changed_values'] == selected['changed_values'] == 130188,
            'Historical native receipt identity differs')
    for key in ('l2','source_l2','relative_l2','max_abs_delta'):
        require(finite(receipt[key]) and math.isclose(receipt[key], selected[key], rel_tol=1e-10),
                'Historical native delta differs')
    require(receipt['source_l2'] > 0 and math.isclose(receipt['relative_l2'],receipt['l2']/receipt['source_l2'],rel_tol=1e-12),
            'Historical relative norm differs')
    require(re.fullmatch('[0-9a-f]{64}',native['preexisting_signed_plan_sha256']), 'Invalid historical plan reference')
    coding = read(root/'local-4b/coding-integration.json')
    require(coding['adapter_sha256'] == ADAPTER and coding['matched_adapter_tensors'] == 32
            and all(coding[k] is True for k in ('backend_binding_verified','bash_tests_completed',
                                               'fixture_only','independent_tests_passed','passed'))
            and coding['permission_bypass'] is False and type(coding['exit_code']) is int and coding['exit_code'] == 0,
            'Historical coding fixture binding or outcome differs')
    require(coding['claude_version'] == '2.1.316 (Claude Code)' and coding['router_version'] == '3.1.3'
            and coding['completed_tool_names'] == ['Bash','Edit','Read']
            and coding['tool_names'] == ['Read','Read','Edit','Bash'] and coding['test_reason'] == 'passed',
            'Historical coding fixture tool path differs')
    require(finite(coding['seconds']) and coding['seconds'] == 20.826, 'Historical coding fixture timing differs')
    require(all(re.fullmatch('[0-9a-f]{64}',coding[k]) for k in ('before_sha256','after_sha256'))
            and coding['before_sha256'] != coding['after_sha256'], 'Historical fixture file identity differs')
    return {'native_patch_seconds':native['wall_seconds'],'native_changed_values':receipt['changed_values'],
            'claude_code_fixture_seconds':coding['seconds'],'claude_code_fixture_passed':True,
            'receipt_fields_verified':True,'historical_processes_reexecuted':False,
            'referenced_governance_signature_authenticated':False}


def verify(root):
    root = Path(root).absolute()
    manifest = verify_inventory(root)
    four, small = verify_4b(root), verify_math(root)
    historical = verify_historical_receipts(root)
    return {'verified':True,'files_verified':len(FILES),'generations_verified':462,
            'local_4b':four,'local_08b':small,'historical_receipts':historical,
            'generated_code_reexecuted':False,'gpu_inference_reexecuted':False,
            'production_authorized':False,'sota_verified':False,'recursive_self_improvement_verified':False,
            'manifest_sha256':sha(root/'manifest.json'),
            'trust':'Unsigned evidence requires an independently trusted repository commit or manifest digest; consistency is not hardware attestation'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--benchmarks',type=Path,default=Path(__file__).resolve().parents[1]/'benchmarks')
    parser.add_argument('--manifest-sha256',help='Expected manifest digest from an independent trusted record')
    args = parser.parse_args()
    if args.manifest_sha256:
        require(re.fullmatch('[0-9a-f]{64}',args.manifest_sha256) and sha(args.benchmarks/'manifest.json') == args.manifest_sha256,
                'External manifest trust anchor differs')
    print(json.dumps(verify(args.benchmarks),indent=2,sort_keys=True,allow_nan=False))


if __name__ == '__main__':
    main()
