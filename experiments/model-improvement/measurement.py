"""Fail-closed paired scoring. Reads model text; never executes tools or generated code."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
HASH = re.compile(r'[a-f0-9]{64}')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def file_sha(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), 'Expected regular nonsymlink file')
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def object_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def unique(pairs):
    value = {}
    for key, item in pairs:
        require(key not in value, 'Duplicate JSON key')
        value[key] = item
    return value


def read_json(path, limit=16 * 1024 * 1024):
    path = Path(path)
    require(path.is_file() and not path.is_symlink() and path.stat().st_size <= limit,
            'Expected bounded regular JSON file')
    return json.loads(path.read_text(), object_pairs_hook=unique,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError('Nonfinite JSON')))


def load_bundle(bundle):
    bundle = Path(bundle)
    protocol = read_json(bundle / 'protocol.json')
    require(protocol['schema'] == 'ruvllm-model-measurement/v1', 'Unsupported protocol')
    require(protocol['model']['repo_id'] == 'Qwen/Qwen3.5-4B'
            and protocol['model']['revision'] == '851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a', 'Wrong model pin')
    require(object_sha(protocol['generation']) == object_sha(dict(max_new_tokens=512, do_sample=False, thinking=False, seed=20261103)),
            'Unexpected decoding protocol')
    require(protocol['runtime_versions'] == dict(torch='2.11.0', transformers='5.12.1', peft='0.21.2', safetensors='0.7.0'),
            'Unexpected runtime pins')
    require(set(protocol['source_sha256']) == {'worker.py', 'measurement.py', 'prepare.py', 'model-assets.json'},
            'Incomplete source binding')
    for name, expected in protocol['source_sha256'].items():
        require(file_sha(HERE / name) == expected, f'Source changed: {name}')
    assets = read_json(HERE / 'model-assets.json')
    require(protocol['model']['asset_files'] == assets['asset_files'], 'Model asset pins changed')
    pins = read_json(HERE.parents[1] / 'research/vast_cross_model/models.json')
    expected_weights = next(row for row in pins if row['model'] == protocol['model']['repo_id'])['weights_sha256']
    require(protocol['model']['weight_files'] == expected_weights, 'Model shard pins changed')
    adapter_root = HERE.parents[1] / 'weights/coding-4b'
    require(protocol['adapter'] == dict(weights_sha256=file_sha(adapter_root / 'adapter_model.safetensors'),
                                       config_sha256=file_sha(adapter_root / 'adapter_config.json')), 'Adapter pin changed')
    require(file_sha(bundle / 'cases.json') == protocol['cases_sha256'], 'Case set changed')
    cases = read_json(bundle / 'cases.json')
    require(len(cases) == protocol['case_count'] == 256, 'Expected 256 frozen cases')
    require(len({case['id'] for case in cases}) == len(cases), 'Duplicate case ID')
    require(sum(case['kind'] == 'tool' for case in cases) == 192
            and sum(case['kind'] == 'math' for case in cases) == 64, 'Wrong panel coverage')
    for case in cases:
        require(isinstance(case['messages'], list) and isinstance(case['tools'], list), 'Invalid case prompt')
    return protocol, cases, file_sha(bundle / 'protocol.json')


def parse_calls(text, tools):
    blocks = re.findall(r'<tool_call>(.*?)</tool_call>', text, re.S)
    if not blocks and text.strip().startswith('{'):
        blocks = [text.strip()]
    else:
        require(text.count('<tool_call>') == len(blocks) == text.count('</tool_call>'), 'Incomplete tool envelope')
    result = []
    for block in blocks:
        block = block.strip()
        if block.startswith('{'):
            value = json.loads(block, object_pairs_hook=unique)
            require(set(value) == {'name', 'arguments'}, 'Unexpected tool fields')
            result.append(value)
            continue
        match = re.fullmatch(r'<function=([^<>]+)>(.*?)</function>', block, re.S)
        require(match is not None, 'Malformed tool call')
        name, body = match.groups()
        schema = next(tool['function']['parameters']['properties'] for tool in tools if tool['function']['name'] == name)
        arguments, offset = {}, 0
        for parameter in re.finditer(r'<parameter=([^<>]+)>(.*?)</parameter>', body, re.S):
            require(not body[offset:parameter.start()].strip(), 'Unparsed argument content')
            key, value = parameter.groups()
            require(key in schema and key not in arguments, 'Unknown/duplicate argument')
            value = value.removeprefix('\n').removesuffix('\n')
            if schema[key]['type'] == 'integer':
                require(re.fullmatch(r'-?\d+', value) is not None, 'Invalid integer')
                value = int(value)
            elif schema[key]['type'] == 'boolean':
                require(value in ('true', 'false'), 'Invalid boolean')
                value = value == 'true'
            arguments[key] = value
            offset = parameter.end()
        require(not body[offset:].strip(), 'Trailing argument content')
        result.append(dict(name=name, arguments=arguments))
    return result


def grade(case, row):
    text = row['text']
    closed = text.rfind('<think>') <= text.rfind('</think>')
    visible = text.rsplit('</think>', 1)[-1]
    visible = re.sub(r'<\|(?:im_end|endoftext)\|>', '', visible).strip()
    completed = row['finish_reason'] == 'stop' and closed
    try:
        if case['kind'] == 'math':
            values = re.findall(r'\\boxed\{\s*([+-]?\d+)\s*\}', visible)
            correct = bool(values) and int(values[-1]) == case['answer']
        else:
            correct = object_sha(parse_calls(visible, case['tools'])) == object_sha([case['expected']])
    except (ValueError, TypeError, KeyError, StopIteration):
        correct = False
    return dict(id=case['id'], kind=case['kind'], correct=bool(correct), completed=completed,
                delivered_correct=bool(correct and completed), truncated=row['finish_reason'] == 'length')


def read_run(folder, arm, protocol, cases, protocol_sha, allow_fixture=False):
    folder = Path(folder)
    runtime = read_json(folder / 'runtime.json')
    require(runtime['schema'] == 'ruvllm-measurement-runtime/v1'
            and {key: value.split('+')[0] for key, value in runtime['versions'].items()} == protocol['runtime_versions'],
            'Runtime version mismatch')
    require(runtime['device'] == 'cuda' and type(runtime['fixture']) is bool, 'CUDA runtime binding required')
    require(not runtime['fixture'] or allow_fixture, 'Fixture results are not real inference')
    require(isinstance(runtime['gpu_name'], str) and runtime['gpu_name']
            and isinstance(runtime['gpu_uuid'], str) and runtime['gpu_uuid']
            and type(runtime['gpu_total_memory_bytes']) is int and runtime['gpu_total_memory_bytes'] > 0, 'Invalid GPU evidence')
    runtime_sha = file_sha(folder / 'runtime.json')
    model_sha = object_sha({**protocol['model']['weight_files'], **protocol['model']['asset_files']})
    bindings = dict(run_id=protocol['run_id'], arm=arm, protocol_sha256=protocol_sha,
                    model_sha256=model_sha, adapter_sha256=protocol['adapter']['weights_sha256'], runtime_sha256=runtime_sha)
    completion = read_json(folder / 'completion.json')
    require(completion['schema'] == 'ruvllm-measurement-completion/v1'
            and completion['inference_performed'] is True, 'Incomplete inference receipt')
    require(all(completion.get(key) == value for key, value in bindings.items()), 'Completion binding mismatch')
    require(completion['generation_count'] == len(cases)
            and completion['responses_sha256'] == file_sha(folder / 'responses.jsonl'), 'Incomplete/changed response file')
    path = folder / 'responses.jsonl'
    require(path.stat().st_size <= 64 * 1024 * 1024, 'Response file too large')
    by_id = {case['id']: case for case in cases}
    records = {}
    with path.open() as stream:
        for line in stream:
            require(len(line.encode()) <= 1024 * 1024, 'Response row too large')
            row = json.loads(line, object_pairs_hook=unique)
            require(row['id'] in by_id and row['id'] not in records, 'Unknown/duplicate case response')
            require(all(row.get(key) == value for key, value in bindings.items()), 'Response binding mismatch')
            case = by_id[row['id']]
            require(row['prompt_sha256'] == object_sha(dict(messages=case['messages'], tools=case['tools'])), 'Prompt mismatch')
            require(row['finish_reason'] in ('stop', 'length') and isinstance(row['text'], str), 'Invalid completion')
            ids = row['token_ids']
            require(type(row['output_tokens']) is int and isinstance(ids, list) and all(type(token) is int and token >= 0 for token in ids)
                    and 0 < len(ids) == row['output_tokens'] <= protocol['generation']['max_new_tokens'], 'Invalid token accounting')
            require((row['finish_reason'] == 'stop') == (ids[-1] in (248044, 248046)), 'Termination reason mismatch')
            require(row['finish_reason'] != 'length' or len(ids) == protocol['generation']['max_new_tokens'], 'Truncation count mismatch')
            require(type(row['latency_seconds']) in (int, float) and math.isfinite(row['latency_seconds'])
                    and row['latency_seconds'] >= 0, 'Invalid latency')
            records[row['id']] = {**grade(case, row), 'output_tokens': row['output_tokens'],
                                  'latency_seconds': row['latency_seconds']}
    require(set(records) == set(by_id), 'Missing case responses')
    return [records[case['id']] for case in cases], runtime_sha, runtime['fixture']


def paired(left, right):
    gains = sum(not a['delivered_correct'] and b['delivered_correct'] for a, b in zip(left, right, strict=True))
    losses = sum(a['delivered_correct'] and not b['delivered_correct'] for a, b in zip(left, right, strict=True))
    n = gains + losses
    p = min(1.0, 2 * sum(math.comb(n, k) for k in range(min(gains, losses) + 1)) / 2**n) if n else 1.0
    return dict(gains=gains, regressions=losses, exact_mcnemar_p=p)


def score_runs(bundle, base_run, adapter_run, allow_fixture=False):
    protocol, cases, protocol_sha = load_bundle(bundle)
    base, base_runtime, fixture = read_run(base_run, 'base', protocol, cases, protocol_sha, allow_fixture)
    tuned, tuned_runtime, other_fixture = read_run(adapter_run, 'adapter', protocol, cases, protocol_sha, allow_fixture)
    require(base_runtime == tuned_runtime, 'Arms used different runtime/GPU bindings')
    panels = {}
    def totals(rows):
        count = len(rows)
        return dict(count=count, delivered_correct=sum(row['delivered_correct'] for row in rows),
                    completed=sum(row['completed'] for row in rows), truncated=sum(row['truncated'] for row in rows),
                    delivered_accuracy=sum(row['delivered_correct'] for row in rows)/count,
                    completion_rate=sum(row['completed'] for row in rows)/count,
                    output_tokens=sum(row['output_tokens'] for row in rows),
                    generation_seconds=sum(row['latency_seconds'] for row in rows))
    for kind in ('tool', 'math'):
        left = [row for row in base if row['kind'] == kind]
        right = [row for row in tuned if row['kind'] == kind]
        original = sum(row['delivered_correct'] for row in left)
        changed = sum(row['delivered_correct'] for row in right)
        panels[kind] = dict(count=len(left), base_correct=original, adapter_correct=changed,
                            base=totals(left), adapter=totals(right),
                            delta_percentage_points=100*(changed-original)/len(left), **paired(left, right))
    reasons = []
    if panels['tool']['adapter_correct'] <= panels['tool']['base_correct']:
        reasons.append('no_primary_improvement')
    if panels['tool']['exact_mcnemar_p'] >= 0.05:
        reasons.append('primary_not_statistically_confirmed')
    if panels['math']['adapter_correct'] < panels['math']['base_correct']:
        reasons.append('math_anchor_regressed')
    if sum(row['completed'] for row in tuned) < sum(row['completed'] for row in base):
        reasons.append('completion_regressed')
    if fixture or other_fixture:
        reasons.append('synthetic_fixture_evidence')
    return dict(schema='ruvllm-paired-model-improvement/v1', run_id=protocol['run_id'],
                protocol_sha256=protocol_sha, runtime_sha256=base_runtime, panels=panels,
                totals=dict(base=totals(base), adapter=totals(tuned)),
                evidence_mode='fixture' if fixture or other_fixture else 'gpu-inference',
                research_improvement_confirmed=not reasons, reasons=reasons, production_activation=False,
                limitations=['Synthetic tool and arithmetic distribution; no broad coding generalization',
                             'One fixed seed and runtime; caller filesystem and executor remain trusted'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('bundle', 'base-run', 'adapter-run', 'out'):
        parser.add_argument('--' + name, required=True, type=Path)
    args = parser.parse_args()
    report = score_runs(args.bundle, args.base_run, args.adapter_run)
    args.out.mkdir(parents=False, exist_ok=False)
    (args.out / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
