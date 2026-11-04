"""Build a Markdown dashboard from a complete, verified pair of saved runs."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re

from measurement import file_sha, grade, load_bundle, object_sha, parse_calls, score_runs


def failure_reason(case, row, verdict):
    if verdict['truncated']:
        return 'Truncation'
    if not verdict['completed']:
        return 'Unfinished reasoning'
    visible = re.sub(r'<\|(?:im_end|endoftext)\|>', '', row['text'].rsplit('</think>', 1)[-1]).strip()
    if case['kind'] == 'math':
        return 'Wrong arithmetic' if re.search(r'\\boxed\{\s*[+-]?\d+\s*\}', visible) else 'Instruction or answer format'
    try:
        calls = parse_calls(visible, case['tools'])
    except StopIteration:
        return 'Wrong tool'
    except (ValueError, TypeError, KeyError):
        return 'Malformed tool call'
    if len(calls) != 1:
        return 'Instruction or call count'
    expected = case['expected']
    if calls[0]['name'] != expected['name']:
        return 'Wrong tool'
    if object_sha(calls[0]['arguments']) != object_sha(expected['arguments']):
        return 'Wrong arguments'
    raise ValueError('Failed tool response has no classification')


def build_dashboard(bundle, base, adapter, *, allow_fixture=False):
    report = score_runs(bundle, base, adapter, allow_fixture=allow_fixture)
    _, cases, _ = load_bundle(bundle)

    def grades(folder):
        receipt = json.loads((folder / 'completion.json').read_text())
        responses = folder / 'responses.jsonl'
        if file_sha(responses) != receipt['responses_sha256']:
            raise ValueError('Raw responses changed after scoring')
        with responses.open() as stream:
            return {row['id']: row for row in map(json.loads, stream)}

    left, right = grades(base), grades(adapter)
    categories = Counter()
    details = []
    for case in cases:
        case_id = case['id']
        verdict = grade(case, right[case_id])
        if grade(case, left[case_id])['delivered_correct'] and not verdict['delivered_correct']:
            target = case['expected']['name'] if case['kind'] == 'tool' else 'Arithmetic'
            reason = failure_reason(case, right[case_id], verdict)
            categories[target, reason] += 1
            details.append((case_id, target, reason))
    if len(details) != sum(panel['regressions'] for panel in report['panels'].values()):
        raise ValueError('Regression count does not match the frozen scorer')

    decision = 'Keep for research' if report['research_improvement_confirmed'] else 'Reject for this panel'
    if report['evidence_mode'] == 'fixture':
        decision = 'Fixture only. No model decision.'
    lines = ['# RuVLLM paired adapter comparison', '',
             f'**Decision:** {decision}', '',
             f"Evidence mode: {report['evidence_mode']}",
             f"Protocol SHA256: `{report['protocol_sha256']}`", '',
             '| Panel | Base | Adapter | Gains | Regressions | Paired p |',
             '| --- | ---: | ---: | ---: | ---: | ---: |']
    for name, panel in report['panels'].items():
        lines.append(f"| {name} | {panel['base_correct']}/{panel['count']} | "
                     f"{panel['adapter_correct']}/{panel['count']} | {panel['gains']} | "
                     f"{panel['regressions']} | {panel['exact_mcnemar_p']:.4g} |")
    lines.extend(['', 'Gate failures: ' + (', '.join(report['reasons']) or 'none'),
                  '', '## Regressions by expected tool and observed failure', ''])
    if details:
        lines.extend(['| Expected tool | Observed failure | Count |', '| --- | --- | ---: |'])
        lines.extend(f'| {tool} | {reason} | {count} |' for (tool, reason), count in sorted(categories.items()))
        lines.extend(['', '| Case | Expected tool | Observed failure |', '| --- | --- | --- |'])
        lines.extend(f'| {case_id} | {tool} | {reason} |' for case_id, tool, reason in details)
    else:
        lines.append('No base passes became adapter failures.')
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('bundle', 'base', 'adapter', 'out'):
        parser.add_argument('--' + name, required=True, type=Path)
    args = parser.parse_args()
    dashboard = build_dashboard(args.bundle, args.base, args.adapter)
    with args.out.open('x', encoding='utf-8') as stream:
        stream.write(dashboard)


if __name__ == '__main__':
    main()
