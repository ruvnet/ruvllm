"""Scoring contract tests use synthetic records and cannot establish model improvement."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE))
import measurement
import prepare


class ScoringContracts(unittest.TestCase):
    def setup_runs(self, root):
        bundle = root / 'bundle'
        prepare.prepare(bundle)
        protocol, cases, protocol_sha = measurement.load_bundle(bundle)
        runtime = dict(schema='ruvllm-measurement-runtime/v1', versions=protocol['runtime_versions'],
                       device='cuda', gpu_name='MOCK-NOT-A-GPU', gpu_uuid='MOCK-UUID',
                       gpu_total_memory_bytes=1, torch_cuda_version='MOCK', python_version='MOCK', fixture=True)
        for arm in ('base', 'adapter'):
            folder = root / arm
            folder.mkdir()
            self.write(folder / 'runtime.json', runtime)
            bindings = dict(run_id=protocol['run_id'], arm=arm, protocol_sha256=protocol_sha,
                            model_sha256=measurement.object_sha({**protocol['model']['weight_files'], **protocol['model']['asset_files']}),
                            adapter_sha256=protocol['adapter']['weights_sha256'],
                            runtime_sha256=measurement.file_sha(folder / 'runtime.json'))
            rows = []
            for case in cases:
                # Give synthetic adapter perfect answers and base wrong answers.
                text = json.dumps(case['expected']) if case['kind'] == 'tool' else '\\boxed{' + str(case['answer']) + '}'
                if arm == 'base':
                    text = 'invalid response'
                rows.append(dict(id=case['id'], **bindings,
                                 prompt_sha256=measurement.object_sha(dict(messages=case['messages'], tools=case['tools'])),
                                 text=text, finish_reason='stop', token_ids=[248046], output_tokens=1, latency_seconds=0.1))
            self.write_rows(folder, rows)
            self.write(folder / 'completion.json', dict(schema='ruvllm-measurement-completion/v1', **bindings,
                responses_sha256=measurement.file_sha(folder / 'responses.jsonl'), generation_count=len(rows), inference_performed=True))
        return bundle

    @staticmethod
    def write(path, value):
        path.write_text(json.dumps(value, indent=2) + '\n')

    @staticmethod
    def write_rows(folder, rows):
        (folder / 'responses.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))

    def mutate(self, folder, action):
        rows = [json.loads(line) for line in (folder / 'responses.jsonl').read_text().splitlines()]
        action(rows)
        self.write_rows(folder, rows)
        receipt = json.loads((folder / 'completion.json').read_text())
        receipt['responses_sha256'] = measurement.file_sha(folder / 'responses.jsonl')
        self.write(folder / 'completion.json', receipt)

    def test_fixture_scores_are_never_a_real_improvement_claim(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bundle = self.setup_runs(root)
            with self.assertRaisesRegex(ValueError, 'Fixture'):
                measurement.score_runs(bundle, root / 'base', root / 'adapter')
            report = measurement.score_runs(bundle, root / 'base', root / 'adapter', allow_fixture=True)
            self.assertEqual(report['panels']['tool']['adapter_correct'], 192)
            self.assertEqual(report['panels']['math']['adapter_correct'], 64)
            self.assertFalse(report['research_improvement_confirmed'])
            self.assertIn('synthetic_fixture_evidence', report['reasons'])
            self.assertEqual(report['totals']['adapter']['completed'], 256)

    def test_duplicate_missing_wrong_prompt_and_stale_run_are_rejected(self):
        changes = [lambda rows: rows.__setitem__(1, rows[0]), lambda rows: rows.pop(),
                   lambda rows: rows[0].__setitem__('prompt_sha256', '0'*64),
                   lambda rows: rows[0].__setitem__('run_id', 'old-run'),
                   lambda rows: rows[0].__setitem__('model_sha256', '0'*64)]
        for change in changes:
            with self.subTest(change=change), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                bundle = self.setup_runs(root)
                self.mutate(root / 'adapter', change)
                with self.assertRaises(ValueError):
                    measurement.score_runs(bundle, root / 'base', root / 'adapter', allow_fixture=True)

    def test_token_count_and_termination_cannot_be_fabricated(self):
        for change in [lambda rows: rows[0].__setitem__('output_tokens', 2),
                       lambda rows: rows[0].__setitem__('finish_reason', 'length'),
                       lambda rows: rows[0].__setitem__('latency_seconds', -1)]:
            with self.subTest(change=change), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                bundle = self.setup_runs(root)
                self.mutate(root / 'adapter', change)
                with self.assertRaises(ValueError):
                    measurement.score_runs(bundle, root / 'base', root / 'adapter', allow_fixture=True)

    def test_changed_runtime_and_case_bytes_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bundle = self.setup_runs(root)
            runtime = json.loads((root / 'adapter/runtime.json').read_text())
            runtime['gpu_uuid'] = 'DIFFERENT-GPU'
            self.write(root / 'adapter/runtime.json', runtime)
            with self.assertRaises(ValueError):
                measurement.score_runs(bundle, root / 'base', root / 'adapter', allow_fixture=True)
            (bundle / 'cases.json').write_text('[]')
            with self.assertRaisesRegex(ValueError, 'Case set changed'):
                measurement.load_bundle(bundle)

    def test_tool_arguments_require_exact_json_types(self):
        case = dict(id='typed', kind='tool', expected=dict(name='Read', arguments={'offset': 1}), tools=[])
        row = dict(text='{"name":"Read","arguments":{"offset":true}}', finish_reason='stop')
        self.assertFalse(measurement.grade(case, row)['correct'])

    def test_truncated_or_unclosed_correct_answer_is_not_delivered(self):
        case = dict(id='arithmetic', kind='math', answer=4)
        for text, finish in [('<think>\\boxed{4}', 'stop'), ('\\boxed{4}', 'length')]:
            self.assertFalse(measurement.grade(case, dict(text=text, finish_reason=finish))['delivered_correct'])

    def test_exact_paired_test_counts_gains_and_regressions(self):
        left = [dict(delivered_correct=False)] * 6
        right = [dict(delivered_correct=True)] * 6
        self.assertEqual(measurement.paired(left, right), dict(gains=6, regressions=0, exact_mcnemar_p=0.03125))


if __name__ == '__main__':
    unittest.main()
