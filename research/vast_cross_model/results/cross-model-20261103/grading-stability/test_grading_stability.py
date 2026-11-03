"""Synthetic audit control tests. No real responses or generated code are executed."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

SPEC = importlib.util.spec_from_file_location('grading_stability', Path(__file__).with_name('grading_stability.py'))
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


class StabilityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.output = self.root / 'audit.json'
        self.targets = []
        for arm in audit.ARM_IDS:
            for ident in audit.CASE_IDS:
                case = {'id': ident, 'kind': 'code', 'panel': 'humaneval164', 'tests': 'synthetic tests only'}
                raw = {'id': ident, 'kind': 'code', 'panel': 'humaneval164', 'text': 'synthetic, never executed',
                       'thinking': False, 'finish_reason': 'stop', 'output_tokens': 1}
                primary = {'id': ident, 'panel': 'humaneval164', 'answer_correct': True, 'completed': True,
                           'delivered_correct': True, 'truncated': False, 'output_tokens': 1, 'reason': 'passed'}
                self.targets.append({'arm': arm, 'case': case, 'raw': raw, 'primary': primary})
        self.provenance = {'protocol_sha256': audit.PROTOCOL, 'complete_primary_generations': 1851,
                           'source_and_evidence_sha256': {'primary/scored.json': 'a' * 64}, 'inventory_sha256': 'b' * 64}
        self.revalidate = Mock()
        self.calls = 0

    def grade(self, case, raw):
        self.calls += 1
        correct = self.calls % 2 == 1
        return {'id': case['id'], 'panel': case['panel'], 'answer_correct': correct, 'completed': True,
                'delivered_correct': correct, 'truncated': False, 'output_tokens': raw['output_tokens'],
                'reason': 'passed' if correct else 'execution or assertion failure'}

    def invoke(self, grade=None):
        return audit.execute(self.targets, self.provenance, self.revalidate, grade or self.grade, self.output)

    def test_fixed_600_calls_keep_all_reasons_and_disagreements_without_replacing_primary(self):
        original = deepcopy(self.targets)
        result = self.invoke()
        self.assertEqual(self.calls, 600)
        self.assertEqual(result['completed_repeat_calls'], 600)
        self.assertEqual(len(result['repeats']), 600)
        self.assertEqual(self.targets, original)
        self.assertFalse(result['model_selection'])
        self.assertFalse(result['primary_scores_replaced'])
        self.assertEqual(self.revalidate.call_count, 2)
        for row in result['cases']:
            self.assertEqual(row['repeat_count'], 30)
            self.assertEqual(row['repeat_answer_passes'], 15)
            self.assertEqual(row['answer_disagreements_with_primary'], 15)
            self.assertTrue(row['repeat_verdicts_vary'])
            self.assertEqual(row['reason_counts'], {'passed': 15, 'execution or assertion failure': 15})
        self.assertEqual(json.loads(self.output.read_text()), result)
        self.assertNotIn(str(self.root), self.output.read_text())
        self.assertEqual(self.output.stat().st_mode & 0o777, 0o444)
        self.assertFalse(self.output.with_name('audit.json.partial.jsonl').exists())

    def test_every_planned_target_is_mandatory(self):
        self.targets.pop()
        with self.assertRaisesRegex(ValueError, 'inventory'):
            self.invoke()
        self.assertEqual(self.calls, 0)
        self.assertEqual(list(self.root.iterdir()), [])

    def test_unknown_reason_cannot_publish_private_text(self):
        self.targets[0]['primary']['reason'] = '/private/operator/path'
        with self.assertRaisesRegex(ValueError, 'nonpublic reason'):
            self.invoke()
        self.assertEqual(self.calls, 0)
        self.assertEqual(list(self.root.iterdir()), [])

    def test_completed_flag_and_delivered_semantics_are_bound_to_raw(self):
        self.targets[0]['primary']['completed'] = False
        with self.assertRaisesRegex(ValueError, 'semantics'):
            self.invoke()
        self.assertEqual(self.calls, 0)

    def test_source_change_before_or_after_repeats_never_publishes_success(self):
        self.revalidate.side_effect = ValueError('changed before')
        with self.assertRaises(ValueError): self.invoke()
        self.assertEqual(self.calls, 0)
        self.assertFalse(self.output.exists())
        self.revalidate.side_effect = [None, ValueError('changed after')]
        with self.assertRaises(ValueError): self.invoke()
        self.assertEqual(self.calls, 600)
        self.assertFalse(self.output.exists())
        partial = self.output.with_name('audit.json.partial.jsonl')
        self.assertEqual(len(partial.read_text().splitlines()), 601)

    def test_exception_preserves_all_attempted_calls_without_error_message_paths(self):
        def failed(case, raw):
            if self.calls == 2:
                raise OSError('/private/sensitive/path')
            return self.grade(case, raw)
        with self.assertRaisesRegex(ValueError, 'partial journal'):
            self.invoke(failed)
        partial = self.output.with_name('audit.json.partial.jsonl').read_text()
        rows = [json.loads(line) for line in partial.splitlines()]
        self.assertEqual(len(rows), 4)
        self.assertEqual(rows[-1]['repeat'], 3)
        self.assertEqual(rows[-1]['status'], 'error')
        self.assertEqual(rows[-1]['error_class'], 'OSError')
        self.assertNotIn('/private/', partial)
        self.assertFalse(self.output.exists())

    def test_interruption_records_attempt_without_completion_claim(self):
        with self.assertRaises(KeyboardInterrupt):
            self.invoke(Mock(side_effect=KeyboardInterrupt))
        rows = [json.loads(line) for line in self.output.with_name('audit.json.partial.jsonl').read_text().splitlines()]
        self.assertEqual(rows[-1]['status'], 'interrupted')
        self.assertFalse(self.output.exists())

    def test_existing_output_or_partial_refuses_before_grading(self):
        for path in (self.output, self.output.with_name('audit.json.partial.jsonl')):
            path.write_text('retain')
            with self.assertRaises(ValueError): self.invoke()
            self.assertEqual(path.read_text(), 'retain')
            path.unlink()
        self.assertEqual(self.calls, 0)

    def test_atomic_publication_does_not_replace_existing_file(self):
        self.output.write_text('retain')
        with self.assertRaises(FileExistsError):
            audit.atomic_publish({'synthetic': True}, self.output)
        self.assertEqual(self.output.read_text(), 'retain')
        self.assertEqual(list(self.root.glob('.grading-stability-*')), [])

    def test_incomplete_full_evidence_gate_precedes_grader_loading(self):
        for name in ('work', 'primary', 'inputs'):
            (self.root / name).mkdir()
        verifier = SimpleNamespace(verify_evidence=Mock(side_effect=ValueError('Incomplete primary experiment')))
        with patch.object(audit, 'trusted_verifier', return_value=verifier), \
                patch.object(audit, 'source_gate', return_value={}), \
                patch.object(audit, 'snapshot', return_value={}), \
                patch.object(audit, 'frozen_grade') as grader:
            with self.assertRaisesRegex(ValueError, 'Incomplete'):
                audit.audit(self.root / 'work', self.root / 'primary', self.root / 'inputs', self.output)
            grader.assert_not_called()
            verifier.verify_evidence.assert_called_once()
        self.assertFalse(self.output.exists())

    def test_output_cannot_change_primary_tree(self):
        for name in ('work', 'primary', 'inputs'):
            (self.root / name).mkdir()
        with patch.object(audit, 'prepare') as prepare:
            for name in ('work', 'primary', 'inputs'):
                with self.assertRaisesRegex(ValueError, 'separate'):
                    audit.audit(self.root / 'work', self.root / 'primary', self.root / 'inputs', self.root / name / 'audit.json')
            prepare.assert_not_called()

    def test_original_frozen_scorer_is_mandatory(self):
        verifier = SimpleNamespace(protocol_inputs=Mock(return_value=({'total_generations': 1851}, audit.PROTOCOL)))
        with patch.object(audit, 'file_sha', return_value='f' * 64):
            with self.assertRaisesRegex(ValueError, 'original frozen'):
                audit.source_gate(verifier)


if __name__ == '__main__':
    unittest.main()
