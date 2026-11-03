"""Tamper checks on real copied evidence; no generated code or network execution."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location('verify_benchmarks', ROOT/'scripts/verify_benchmarks.py')
VERIFY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VERIFY)


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        shutil.copytree(ROOT/'benchmarks', self.root/'benchmarks', ignore=shutil.ignore_patterns('__pycache__'))
        (self.root/'scripts').mkdir()
        shutil.copy2(ROOT/'scripts/verify_benchmarks.py', self.root/'scripts/verify_benchmarks.py')
        self.data = self.root/'benchmarks'

    def tearDown(self):
        self.temp.cleanup()

    def rewrite(self, name, value, jsonl=False):
        p = self.data/name
        p.write_text(''.join(json.dumps(row)+'\n' for row in value) if jsonl else json.dumps(value))
        manifest = VERIFY.read(self.data/'manifest.json')
        record = manifest['files'][name]
        record['sha256'] = VERIFY.sha(p)
        record['bytes'] = p.stat().st_size
        if 'origin' in record:
            record['origin']['sha256'] = record['sha256']
        (self.data/'manifest.json').write_text(json.dumps(manifest))

    def test_complete_evidence(self):
        result = VERIFY.verify(self.data)
        self.assertEqual(result['generations_verified'], 462)
        self.assertFalse(result['generated_code_reexecuted'])
        self.assertFalse(result['local_08b']['behavioral_gate_passed'])
        self.assertEqual(result['local_4b']['arms']['selected-humaneval']['passed'], 128)

    def test_corruption_rejected(self):
        with (self.data/'local-4b/baseline-release/responses.jsonl').open('a') as f:
            f.write(' ')
        with self.assertRaisesRegex(ValueError, 'Byte count|Digest'):
            VERIFY.verify(self.data)

    def test_extra_hidden_file_rejected(self):
        (self.data/'.private-key').write_text('not for publication')
        with self.assertRaisesRegex(ValueError, 'Unexpected evidence file'):
            VERIFY.verify(self.data)

    def test_missing_file_rejected(self):
        (self.data/'local-08b/cases.json').unlink()
        with self.assertRaisesRegex(ValueError, 'Missing evidence'):
            VERIFY.verify(self.data)

    def test_symlink_rejected(self):
        target = self.data/'local-08b/cases.json'
        target.unlink()
        target.symlink_to(self.data/'local-08b/plan.json')
        with self.assertRaisesRegex(ValueError, 'Symlink'):
            VERIFY.verify(self.data)

    @unittest.skipUnless(hasattr(os, 'mkfifo'), 'Requires Unix FIFO')
    def test_fifo_rejected_without_open(self):
        target = self.data/'local-08b/cases.json'
        target.unlink()
        os.mkfifo(target)
        with self.assertRaisesRegex(ValueError, 'Nonregular'):
            VERIFY.verify(self.data)

    def test_manifest_traversal_rejected(self):
        manifest = VERIFY.read(self.data/'manifest.json')
        manifest['files']['../account.json'] = manifest['files'].pop('README.md')
        (self.data/'manifest.json').write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, 'allowlist'):
            VERIFY.verify(self.data)

    def test_math_text_rescored_after_rehash(self):
        name = 'local-08b/release-generations.jsonl'
        data = VERIFY.rows(self.data/name)
        self.assertTrue(data[0]['delivered_correct'])
        data[0]['text'] = r'\boxed{0}'
        self.rewrite(name, data, jsonl=True)
        with self.assertRaisesRegex(ValueError, 'Independent math rescore'):
            VERIFY.verify(self.data)

    def test_math_duplicate_row_rejected_after_rehash(self):
        name = 'local-08b/release-generations.jsonl'
        data = VERIFY.rows(self.data/name)
        data[1] = data[0]
        self.rewrite(name, data, jsonl=True)
        with self.assertRaisesRegex(ValueError, 'Missing, duplicate'):
            VERIFY.verify(self.data)

    def test_reopened_thinking_rejected_after_rehash(self):
        name = 'local-08b/release-generations.jsonl'
        data = VERIFY.rows(self.data/name)
        row = next(r for r in data if r['arm'].endswith('thinking') and r['delivered_correct'])
        row['text'] += '<think>unfinished'
        self.rewrite(name,data,jsonl=True)
        with self.assertRaisesRegex(ValueError, 'Independent math rescore'):
            VERIFY.verify(self.data)

    def test_saved_verdict_summary_recomputed_after_rehash(self):
        name = 'local-4b/selected-humaneval/summary.json'
        data = VERIFY.read(self.data/name)
        data['passed'] = 164
        self.rewrite(name,data)
        with self.assertRaisesRegex(ValueError, 'Saved summary differs'):
            VERIFY.verify(self.data)

    def test_paired_statistics_recomputed_after_rehash(self):
        name = 'local-4b/paired-verdicts.json'
        data = VERIFY.read(self.data/name)
        data['external_paired']['exact_mcnemar_p'] = 0.001
        self.rewrite(name,data)
        with self.assertRaisesRegex(ValueError, 'Paired verdict report'):
            VERIFY.verify(self.data)

    def test_rollback_mismatch_rejected_after_rehash(self):
        name = 'local-08b/rollback-verification.json'
        data = VERIFY.read(self.data/name)
        data['restored_sha256'] = '0'*64
        self.rewrite(name,data)
        with self.assertRaisesRegex(ValueError, 'Rollback receipt'):
            VERIFY.verify(self.data)

    def test_json_duplicate_and_nonfinite_rejected(self):
        for data in (b'{"a":1,"a":2}', b'{"a":NaN}', b'{"a":Infinity}'):
            with self.assertRaises(ValueError):
                VERIFY.decode(data)

    def test_native_timing_receipt_rejected_after_rehash(self):
        name = 'local-08b/native-release-alpha-100.json'
        data = VERIFY.read(self.data/name)
        data['wall_seconds'] = 0.1
        self.rewrite(name,data)
        with self.assertRaisesRegex(ValueError, 'Historical native timing'):
            VERIFY.verify(self.data)

    def test_coding_adapter_receipt_rejected_after_rehash(self):
        name = 'local-4b/coding-integration.json'
        data = VERIFY.read(self.data/name)
        data['adapter_sha256'] = '0'*64
        self.rewrite(name,data)
        with self.assertRaisesRegex(ValueError, 'Historical coding fixture binding'):
            VERIFY.verify(self.data)


if __name__ == '__main__':
    unittest.main()
