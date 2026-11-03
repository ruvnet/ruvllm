"""Synthetic publication fixtures only. No real benchmark or cleanup claim."""
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch

from scripts import verify_cloud_publication as verify


class PublicationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.publication = self.root / 'publication'
        self.publication.mkdir()
        self.archive = self.root / 'synthetic-evidence.tar.gz'
        self.manifest = 'a' * 64
        self.timing = {'schema': 'synthetic-timing-only', 'protocol_sha256': verify.PROTOCOL,
                       'analysis_sha256': 'b' * 64, 'arms': {'synthetic': {'tokens': 4}}}
        keys = ('destruction cleanup_summary run_spec rental guard_config guard_deployment guard_identity '
                'guard_readiness guard_execution cleanup_complete cleanup_intent absence_proof absence_observation_1 '
                'absence_observation_2 job_absence identity_absence grant_absence grant_before guard_stop continuation_intent '
                'effective_quote creation_intent rental_observation shared_budget scientific_run_receipt').split()
        self.closeout = {'schema': 'ruvllm.cloud-closeout.v1', 'status': 'closed', 'protocol_sha256': verify.PROTOCOL,
            'scientific_bundle_manifest_sha256': self.manifest, 'generated_at': '2026-11-03T08:00:00Z',
            'resource_binding_sha256': 'c' * 64,
            'budget': {'authorized_cap_micro_usd': 200000000, 'total_reserved_micro_usd': 34458369,
                       'this_run_reserved_micro_usd': 30000000, 'reservations_are_invoices': False},
            'estimate': {'rental_started_at': '2026-11-03T06:00:00Z', 'rental_absence_confirmed_at': '2026-11-03T07:00:00Z',
                'elapsed_seconds': 3600, 'gpu_storage_hourly_usd': 1.25, 'gpu_storage_elapsed_usd': 1.25,
                'download_allowance_gb': 200, 'download_rate_usd_per_gb': .002, 'download_allowance_usd': .4,
                'upload_allowance_gb': 10, 'upload_rate_usd_per_gb': .003, 'upload_allowance_usd': .03,
                'transfer_allowances_are_enforced_caps': False, 'transfer_actual_bytes': None,
                'guard_reserved_usd': 2, 'guard_actual_usd': None, 'invoice_usd': None, 'final_invoice_reconciled': False},
            'cleanup': {'cleanup_verified': True, 'gpu_absence_verified': True, 'job_absent': True,
                'service_account_absent': True, 'secret_binding_absent': True, 'independent_absence_reads': 2,
                'absence_observed_at': ['2026-11-03T07:00:05Z', '2026-11-03T07:00:10Z'],
                'matching_owned_resources_remaining': 0, 'completed_at': '2026-11-03T07:02:00Z'},
            'private_source_sha256': {key: 'd' * 64 for key in keys},
            'authority': 'Operator receipt consistency only; no provider signature, live API check, invoice reconciliation or production activation.'}
        self.contents = {'summary.json': b'{"synthetic":true}\n', 'scored.json': b'{"synthetic":[]}\n',
            'REPORT.md': b'Synthetic test fixture only.\n', 'timing.json': self.encode(self.timing),
            'closeout.json': self.encode(self.closeout)}
        for name, data in self.contents.items():
            (self.publication / name).write_bytes(data)
        self.members = [(self.member('synthetic/' + 'results/' + name, data), data)
                        for name, data in self.contents.items() if name in ('summary.json', 'scored.json', 'REPORT.md')]
        self.history = {'attempts': [{'status': 'completed', 'protocol_sha256': verify.PROTOCOL,
                                    'origin_receipt_sha256': 'd' * 64, 'completed_at': '2026-11-03T06:45:00Z'}]}
        history = self.encode(self.history)
        self.members.append((self.member('synthetic/history/runtime-attempts.json', history), history))
        marker = self.root / 'archive-code-must-not-run'
        malicious = ('from pathlib import Path\nPath(' + repr(str(marker)) + ').write_text("unsafe")\n').encode()
        self.members.append((self.member('synthetic/verify.py', malicious), malicious))
        self.build_archive()
        self.metadata = {'schema': 'ruvllm.cloud-publication.v1', 'protocol_sha256': verify.PROTOCOL,
            'archive': {'filename': self.archive.name, 'bytes': self.archive.stat().st_size,
                        'sha256': self.sha(self.archive.read_bytes()), 'top_directory': 'synthetic'},
            'scientific_bundle_manifest_sha256': self.manifest,
            'files': {name: {'bytes': len(data), 'sha256': self.sha(data)} for name, data in self.contents.items()}}
        self.save_metadata()
        bundle_patch = patch.object(verify, 'trusted_bundle_check', return_value={'read_only_verification': True,
            'protocol_sha256': verify.PROTOCOL, 'generations': 1851, 'manifest_sha256': self.manifest})
        timing_patch = patch.object(verify, 'trusted_timing', return_value=self.timing)
        self.bundle_check, self.timing_check = bundle_patch.start(), timing_patch.start()
        self.addCleanup(bundle_patch.stop)
        self.addCleanup(timing_patch.stop)

    @staticmethod
    def encode(value):
        return (json.dumps(value, indent=2) + '\n').encode()

    @staticmethod
    def sha(data):
        return hashlib.sha256(data).hexdigest()

    @staticmethod
    def member(name, data=b''):
        item = tarfile.TarInfo(name)
        item.size, item.mode = len(data), 0o444
        return item

    def build_archive(self):
        with tarfile.open(self.archive, 'w:gz', format=tarfile.PAX_FORMAT) as output:
            for member, data in self.members:
                output.addfile(member, io.BytesIO(data) if member.isfile() else None)

    def save_metadata(self):
        (self.publication / 'publication.json').write_bytes(self.encode(self.metadata))

    def rebind_archive(self):
        self.build_archive()
        self.metadata['archive'].update(bytes=self.archive.stat().st_size, sha256=self.sha(self.archive.read_bytes()))
        self.save_metadata()

    def replace_checked_file(self, name, value):
        data = value if isinstance(value, bytes) else self.encode(value)
        (self.publication / name).write_bytes(data)
        self.metadata['files'][name] = {'bytes': len(data), 'sha256': self.sha(data)}
        self.save_metadata()

    def invoke(self):
        return verify.verify(self.publication, self.archive)

    def test_complete_synthetic_boundary_calls_trusted_checks_and_never_archive_code(self):
        result = self.invoke()
        self.assertTrue(result['verified'])
        self.assertFalse(result['archive_code_executed'])
        self.assertFalse(result['private_cleanup_receipts_authenticated'])
        self.bundle_check.assert_called_once()
        self.timing_check.assert_called_once()
        self.assertFalse((self.root / 'archive-code-must-not-run').exists())
        self.assertTrue(verify.verify(self.publication / 'publication.json', self.archive)['verified'])

    def test_archive_digest_checked_before_parser_or_trusted_checks(self):
        self.archive.write_bytes(b'invalid archive content')
        with patch.object(verify.tarfile, 'open', side_effect=AssertionError('must not parse')):
            with self.assertRaises(ValueError): self.invoke()
        self.bundle_check.assert_not_called()

    def test_aliased_temporary_directory_is_canonical_before_trusted_checks(self):
        destination = self.root / 'canonical-temp'
        destination.mkdir()
        alias = self.root / 'temp-alias'
        alias.symlink_to(destination, target_is_directory=True)
        with patch.object(verify.tempfile, 'TemporaryDirectory') as temporary:
            temporary.return_value.__enter__.return_value = str(alias)
            self.assertTrue(self.invoke()['verified'])
        checked_root = self.bundle_check.call_args.args[0]
        self.assertEqual(checked_root, destination / 'synthetic')
        self.assertEqual(checked_root, checked_root.resolve(strict=True))
        self.assertEqual(self.timing_check.call_args.args[0], checked_root)

    def test_wrong_protocol_and_unknown_publication_field_rejected(self):
        for field, value in (('protocol_sha256', 'f' * 64), ('private_account', {'id': 'unwanted'})):
            saved = deepcopy(self.metadata)
            self.metadata[field] = value
            self.save_metadata()
            with self.subTest(field=field), self.assertRaises(ValueError): self.invoke()
            self.metadata = saved

    def test_each_checked_file_is_required_and_bound(self):
        for name in verify.FILES:
            path = self.publication / name
            data = path.read_bytes()
            path.unlink()
            with self.subTest(name=name), self.assertRaises(FileNotFoundError): self.invoke()
            path.write_bytes(data + b'x')
            with self.subTest(tamper=name), self.assertRaises(ValueError): self.invoke()
            path.write_bytes(data)

    def test_checked_summary_rehash_cannot_override_archive_result(self):
        self.replace_checked_file('summary.json', b'{"forged":true}\n')
        with self.assertRaisesRegex(ValueError, 'result differs'): self.invoke()

    def test_timing_rehash_cannot_override_recomputation(self):
        changed = deepcopy(self.timing)
        changed['arms']['synthetic']['tokens'] = 99
        self.replace_checked_file('timing.json', changed)
        with self.assertRaisesRegex(ValueError, 'timing differs'): self.invoke()

    def test_timing_comparison_preserves_json_scalar_types(self):
        self.timing['arms']['synthetic']['tokens'] = 1
        changed = deepcopy(self.timing)
        changed['arms']['synthetic']['tokens'] = True
        self.replace_checked_file('timing.json', changed)
        with self.assertRaisesRegex(ValueError, 'timing differs'): self.invoke()

    def test_truncated_gzip_trailer_rejected_after_digest_rebind(self):
        self.archive.write_bytes(self.archive.read_bytes()[:-8])
        self.metadata['archive'].update(bytes=self.archive.stat().st_size, sha256=self.sha(self.archive.read_bytes()))
        self.save_metadata()
        with self.assertRaises(EOFError): self.invoke()

    def test_scientific_bundle_proof_requires_all_generations_and_exact_manifest(self):
        for key, value in (('generations', 1850), ('protocol_sha256', 'e' * 64), ('manifest_sha256', 'e' * 64)):
            old = self.bundle_check.return_value[key]
            self.bundle_check.return_value[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError): self.invoke()
            self.bundle_check.return_value[key] = old

    def test_unsafe_archive_paths_rejected_even_after_digest_rebind(self):
        original = list(self.members)
        for name in ('../escape', '/absolute', 'synthetic/../escape', 'other/file', 'synthetic//alias', 'synthetic/./alias', 'synthetic\\file'):
            self.members = original + [(self.member(name, b'x'), b'x')]
            self.rebind_archive()
            with self.subTest(name=name), self.assertRaises(ValueError): self.invoke()

    def test_duplicate_archive_member_rejected(self):
        self.members.append(self.members[0])
        self.rebind_archive()
        with self.assertRaisesRegex(ValueError, 'Duplicate'): self.invoke()

    def test_links_directories_and_special_archive_entries_rejected(self):
        original = list(self.members)
        for kind in (tarfile.SYMTYPE, tarfile.LNKTYPE, tarfile.FIFOTYPE, tarfile.CHRTYPE, tarfile.DIRTYPE):
            item = self.member('synthetic/untrusted')
            item.type, item.linkname = kind, '/outside'
            self.members = original + [(item, b'')]
            self.rebind_archive()
            with self.subTest(kind=kind), self.assertRaises(ValueError): self.invoke()

    def test_member_count_and_expansion_are_bounded(self):
        with patch.object(verify, 'MAX_MEMBERS', 2):
            with self.assertRaisesRegex(ValueError, 'members'): self.invoke()
        stream = verify.BoundedReader(io.BytesIO(b'123456'))
        with patch.object(verify, 'MAX_TOTAL', 2), patch.object(verify, 'MAX_MEMBERS', 0):
            stream.read(2)
            with self.assertRaisesRegex(ValueError, 'Expanded'): stream.read(2)

    def test_closeout_rejects_unfinished_cleanup_and_wrong_bundle(self):
        for key in ('job_absent', 'service_account_absent', 'secret_binding_absent'):
            value = deepcopy(self.closeout)
            value['cleanup'][key] = False
            self.replace_checked_file('closeout.json', value)
            with self.subTest(key=key), self.assertRaises(ValueError): self.invoke()
        value = deepcopy(self.closeout)
        value['scientific_bundle_manifest_sha256'] = 'f' * 64
        self.replace_checked_file('closeout.json', value)
        with self.assertRaisesRegex(ValueError, 'bind'): self.invoke()

    def test_unknown_costs_cannot_be_invented_as_zero(self):
        for key in ('invoice_usd', 'guard_actual_usd', 'transfer_actual_bytes'):
            value = deepcopy(self.closeout)
            value['estimate'][key] = 0
            self.replace_checked_file('closeout.json', value)
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'null'): self.invoke()

    def test_closeout_run_receipt_must_match_verified_history(self):
        value = deepcopy(self.closeout)
        value['private_source_sha256']['scientific_run_receipt'] = 'e' * 64
        self.replace_checked_file('closeout.json', value)
        with self.assertRaisesRegex(ValueError, 'run receipt'): self.invoke()

    def test_scientific_completion_must_fall_within_rental(self):
        for completed in ('2026-11-03T05:59:00Z', '2026-11-03T07:01:00Z'):
            self.history['attempts'][-1]['completed_at'] = completed
            data = self.encode(self.history)
            self.members = [(item, content) for item, content in self.members
                            if item.name != 'synthetic/history/runtime-attempts.json']
            self.members.append((self.member('synthetic/history/runtime-attempts.json', data), data))
            self.rebind_archive()
            with self.subTest(completed=completed), self.assertRaisesRegex(ValueError, 'rental timeline'): self.invoke()

    def test_closeout_arithmetic_and_public_source_allowlist(self):
        value = deepcopy(self.closeout)
        value['estimate']['gpu_storage_elapsed_usd'] = 0
        self.replace_checked_file('closeout.json', value)
        with self.assertRaisesRegex(ValueError, 'arithmetic'): self.invoke()
        value = deepcopy(self.closeout)
        value['private_source_sha256']['private_account_name'] = 'f' * 64
        self.replace_checked_file('closeout.json', value)
        with self.assertRaisesRegex(ValueError, 'source labels'): self.invoke()

    def test_nonfinite_duplicate_json_and_symlink_rejected(self):
        manifest = self.publication / 'publication.json'
        original = manifest.read_bytes()
        for data in (b'{"schema":NaN}', b'{"schema":"one","schema":"two"}'):
            manifest.write_bytes(data)
            with self.subTest(data=data), self.assertRaises(ValueError): self.invoke()
        manifest.write_bytes(original)
        path = self.publication / 'REPORT.md'
        path.unlink()
        target = self.root / 'report.md'
        target.write_bytes(self.contents['REPORT.md'])
        path.symlink_to(target)
        with self.assertRaisesRegex(ValueError, 'Linked'): self.invoke()


class TrustedTimingImportTests(unittest.TestCase):
    def test_imports_only_local_code_and_restores_globals_on_error(self):
        old_path = list(sys.path)
        old_modules = {key: sys.modules.get(key) for key in ('common', 'score')}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            helper = root / 'source/project/research/vast_cross_model/analysis/timing.py'
            helper.parent.mkdir(parents=True)
            helper.write_bytes((verify.ROOT / 'research/vast_cross_model/analysis/timing.py').read_bytes())
            (root / 'work').mkdir()
            with self.assertRaises(FileNotFoundError):
                verify.trusted_timing(root)
        self.assertEqual(sys.path, old_path)
        self.assertEqual({key: sys.modules.get(key) for key in old_modules}, old_modules)


if __name__ == '__main__':
    unittest.main()
