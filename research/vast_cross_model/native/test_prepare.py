"""Bounded preparation checks; no cloud provisioning, installation or compilation."""
import io
import importlib.util
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location('native_prepare', Path(__file__).with_name('prepare.py'))
prepare = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prepare)


class PreparationTests(unittest.TestCase):
    def make_archive(self, root, member):
        archive = root / 'component.tar.xz'
        with tarfile.open(archive, 'w:xz') as tar:
            member.size = 4 if member.isfile() else 0
            tar.addfile(member, io.BytesIO(b'test') if member.isfile() else None)
        return archive

    def test_official_requirement_digest(self):
        required = prepare.requirements()
        self.assertEqual(required['rust_version'], '1.97.1')
        self.assertEqual(set(required['components']), {'rustc', 'cargo', 'rust-std'})

    def test_regular_component_extraction(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            member = tarfile.TarInfo('component/bin/test')
            member.mode = 0o755
            archive = self.make_archive(root, member)
            extracted = prepare.extract_component(archive, root / 'out', 'component')
            self.assertEqual((extracted / 'bin/test').read_bytes(), b'test')
            self.assertEqual((extracted / 'bin/test').stat().st_mode & 0o777, 0o755)

    def test_parent_traversal_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive = self.make_archive(root, tarfile.TarInfo('component/../escape'))
            with self.assertRaises(ValueError):
                prepare.extract_component(archive, root / 'out', 'component')
            self.assertFalse((root / 'escape').exists())

    def test_link_archive_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            member = tarfile.TarInfo('component/link')
            member.type = tarfile.SYMTYPE
            member.linkname = '/tmp'
            archive = self.make_archive(root, member)
            with self.assertRaises(ValueError):
                prepare.extract_component(archive, root / 'out', 'component')

    def test_mismatched_download_rejected_before_network(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'download'
            path.write_bytes(b'wrong')
            with patch.object(prepare.urllib.request, 'urlopen') as opener:
                with self.assertRaises(ValueError):
                    prepare.download_verified('https://static.rust-lang.org/dist/test', '0' * 64, path, 10)
                opener.assert_not_called()

    def test_nonlinux_refused(self):
        with patch.object(prepare.platform, 'system', return_value='Darwin'):
            with self.assertRaises(ValueError):
                prepare.linux_worker()

    def test_child_environment_excludes_provider_credentials(self):
        with patch.dict(prepare.os.environ, {'CONTAINER_API_KEY': 'test-only', 'HF_TOKEN': 'test-only',
                                            'AWS_SECRET_ACCESS_KEY': 'test-only', 'PATH': '/bin',
                                            'LD_LIBRARY_PATH': '/public/lib'}, clear=True):
            self.assertEqual(prepare.clean_environment(), {'PATH': '/bin', 'LD_LIBRARY_PATH': '/public/lib'})

    def test_exact_focused_test_counts(self):
        self.assertEqual(prepare.focused_test_counts('test result: ok. 17 passed; 0 failed; 0 ignored; 0 measured;\n'),
                         {'passed': 17, 'failed': 0, 'ignored': 0})
        for log in ['test result: ok. 0 passed; 0 failed; 0 ignored;',
                    'test result: ok. 16 passed; 0 failed; 1 ignored;',
                    'test result: FAILED. 17 passed; 1 failed; 0 ignored;', '']:
            with self.assertRaises(ValueError):
                prepare.focused_test_counts(log)

    def test_source_mutation_and_unlisted_file_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / 'Cargo.toml'
            source.write_bytes(b'test')
            source.chmod(0o644)
            manifest = {'source_commit': prepare.COMMIT, 'source_tree': prepare.TREE,
                        'base_commit': prepare.BASE, 'files': {'Cargo.toml': {
                            'sha256': prepare.sha(source), 'bytes': 4, 'mode': 0o644}}}
            (root / prepare.MANIFEST).write_text(json.dumps(manifest))
            self.assertEqual(prepare.verify_source(root)['verified_files'], 1)
            source.write_bytes(b'fake')
            with self.assertRaises(ValueError): prepare.verify_source(root)
            source.write_bytes(b'test')
            (root / 'unexpected').write_text('extra')
            with self.assertRaises(ValueError): prepare.verify_source(root)


if __name__ == '__main__':
    unittest.main()
