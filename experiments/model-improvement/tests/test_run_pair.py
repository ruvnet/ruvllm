import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
spec = importlib.util.spec_from_file_location('run_pair', HERE / 'run-pair.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class HostCommands(unittest.TestCase):
    def test_failed_base_arm_cannot_run_adapter_or_produce_success(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(module, 'load_bundle'):
            calls = []
            def reject(argv, **kwargs):
                calls.append(argv)
                raise subprocess.CalledProcessError(1, argv)
            with self.assertRaises(subprocess.CalledProcessError):
                module.run_pair('/bundle', '/model', '/adapter', Path(tmp) / 'run', 10, execute=reject)
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0][calls[0].index('--arm') + 1], 'base')
            self.assertFalse((Path(tmp) / 'run/scored/report.json').exists())

    def test_fixed_argv_and_complete_report_required(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(module, 'load_bundle'):
            out = Path(tmp) / 'run'
            calls = []
            def execute(argv, **kwargs):
                self.assertTrue(kwargs['check'])
                self.assertNotIn('shell', kwargs)
                calls.append(argv)
                if argv[1].endswith('measurement.py'):
                    (out / 'scored').mkdir()
                    (out / 'scored/report.json').write_text('{"fixture":"not a measurement"}')
            report = module.run_pair('/bundle', '/model', '/adapter', out, 10, execute=execute)
            self.assertEqual(len(calls), 3)
            self.assertEqual(report, out / 'scored/report.json')
            with self.assertRaises(ValueError):
                module.run_pair('/bundle', '/model', '/adapter', out, 10, execute=execute)


if __name__ == '__main__':
    unittest.main()
