"""Host-side one-command measurement. No provisioning, credentials or model downloads."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from measurement import load_bundle

HERE = Path(__file__).resolve().parent


def run_pair(bundle, model_dir, adapter_dir, out, deadline_seconds, execute=subprocess.run):
    load_bundle(bundle)
    out = Path(out)
    if not out.is_absolute() or out.is_symlink() or out.exists():
        raise ValueError('Output must be an absolute new directory')
    if type(deadline_seconds) is not int or not 1 <= deadline_seconds <= 86400:
        raise ValueError('Per-arm deadline must be 1..86400 seconds')
    out.mkdir(parents=False, exist_ok=False)
    for arm in ('base', 'adapter'):
        execute([sys.executable, str(HERE / 'worker.py'), '--bundle', str(bundle),
                 '--model-dir', str(model_dir), '--adapter-dir', str(adapter_dir), '--out', str(out / arm),
                 '--arm', arm, '--deadline-seconds', str(deadline_seconds)], check=True, timeout=deadline_seconds + 30)
    execute([sys.executable, str(HERE / 'measurement.py'), '--bundle', str(bundle),
             '--base-run', str(out / 'base'), '--adapter-run', str(out / 'adapter'),
             '--out', str(out / 'scored')], check=True, timeout=120)
    report = out / 'scored/report.json'
    if not report.is_file():
        raise ValueError('Scorer did not produce a report')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('bundle', 'model-dir', 'adapter-dir', 'out'):
        parser.add_argument('--' + flag, type=Path, required=True)
    parser.add_argument('--deadline-seconds', type=int, default=1800)
    args = parser.parse_args()
    report = run_pair(args.bundle, args.model_dir, args.adapter_dir, args.out, args.deadline_seconds)
    print(json.dumps(dict(report_path=str(report), meaning='Measurement completed; read report for improvement verdict')))


if __name__ == '__main__':
    main()
