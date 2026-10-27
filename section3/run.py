"""Run or reuse the Section 3 analysis, recording source fingerprints and hashes for every generated output."""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from readmission.common import ROOT, sha256, environment
from readmission.pipeline import run_phase
from .analysis import OUT, run, write_json

SOURCES = ['config.json', 'section3/config.json', 'section3/analysis.py', 'section3/run.py',
           'readmission/common.py', 'readmission/data.py', 'readmission/features.py', 'readmission/pipeline.py']


def fingerprint():
    files = SOURCES + ['data/raw/diabetic_data.csv', 'data/processed/cohort.csv']
    text = ''.join(sha256(ROOT / p) for p in files) + json.dumps(environment(), sort_keys=True)
    return hashlib.sha256(text.encode()).hexdigest()


def current():
    marker = OUT / 'manifest.json'
    if not marker.exists() or not (ROOT / 'data/processed/cohort.csv').exists():
        return False
    value = json.loads(marker.read_text(encoding='utf-8'))
    try:
        return value['fingerprint'] == fingerprint() and all(
            (OUT / name).is_file() and sha256(OUT / name) == digest for name, digest in value['outputs'].items())
    except FileNotFoundError:
        return False


def run_section(force=False):
    run_phase('data')
    if not force and current():
        print('Section 3: reusing outputs with matching sources, data, environment and file hashes.', flush=True)
        return json.loads((OUT / 'summary.json').read_text(encoding='utf-8'))
    marker = OUT / 'manifest.json'
    if marker.exists():
        marker.unlink()
    result = run()
    names = sorted(p for p in OUT.rglob('*') if p.is_file() and p.name not in {'manifest.json', 'verification.json', 'test_results.xml'})
    write_json('manifest.json', {'completed_at_utc': datetime.now(timezone.utc).isoformat(), 'fingerprint': fingerprint(),
                               'outputs': {p.relative_to(OUT).as_posix(): sha256(p) for p in names}})
    print(f"Section 3 complete: {result['development_encounters']:,} development encounters; "
          f"{result['retained_raw_inputs']} retained inputs; {result['pca_components']} PCA components.", flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--force', action='store_true', help='Recompute Section 3 only; never trains classifiers.')
    run_section(parser.parse_args().force)
