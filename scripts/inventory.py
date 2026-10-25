"""Document every delivered file's purpose and package portable source, evidence, data and models without local environments."""
from pathlib import Path
import argparse
import ast
import os
import zipfile

ROOT = Path(__file__).resolve().parents[1]
EXCLUDE = {'.venv','.cache','.git','__pycache__','.pytest_cache','.ipynb_checkpoints'}
DESCRIPTIONS = {
    'README.md':'Entry point: introduction, example use, setup, phases and folder structure.',
    'readme.me':'Requested-filename overview with a short introduction, usage example and folder structure.',
    'FILE_GUIDE.md':'This inventory explains the purpose of each portable project file.',
    '.gitignore':'Excludes environments, caches and large reproducible files from Git.',
    'requirements.txt':'Pinned direct dependencies needed to reproduce the experiment.',
    'requirements-lock.txt':'Exact installed dependency versions for environment auditing.',
    'config.json':'Fixed seeds, CV/search sizes, thresholds and sample sizes for reproducibility.',
    'artifacts/integrity_tests.xml':'JUnit evidence from the 12 integrity tests covering patient isolation, target semantics and frozen calibration.',
    'docs/EXPERIMENT_DESIGN.md':'Technical choices, scope limits and source references supporting reproducibility.',
    'docs/TEAM_WORK.md':'Suggested review ownership and evidence checklist for your own report.',
    'data/raw/uci-diabetes.zip':'Original public UCI archive, retained for checksum/provenance and offline reproduction.',
    'data/raw/diabetic_data.csv':'Original encounter records; the immutable source for cohort construction.',
    'data/raw/IDS_mapping.csv':'Source meanings of admission/discharge/source codes; verifies cohort exclusions.',
    'data/processed/cohort.csv':'Eligible encounters with target and patient-disjoint split assignments.',
    'data/processed/splits.csv':'Explicit encounter-to-patient partition ledger for leakage auditing.',
    'examples/encounters.csv':'Three public feature-only rows demonstrating the CSV prediction interface.'}


def files():
    included = []
    for directory,subdirs,names in os.walk(ROOT):
        subdirs[:] = [name for name in subdirs if name not in EXCLUDE]
        included.extend(Path(directory)/name for name in names)
    return sorted(included)


def describe(p):
    relative = p.relative_to(ROOT).as_posix()
    if relative in DESCRIPTIONS:
        return DESCRIPTIONS[relative]
    if p.suffix=='.py':
        return (ast.get_docstring(ast.parse(p.read_text(encoding='utf-8'))) or 'Python implementation helper.').splitlines()[0]
    if p.suffix=='.ipynb':
        return 'Executed phase notebook with visible code, tables/figures and interpretation prompts; supports course submission.'
    if p.suffix=='.png':
        return f'Generated {p.stem.replace("_"," ")} figure; visual evidence for inspection and your report.'
    if p.suffix=='.joblib':
        return 'Serialized fitted preprocessing/classifier (and calibration when named); reproduces saved predictions without retraining.'
    if 'status' in p.parts:
        return 'Completed phase status, source fingerprint, configuration, runtime and package versions; prevents stale reuse.'
    if p.suffix=='.csv':
        return f'Generated {p.stem.replace("_"," ")} evidence table; supports reproducible comparisons or prediction auditing.'
    if p.suffix=='.json':
        return f'Machine-readable {p.stem.replace("_"," ")} metadata; records design, provenance or verification.'
    return 'Supporting project file; retained with the reproducible deliverable.'


def inventory():
    target = ROOT/'FILE_GUIDE.md'
    target.touch(exist_ok=True)
    lines = ['# File guide','', 'Every portable file is listed below. Local environments, caches and Git internals are excluded because they are machine-specific. Source modules also have purpose docstrings. Generated outputs are evidence, not a written academic report.','', '| File | Purpose and why it is needed |','|---|---|']
    for p in files():
        lines.append(f'| `{p.relative_to(ROOT).as_posix()}` | {describe(p).replace("|","/")} |')
    target.write_text('\n'.join(lines)+'\n',encoding='utf-8')


def package(destination):
    destination.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(destination,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for p in files():
            archive.write(p,Path('hospital-readmission')/p.relative_to(ROOT))
    print(f'Created {destination} ({destination.stat().st_size/1024/1024:.1f} MiB)')


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--zip',type=Path)
    args = parser.parse_args()
    inventory()
    if args.zip:
        package(args.zip.resolve())
