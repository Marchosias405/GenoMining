"""Describe every focused deliverable and build a portable Section 3 ZIP without unrelated model experiments."""
import argparse
import ast
from pathlib import Path
import zipfile
from readmission.common import ROOT

SHARED = ['README.md', 'readme.me', 'FILE_GUIDE.md', '.gitignore', '.gitattributes', 'config.json', 'requirements.txt', 'requirements-lock.txt',
          'readmission/__init__.py', 'readmission/common.py', 'readmission/data.py', 'readmission/features.py', 'readmission/pipeline.py',
          'tests/test_section3.py', 'docs/FULL_PROJECT_GUIDE.md', 'artifacts/data_audit.json',
          'artifacts/tables/data_dictionary.csv', 'artifacts/tables/split_summary.csv', 'artifacts/status/data.json',
          'data/raw/uci-diabetes.zip', 'data/raw/diabetic_data.csv', 'data/raw/IDS_mapping.csv',
          'data/processed/cohort.csv', 'data/processed/splits.csv']
DESCRIPTIONS = {
    'README.md': 'Main walkthrough: commands, file relationships, measured results and explanations.',
    'readme.me': 'Short overview preserving the requested filename and pointing to the complete README.',
    'FILE_GUIDE.md': 'Branch-level pointer to this focused inventory and the original project guide.',
    '.gitignore': 'Keeps caches, environments and large reproducible artifacts out of Git.',
    '.gitattributes': 'Keeps source text in LF format and PDFs/images binary, preventing checkout corruption and hash drift.',
    'config.json': 'Original seed and settings, preserving the patient split and CPU limits.',
    'requirements.txt': 'Pinned direct packages for analysis, notebook execution and testing.',
    'requirements-lock.txt': 'Original environment record for auditing package versions.',
    'docs/FULL_PROJECT_GUIDE.md': 'Original project instructions for context; later classifiers are outside this ZIP.',
    'section3/config.json': 'Focused choices for MI size, anomaly screening and PCA variance retention.',
    'section3/requirements-report.txt': 'Optional PDF rebuilding dependency, separate from analysis installation.',
    'section3/FILE_GUIDE.md': 'This per-file inventory, generated from the exact package contents.',
    'section3/REQUIREMENTS_COVERAGE.md': 'Maps each screenshot requirement to code, evidence and discussion.',
    'section3/STUDY_GUIDE.md': 'Exam-style questions and worked explanations of methods and leakage risks.',
    'section3/SUBMISSION_CHECKLIST.md': 'Submission contents, verification commands and three-person review suggestions.',
    'section3/section3_eda_preprocessing.ipynb': 'Executed teaching notebook with real outputs and transformation checks.',
    'section3/reports/section3_reference_report.pdf': 'Six-page methods/results reference containing four essential figures.',
    'section3/reports/section3_reference_report.md': 'Editable narrative matching the reference PDF, with relative figure links.',
    'section3/outputs/summary.json': 'Measured counts, dimensions, findings and scope consumed by the notebook/report.',
    'section3/outputs/manifest.json': 'Source fingerprint and output hashes that detect stale or corrupted artifacts.',
    'section3/outputs/verification.json': 'Recorded artifact, notebook, row-alignment and test-evidence checks.',
    'section3/outputs/test_results.xml': 'JUnit evidence from the six focused integrity tests.',
    'section3/outputs/data/development_top20.npz': 'Compressed sparse 59,670 by 147 matrix; rows match the ledger.',
    'section3/outputs/data/development_pca.npz': 'Development PCA scores under key scores; rows match the ledger.',
    'section3/outputs/data/development_row_index.csv': 'Matrix-row ledger linking encounter/patient IDs and labels without using IDs as inputs.',
    'section3/outputs/data/outlier_flags.csv': 'Encounter-level review flags; no observations are deleted.',
    'section3/outputs/transformers/preprocessor_top20.joblib': 'Fitted cleaner, MI selector and encoder for the selected-input representation.',
    'section3/outputs/transformers/preprocessor_all.joblib': 'Fitted all-input preprocessing used before the PCA representation.',
    'section3/outputs/transformers/pca.joblib': 'Fitted centered PCA, applied after the all-input preprocessor.',
    'artifacts/data_audit.json': 'Source hashes, eligibility counts, exclusions and patient-separation evidence.',
    'artifacts/tables/data_dictionary.csv': 'Original column definitions/types/roles, including identifiers and target.',
    'artifacts/tables/split_summary.csv': 'Original partition counts/rates; Section 3 uses development only.',
    'artifacts/status/data.json': 'Completed data-preparation status and fingerprint for reuse.',
    'data/raw/uci-diabetes.zip': 'Original public UCI archive for provenance and offline reproduction.',
    'data/raw/diabetic_data.csv': 'Public encounter source; parsed with ? missing and literal None preserved.',
    'data/raw/IDS_mapping.csv': 'Original code meanings used to verify death/hospice exclusions.',
    'data/processed/cohort.csv': 'Eligible encounters with binary target and patient-disjoint partition assignment.',
    'data/processed/splits.csv': 'Encounter and patient partition ledger for checking overlap.'}
TABLES = {
    'analysis_population': 'Development counts and positive rate; fixes the EDA denominator.',
    'split_sizes': 'Patient/encounter counts for reserved and development partitions.',
    'missingness': 'True missing-value counts and percentages for all original candidate inputs.',
    'numeric_summary': 'All eight numeric distributions: quantiles, range, zeros and skewness.',
    'spearman_correlations': 'Full numeric rank-correlation matrix used in the heatmap.',
    'correlation_pairs': 'Unique pairs ordered by absolute correlation for redundancy checks.',
    'categorical_associations': 'Counts and descriptive outcome rates by age, HbA1c category and discharge code.',
    'laboratory_categories': 'Laboratory category counts explaining why None is retained.',
    'column_decisions': 'Every retained/dropped input and its measured reason.',
    'numeric_transform_audit': 'Learned imputation/scaling parameters and post-transform means/stds.',
    'categorical_encoding_audit': 'Category vocabularies, rare-category pooling and encoding widths.',
    'transformation_examples': 'Three public encounters traced through numeric transformation.',
    'mutual_information': 'All 42 MI ranks/scores and the top-20 selection before one-hot expansion.',
    'encoded_feature_names': 'Matrix column index to encoded feature-name mapping.',
    'outlier_iqr': 'Quartiles, fences, zero-IQR indicators and flagged counts.',
    'outlier_profiles': 'Flagged/unflagged Isolation Forest feature medians with sample sizes.',
    'pca_variance': 'Variance per component and cumulative variance, checking the 95% target.',
    'pca_loadings': 'First two component weights for each encoded input.'}
FIGURES = {
    '01_population_missingness': 'Figure 1: class counts/proportions and true missingness.',
    '02_distributions_boxplots': 'Figure 2: key histograms and outcome-wise boxplots with explicit whisker rules.',
    '03_correlations': 'Figure 3: numeric Spearman heatmap with coefficients.',
    '04_selection_pca': 'Figure 4: raw-input MI ranking and PCA cumulative variance.'}


def files():
    selected = [ROOT / name for name in SHARED]
    selected += [p for p in (ROOT/'section3').rglob('*') if p.is_file()
                 and not {'__pycache__', 'submission', '.ipynb_checkpoints'} & set(p.relative_to(ROOT/'section3').parts)
                 and p.suffix != '.pyc']
    missing = [str(p.relative_to(ROOT)) for p in selected if not p.exists()]
    if missing:
        raise FileNotFoundError('Run analysis and verification before packaging: '+', '.join(missing))
    return sorted(set(selected))


def describe(path):
    name = path.relative_to(ROOT).as_posix()
    if name in DESCRIPTIONS:
        return DESCRIPTIONS[name]
    if path.suffix == '.py':
        return (ast.get_docstring(ast.parse(path.read_text(encoding='utf-8'))) or 'Shared Python package marker.').splitlines()[0]
    if path.parent.name == 'tables' and path.stem in TABLES:
        return TABLES[path.stem]
    if path.parent.name == 'figures' and path.stem in FIGURES:
        return FIGURES[path.stem]
    raise ValueError('Missing file description: '+name)


def inventory():
    target = ROOT/'section3/FILE_GUIDE.md'
    target.touch(exist_ok=True)
    lines = ['# Section 3 file guide', '', 'Every file below is included in the focused ZIP. Large data, matrices and fitted transformers are generated locally and omitted from Git. This package has no classifier outputs. Read the root README for the workflow connecting the files.', '', '| File | Purpose and why it is included |', '|---|---|']
    for p in files():
        lines.append(f'| `{p.relative_to(ROOT).as_posix()}` | {describe(p)} |')
    target.write_text('\n'.join(lines)+'\n', encoding='utf-8')


def package(destination):
    from .verify import verify
    verify()
    inventory()
    destination = destination.resolve()
    members = files()
    if destination in members:
        raise ValueError('Write the ZIP outside the source tree or in section3/submission/.')
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for p in members:
            archive.write(p, 'GenoMining/'+p.relative_to(ROOT).as_posix())
    with zipfile.ZipFile(destination) as archive:
        assert archive.testzip() is None
    print(f'Created {destination} ({destination.stat().st_size/1024**2:.1f} MiB; {len(members)} files)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--zip', type=Path)
    args = parser.parse_args()
    if args.zip:
        package(args.zip)
    else:
        inventory()
