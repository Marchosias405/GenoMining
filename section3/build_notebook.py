"""Generate the focused Section 3 teaching notebook and optionally execute it with retained outputs."""
import argparse
import asyncio
import json
import os
from pathlib import Path
import sys
import nbformat as nbf
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / 'section3/section3_eda_preprocessing.ipynb'


def build(execute=False):
    notebook = nbf.v4.new_notebook()
    md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
    cells = [md('''# Section 3 Data preprocessing and exploratory analysis

This notebook examines diabetes hospital encounters and makes the preprocessing choices visible. It covers the supplied Section 3 requirements only. EDA and fitted transforms use **development patients only**. Run from a full repository checkout or the focused ZIP, after installing `requirements.txt` (see the root README).

The sequence is: validate data -> explore -> screen unusual records -> clean and encode -> select raw features -> compare a PCA representation. Four essential figures summarize the evidence. No classifier is trained. The PDF and Markdown reference report explain the measured findings.'''),
        code('''from pathlib import Path
import os, sys, json
for name in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ.setdefault(name, '2')
candidates = [Path.cwd(), *Path.cwd().parents, Path('/content/GenoMining'), Path('/kaggle/working/GenoMining')]
ROOT = next((p for p in candidates if (p/'section3/config.json').exists() and (p/'readmission').is_dir()), None)
if ROOT is None:
    raise FileNotFoundError('Set the working directory to the extracted project. See README.md.')
os.chdir(ROOT)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import joblib
from scipy import sparse
from IPython.display import display, Image
from section3.run import run_section
from readmission.data import load_cohort, xy
OUT = ROOT/'section3/outputs'
def table(name):
    return pd.read_csv(OUT/'tables'/f'{name}.csv', keep_default_na=False)
def figure(name):
    display(Image(filename=str(OUT/'figures'/f'{name}.png'), width=1000))
pd.set_option('display.max_columns', 16)
print('Project:', ROOT.name)'''),
        md('## 1 Data structure and analysis population\n\nThe original 50 columns consist of 47 candidate inputs, two identifiers and the outcome. Death/hospice encounter exclusions define the eligible cohort. Patients remain in a single partition. EDA uses only the 59,670 development encounters; the other partitions are reserved for later modeling.'),
        code("summary = run_section()  # Set force=True only to recompute this section.\ndisplay(table('split_sizes'))\ndisplay(table('analysis_population'))\nprint('Raw inputs:', summary['data']['candidate_features'], '| Numeric:', summary['numeric_inputs'], '| Categorical:', summary['categorical_candidate_inputs'])"),
        code("cohort = load_cohort()\nassert cohort.groupby('patient_nbr')['split'].nunique().max() == 1\ndev = cohort.loc[cohort.split.eq('development')].reset_index(drop=True)\nX, y = xy(dev)\nprint('Patient overlap: 0; development inputs:', X.shape)\ndel cohort"),
        md('## 2 Missingness and class imbalance\n\n`?` is missing. The literal laboratory category `None` means the test was not performed and remains a valid category. The positive label is the existing `<30` outcome, not a reconstructed exact follow-up interval. The proportions below are percentages of encounters, not unique people.'),
        code("figure('01_population_missingness')\ndisplay(table('missingness').head(7))\ndisplay(table('laboratory_categories'))"),
        md('## 3 Distributions and associations\n\nHistograms include every development observation. A log count axis reveals the rare tail without changing the observations. Boxplot whiskers show the 5th and 95th percentiles, and points outside them are hidden only in the figure. They are not deleted. Spearman correlation measures ranked monotonic association; it does not show causation.'),
        code("figure('02_distributions_boxplots')\ndisplay(table('numeric_summary'))"),
        code("figure('03_correlations')\ndisplay(table('correlation_pairs').head(5))\ndisplay(table('categorical_associations').query(\"feature == 'A1Cresult'\"))"),
        md('## 4 Outlier handling\n\nIQR fences screen each numerical variable. Zero-heavy visit counts can have IQR zero, so every positive value can be flagged. Isolation Forest supplies a complementary multivariable screen on log1p numerical inputs. Its 3% flag setting is a review budget, not an estimate of erroneous records. All eligible encounters are retained because unusual utilization is plausible and no error labels justify deletion.'),
        code("display(table('outlier_iqr'))\ndisplay(table('outlier_profiles'))\nprint('IQR any flag:', summary['iqr_any_flagged'])\nprint('Isolation Forest flags:', summary['isolation_flagged'])\nprint('Rows removed as outliers:', summary['outlier_rows_removed'])"),
        md('## 5 Cleaning and encoding\n\nThe saved preprocessing Pipeline contains three steps: ClinicalCleaner, RawMutualInformationSelector and ColumnTransformer. The cleaner groups diagnoses and learns exclusions from development data. The encoder median-imputes and standardizes numeric inputs; it fills missing categories with Missing and one-hot encodes them. Integer admission/discharge codes remain categorical. Inspect the learned values rather than assuming imputation changed every column.'),
        code("full = joblib.load(OUT/'transformers/preprocessor_all.joblib')\nselected = joblib.load(OUT/'transformers/preprocessor_top20.joblib')\nprint('Pipeline steps:', list(full.named_steps))\ndisplay(table('column_decisions').query('not retained'))\ndisplay(table('numeric_transform_audit'))\ndisplay(table('categorical_encoding_audit'))"),
        code("display(table('transformation_examples').head(8))\nprint('All inputs after encoding:', summary['all_encoded'])\nprint('Selected inputs after encoding:', summary['selected_encoded'])"),
        md('## 6 Independent feature selection and PCA\n\nMutual information uses development labels to rank whole raw features before one-hot expansion. It is independent of classifier fitting. The top 20 are a predeclared comparison size, not proven optimal. PCA uses no labels and creates components from all cleaned encoded inputs. It centers the representation and retains at least 95% of its variance; variance retained is not predictive accuracy. PCA and MI are separate alternatives here.'),
        code("figure('04_selection_pca')\ndisplay(table('mutual_information'))\ndisplay(table('pca_variance').tail())\nprint('PCA dimensions:', summary['all_encoded']['columns'], '->', summary['pca_components'])\nprint('Retained variance:', summary['pca_retained_variance'])"),
        md('## 7 Reuse the prepared inputs safely\n\nThe sparse matrix, labels and row ledger have exactly the same development row order. They support inspection or a single development-trained fit. For later cross-validation, fit a fresh preprocessing Pipeline inside each training fold; using this already-selected full-development matrix in CV would leak validation-label information.'),
        code("matrix = sparse.load_npz(OUT/'data/development_top20.npz')\nledger = pd.read_csv(OUT/'data/development_row_index.csv')\nassert matrix.shape[0] == len(ledger) == len(dev)\nnp.testing.assert_array_equal(ledger.encounter_id, dev.encounter_id)\nnp.testing.assert_allclose(selected.transform(X.iloc[:3]).toarray(), matrix[:3].toarray(), rtol=1e-6, atol=1e-6)\nprint('Prepared matrix:', matrix.shape, '| row order and transform reproduction passed')"),
        code("# Synthetic robustness check only: this category is not a real observation.\nprobe = X.iloc[:1].copy()\nprobe['age'] = 'unseen age category for test'\nbefore = full.named_steps['encode'].named_transformers_['numeric'].named_steps['scale'].mean_.copy()\nencoded_probe = full.transform(probe)\nnp.testing.assert_array_equal(before, full.named_steps['encode'].named_transformers_['numeric'].named_steps['scale'].mean_)\nassert encoded_probe.shape[1] == summary['all_encoded']['columns']\nprint('Unknown-category transform succeeded without refitting the scaler.')"),
        md('## 8 Explain the decisions\n\nNo rows were synthesized or oversampled. The original prevalence is retained; data augmentation is discussed but is not necessary for this section. Answer these before writing your final report: Why does patient separation matter? Why is None not missing? Why does standardization not remove skew? Why can IQR overflag rare visits? What does MI need that PCA does not? Why must selection be repeated within CV? The README and study guide provide worked explanations.\n\nRead `section3/SUBMISSION_CHECKLIST.md` for the exact evidence to include. The reference report is a study aid; the team writes and verifies its final report.')]
    notebook.cells = cells
    notebook.metadata['kernelspec'] = {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'}
    notebook.metadata['language_info'] = {'name': 'python', 'version': '3.12'}
    if execute:
        if sys.platform == 'win32':
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        kernel_root = ROOT/'.cache/section3-jupyter'
        kernel = kernel_root/'kernels/section3'
        kernel.mkdir(parents=True, exist_ok=True)
        (kernel/'kernel.json').write_text(json.dumps({'argv': [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}'], 'display_name': 'Section 3', 'language': 'python'}), encoding='utf-8')
        os.environ['JUPYTER_PATH'] = str(kernel_root)
        for variable, directory in [('JUPYTER_RUNTIME_DIR', 'jupyter-runtime'), ('IPYTHONDIR', 'ipython'), ('MPLCONFIGDIR', 'matplotlib'), ('TEMP', 'tmp'), ('TMP', 'tmp')]:
            target = ROOT/'.cache'/directory
            target.mkdir(parents=True, exist_ok=True)
            os.environ[variable] = str(target)
        NotebookClient(notebook, timeout=1200, kernel_name='section3', resources={'metadata': {'path': str(ROOT)}}, allow_errors=False).execute()
    nbf.write(notebook, TARGET)
    print('Saved', TARGET.name, 'with executed outputs' if execute else 'without outputs')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    build(parser.parse_args().execute)
