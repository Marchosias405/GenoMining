"""Preserve the authored notebooks 01/02; generate later phase notebooks and optionally execute them."""
import argparse
import asyncio
import json
import os
from pathlib import Path
import sys
import nbformat as nbf
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
BOOTSTRAP = '''from pathlib import Path
import os, sys
for name in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ.setdefault(name, '2')
candidates = [Path.cwd(), *Path.cwd().parents,
              Path('/content/hospital-readmission'), Path('/kaggle/working/hospital-readmission')]
ROOT = next((p for p in candidates if (p / 'config.json').exists() and (p / 'readmission').is_dir()), None)
if ROOT is None:
    raise FileNotFoundError('Extract the full project and set the working directory to hospital-readmission first. See README.md.')
os.chdir(ROOT)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import pandas as pd
from IPython.display import display, Image, JSON
from readmission.common import read_json
from readmission.pipeline import run_phase
pd.set_option('display.max_columns', 30)
print('Project:', ROOT.name)
'''


def table(name, code=''):
    return f"display(pd.read_csv(ROOT / 'artifacts/tables/{name}'){code})"


def figure(name):
    return f"display(Image(filename=str(ROOT / 'artifacts/figures/{name}.png'), width=1000))"


def specs():
    return [
        ('01_data_preparation','data','Data preparation and patient separation',
         'Authored directly in notebook cells; preserve its content.', [], ''),
        ('02_exploratory_analysis','eda','Exploratory analysis and preprocessing',
         'Authored directly in notebook cells; preserve its content.', [], ''),
        ('03_clustering_outliers','unsupervised','Clustering and outlier comparisons',
         'Compare three clustering methods and three outlier detectors; inspect assumptions and keep valid unusual encounters.',
         [table('clustering_metrics.csv'),figure('clusters_pca'),table('cluster_profiles.csv'),
          table('outlier_metrics.csv'),figure('outliers_pca'),table('outlier_profiles.csv'),table('outlier_overlap.csv'),
          "display(JSON(read_json('artifacts/unsupervised_design.json')))"],
         'Compare numerical utilization clusters, silhouette versus other indices, detector agreement, and the evidence for retaining flagged encounters.'),
        ('04_model_training','models','Classifiers, feature selection and tuning',
         'Compare classifiers on shared development folds, record independent feature selection, and choose a model without test outcomes.',
         [table('cv_split_audit.csv'),table('model_selection.csv'),table('cv_fold_scores.csv'),
          "features = pd.read_csv(ROOT / 'artifacts/tables/feature_selection.csv')\ndisplay(features.query(\"experiment == 'logistic_regression__mi20'\").sort_values('mutual_information', ascending=False))",
          "display(JSON(read_json('artifacts/model_choice.json')))",
          "for filename in sorted((ROOT / 'artifacts/tables').glob('search_*.csv')):\n    print(filename.name)\n    display(pd.read_csv(filename)[['params', 'mean_test_score', 'std_test_score', 'mean_fit_time', 'rank_test_score']])"],
         'Explain the tuning search, baseline retention rule, selected raw/encoded features, and why held-out test results must not choose the model.'),
        ('05_evaluation','evaluation','Held-out performance and calibration',
         'Evaluate frozen classifiers, independently fit calibration, examine patient-bootstrap uncertainty and score example encounters.',
         [table('test_metrics.csv'),figure('roc_precision_recall'),figure('calibration_comparison'),figure('confusion_matrices'),figure('feature_selection_comparison'),
          table('patient_bootstrap_intervals.csv'),table('subgroup_metrics.csv'),
          "display(JSON(read_json('artifacts/evaluation_design.json')))",
          "import subprocess\nsubprocess.run([sys.executable, 'run.py', 'predict', '--input', 'examples/encounters.csv'], check=True)\ndisplay(pd.read_csv(ROOT / 'artifacts/predictions/example_predictions.csv'))"],
         'Interpret the low-recall risk of threshold 0.5, whether calibration helped, uncertainty, subgroup limitations, and lack of external validation.')]


def build(execute=False, through=5):
    notebook_dir = ROOT/'notebooks'
    notebook_dir.mkdir(exist_ok=True)
    if execute:
        if sys.platform == 'win32':
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        kernel_root = ROOT/'.cache/jupyter'
        kernel = kernel_root/'kernels/readmission-project'
        kernel.mkdir(parents=True,exist_ok=True)
        (kernel/'kernel.json').write_text(json.dumps({'argv':[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'],
            'display_name':'Readmission project','language':'python'}),encoding='utf-8')
        os.environ['JUPYTER_PATH'] = str(kernel_root)
        for variable, directory in [('JUPYTER_RUNTIME_DIR','jupyter-runtime'),('IPYTHONDIR','ipython'),('MPLCONFIGDIR','matplotlib'),('TEMP','tmp'),('TMP','tmp')]:
            target = ROOT/'.cache'/directory
            target.mkdir(parents=True,exist_ok=True)
            os.environ[variable] = str(target)
    for filename,phase,title,purpose,outputs,prompts in specs()[:through]:
        target = notebook_dir/(filename+'.ipynb')
        if filename in {'01_data_preparation', '02_exploratory_analysis'}:
            if not target.exists():
                raise FileNotFoundError(f'{target.name} is an authored notebook; restore it from Git.')
            if execute:
                print(f'Executing {target.name}', flush=True)
                notebook = nbf.read(target, as_version=4)
                NotebookClient(notebook, timeout=7200, kernel_name='readmission-project',
                               resources={'metadata': {'path': str(ROOT)}}, allow_errors=False).execute()
                nbf.write(notebook, target)
            else:
                print(f'Preserved {target.name}; edit its cells directly.', flush=True)
            continue
        notebook = nbf.v4.new_notebook()
        notebook.metadata['kernelspec'] = {'name':'python3','display_name':'Python 3','language':'python'}
        notebook.metadata['language_info'] = {'name':'python','version':'3.12'}
        notebook.cells = [nbf.v4.new_markdown_cell(f'# {title}\n\n**Purpose:** {purpose}\n\nRequires the complete project folder and dependencies from `requirements.txt`; see README.md for local/Colab/Kaggle setup. Matching saved artifacts are reused. Use `force=True` in the phase cell to recompute. Outputs must remain visible when saving.'),
                          nbf.v4.new_code_cell(BOOTSTRAP),nbf.v4.new_code_cell(f"run_phase('{phase}')")]
        notebook.cells.extend(nbf.v4.new_code_cell(code) for code in outputs)
        notebook.cells.append(nbf.v4.new_markdown_cell('## Team interpretation prompts\n\n'+prompts+'\n\nWrite your own report discussion after checking the evidence.'))
        target = notebook_dir/(filename+'.ipynb')
        if execute:
            print(f'Executing {target.name}',flush=True)
            NotebookClient(notebook,timeout=7200,kernel_name='readmission-project',resources={'metadata':{'path':str(ROOT)}},allow_errors=False).execute()
        nbf.write(notebook,target)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--through',type=int,choices=range(1,6),default=5,help='Build/execute only the first N phase notebooks.')
    args = parser.parse_args()
    build(args.execute,args.through)
