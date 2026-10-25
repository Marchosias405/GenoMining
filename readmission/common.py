"""Paths, configuration, and small output helpers shared across all experiment phases."""
from pathlib import Path
import hashlib
import json
import platform
import importlib.metadata

ROOT = Path(__file__).resolve().parents[1]
NUMERIC = ['time_in_hospital', 'num_lab_procedures', 'num_procedures',
           'num_medications', 'number_outpatient', 'number_emergency',
           'number_inpatient', 'number_diagnoses']
IDS = ['encounter_id', 'patient_nbr']
TARGET = 'readmitted'
EXCLUDED_DISCHARGES = {11, 13, 14, 19, 20, 21}
DATA_URL = 'https://archive.ics.uci.edu/static/public/296/diabetes+130-us+hospitals+for+years+1999-2008.zip'
DATA_PAGE = 'https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008'


def config():
    return json.loads((ROOT / 'config.json').read_text(encoding='utf-8'))


def path(relative):
    result = ROOT / relative
    result.parent.mkdir(parents=True, exist_ok=True)
    return result


def write_json(relative, value):
    path(relative).write_text(json.dumps(value, indent=2, default=str, allow_nan=False), encoding='utf-8')


def read_json(relative):
    return json.loads(path(relative).read_text(encoding='utf-8'))


def sha256(file):
    digest = hashlib.sha256()
    with open(file, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def environment():
    packages = ['numpy', 'pandas', 'scipy', 'scikit-learn', 'xgboost', 'matplotlib', 'joblib']
    return {'python': platform.python_version(), 'platform': platform.platform(),
            'packages': {name: importlib.metadata.version(name) for name in packages}}


def save_figure(fig, name):
    import matplotlib.pyplot as plt
    fig.savefig(path('artifacts/figures/' + name + '.png'), dpi=160, bbox_inches='tight')
    plt.close(fig)
