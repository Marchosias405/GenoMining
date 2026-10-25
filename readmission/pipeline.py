"""Run resumable project phases and track source/configuration provenance for each completed phase."""
from datetime import datetime, timezone
from time import perf_counter
import hashlib
import importlib
from .common import ROOT, config, path, sha256, write_json, read_json, environment

PHASES = ['data','eda','unsupervised','models','evaluation']
FILES = {'data':['data','common'], 'eda':['eda','features'], 'unsupervised':['unsupervised'],
         'models':['models','features'], 'evaluation':['evaluation']}
REQUIRED = {'data':['data/processed/cohort.csv','data/processed/splits.csv','artifacts/data_audit.json'],
    'eda':['artifacts/tables/missingness.csv','artifacts/preprocessing_audit.json','artifacts/figures/categorical_outcomes.png'],
    'unsupervised':['artifacts/tables/clustering_metrics.csv','artifacts/tables/outlier_metrics.csv'],
    'models':['artifacts/models/index.json','artifacts/tables/model_selection.csv'],
    'evaluation':['artifacts/tables/test_metrics.csv','artifacts/models/calibrated_index.json','artifacts/predictions/test_predictions.csv']}


def fingerprint(phase):
    names = set(['common','data','pipeline'])
    if phase in ['eda','models','evaluation']:
        names.add('features')
    if phase=='evaluation':
        names.add('models')
    names.update(FILES[phase])
    text = sha256(ROOT/'config.json') + ''.join(sha256(ROOT/'readmission'/f'{n}.py') for n in sorted(names))
    if (ROOT/'data/raw/diabetic_data.csv').exists():
        text += sha256(ROOT/'data/raw/diabetic_data.csv')
    if phase != 'data' and (ROOT/'data/processed/cohort.csv').exists():
        text += sha256(ROOT/'data/processed/cohort.csv')
    if phase == 'evaluation' and (ROOT/'artifacts/models/index.json').exists():
        text += sha256(ROOT/'artifacts/models/index.json')
        for item in read_json('artifacts/models/index.json'):
            if (ROOT/item['file']).exists():
                text += sha256(ROOT/item['file'])
    return hashlib.sha256(text.encode()).hexdigest()


def run_phase(phase, force=False):
    if phase not in PHASES:
        raise ValueError(f'Unknown phase: {phase}')
    if phase != 'data':
        run_phase('data')
    if phase == 'evaluation':
        run_phase('models')
    marker = path(f'artifacts/status/{phase}.json')
    complete = all((ROOT/p).exists() for p in REQUIRED[phase])
    if complete and phase in ['models','evaluation']:
        index_name = 'index.json' if phase=='models' else 'calibrated_index.json'
        complete = all((ROOT/item['file']).exists() for item in read_json('artifacts/models/'+index_name))
    if complete and marker.exists() and not force and read_json(f'artifacts/status/{phase}.json')['fingerprint']==fingerprint(phase):
        print(f'Phase {phase}: reusing verified matching artifacts.',flush=True)
        return
    if marker.exists():
        marker.unlink()
    print(f'Phase {phase}: starting.',flush=True)
    start = perf_counter()
    importlib.import_module('readmission.'+phase).run()
    write_json(f'artifacts/status/{phase}.json',{'phase':phase,'status':'complete',
        'completed_at_utc':datetime.now(timezone.utc).isoformat(),'seconds':perf_counter()-start,
        'fingerprint':fingerprint(phase),'configuration':config(),'environment':environment()})
    print(f'Phase {phase}: complete in {perf_counter()-start:.1f}s.',flush=True)


def run_all(force=False):
    for phase in PHASES:
        run_phase(phase,force=force)
