"""Check phase evidence, patient isolation, held-out metrics, serialized-model reproducibility and executed notebook outputs."""
from pathlib import Path
import json
import sys
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import joblib
import nbformat

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from readmission.common import read_json,write_json
from readmission.data import load_cohort,xy
from readmission.evaluation import metrics
from readmission.pipeline import PHASES,fingerprint


def verify():
    checks = {}
    suites = ET.parse(ROOT/'artifacts/integrity_tests.xml').getroot().findall('testsuite')
    assert sum(int(s.attrib.get('tests',0)) for s in suites)==12
    assert all(int(s.attrib.get('failures',0))==0 and int(s.attrib.get('errors',0))==0 for s in suites)
    checks['integrity_tests_passed'] = 12
    for phase in PHASES:
        status = read_json(f'artifacts/status/{phase}.json')
        assert status['status']=='complete'
        assert status['fingerprint']==fingerprint(phase), f'Stale {phase} artifacts'
    checks['all_five_phases_current'] = True
    data = load_cohort()
    assert len(data)==99343 and data.target.sum()==11314
    assert data.groupby('patient_nbr')['split'].nunique().eq(1).all()
    checks['patient_overlap_zero'] = True
    cv = pd.read_csv(ROOT/'artifacts/tables/cv_split_audit.csv')
    assert len(cv)==5 and cv.patient_overlap.eq(0).all()
    selection = pd.read_csv(ROOT/'artifacts/tables/model_selection.csv')
    assert len(selection)==6
    assert all(selection.selected_cv_ap >= selection.baseline_cv_ap-1e-12)
    choice = read_json('artifacts/model_choice.json')
    expected = selection.sort_values(['selected_cv_ap','experiment'],ascending=[False,True]).iloc[0].experiment
    assert choice['experiment']==expected and choice['test_used_for_selection'] is False
    checks['model_selected_by_development_cv'] = True
    clustering = pd.read_csv(ROOT/'artifacts/tables/clustering_metrics.csv')
    outliers = pd.read_csv(ROOT/'artifacts/tables/outlier_metrics.csv')
    assert len(clustering)==9 and clustering.algorithm.nunique()==3
    assert len(outliers)==3 and outliers.algorithm.nunique()==3
    checks['three_clusterers_and_three_outlier_detectors'] = True
    test = load_cohort('test')
    predictions = pd.read_csv(ROOT/'artifacts/predictions/test_predictions.csv')
    results = pd.read_csv(ROOT/'artifacts/tables/test_metrics.csv')
    assert len(results)==20 and len(predictions)==len(test)
    np.testing.assert_array_equal(test.encounter_id,predictions.encounter_id)
    assert set(predictions.patient_nbr) <= set(test.patient_nbr)
    for row in results.itertuples():
        if row.stage=='dummy':
            continue
        suffix = 'raw' if row.calibration=='none' else 'sigmoid'
        column = row.experiment+'__'+row.stage+'__'+suffix
        recomputed = metrics(test.target,predictions[column],row.threshold)
        for metric,value in recomputed.items():
            np.testing.assert_allclose(value,getattr(row,metric),rtol=1e-9,atol=1e-10)
    checks['all_saved_metrics_match_predictions'] = True
    X,_ = xy(test.head(30))
    for item in read_json('artifacts/models/index.json'):
        fitted = joblib.load(ROOT/item['file'])
        assert fitted.memory is None, 'Serialized model must not depend on a machine-specific cache.'
        column = item['experiment']+'__'+item['stage']+'__raw'
        np.testing.assert_allclose(fitted.predict_proba(X)[:,1],predictions[column].head(30),atol=1e-10)
    for item in read_json('artifacts/models/calibrated_index.json'):
        fitted = joblib.load(ROOT/item['file'])
        column = item['experiment']+'__tuned__sigmoid'
        np.testing.assert_allclose(fitted.predict_proba(X)[:,1],predictions[column].head(30),atol=1e-10)
    checks['all_18_saved_models_reproduce_predictions'] = True
    example = pd.read_csv(ROOT/'artifacts/predictions/example_predictions.csv')
    assert len(example)==3 and example.recorded_early_readmission_probability.between(0,1).all()
    checks['example_command_output_valid'] = True
    notebooks = sorted((ROOT/'notebooks').glob('*.ipynb'))
    assert len(notebooks)==5
    cell_count = 0
    for filename in notebooks:
        notebook = nbformat.read(filename,as_version=4)
        nbformat.validate(notebook)
        code = [cell for cell in notebook.cells if cell.cell_type=='code']
        assert code and all(cell.execution_count is not None for cell in code),filename
        assert not any(output.output_type=='error' for cell in code for output in cell.outputs)
        assert any(cell.outputs for cell in code)
        cell_count += len(code)
    checks['executed_notebooks'] = len(notebooks)
    checks['executed_code_cells'] = cell_count
    checks['figure_count'] = len(list((ROOT/'artifacts/figures').glob('*.png')))
    assert checks['figure_count']>=12
    assert len(pd.read_csv(ROOT/'artifacts/tables/patient_bootstrap_replicates.csv'))==200
    checks['patient_bootstrap_replicates'] = 200
    write_json('artifacts/delivery_verification.json',{'status':'passed','checks':checks,
        'note':'Artifact checks supplement the pytest integrity tests; model scores are measured outcomes, not acceptance targets.'})
    print(json.dumps(checks,indent=2))


if __name__=='__main__':
    verify()
