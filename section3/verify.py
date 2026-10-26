"""Verify focused artifacts, patient and row alignment, notebook outputs, tests and report evidence before packaging."""
import json
import xml.etree.ElementTree as ET
import nbformat
import numpy as np
import pandas as pd
from scipy import sparse
from readmission.common import ROOT
from readmission.data import load_cohort
from .run import current
from .analysis import OUT, write_json


def verify():
    assert current(), 'Analysis outputs are missing or stale; run python -m section3.run.'
    s = json.loads((OUT/'summary.json').read_text(encoding='utf-8'))
    cohort = load_cohort()
    dev = cohort.loc[cohort.split.eq('development')].reset_index(drop=True)
    assert cohort.groupby('patient_nbr')['split'].nunique().max() == 1
    assert len(dev) == s['development_encounters'] and int(dev.target.sum()) == s['development_positives']
    ledger = pd.read_csv(OUT/'data/development_row_index.csv')
    matrix = sparse.load_npz(OUT/'data/development_top20.npz')
    scores = np.load(OUT/'data/development_pca.npz')['scores']
    np.testing.assert_array_equal(ledger.encounter_id, dev.encounter_id)
    np.testing.assert_array_equal(ledger.target, dev.target)
    assert matrix.shape == (len(dev), s['selected_encoded']['columns'])
    assert scores.shape == (len(dev), s['pca_components'])
    assert np.isfinite(matrix.data).all() and np.isfinite(scores).all()
    ranking = pd.read_csv(OUT/'tables/mutual_information.csv')
    assert ranking.selected_top20.sum() == s['selected_raw_inputs'] == 20
    assert s['outlier_rows_removed'] == s['synthetic_rows_added'] == 0
    assert s['pca_retained_variance'] >= .95
    notebook = nbformat.read(ROOT/'section3/section3_eda_preprocessing.ipynb', as_version=4)
    cells = [c for c in notebook.cells if c.cell_type == 'code']
    assert cells and all(c.execution_count is not None for c in cells)
    assert not any(o.output_type == 'error' for c in cells for o in c.get('outputs', []))
    suites = ET.parse(OUT/'test_results.xml').getroot().findall('.//testsuite')
    assert sum(int(s.attrib.get('tests', 0)) for s in suites) == 6
    assert all(int(s.attrib.get('failures', 0)) == int(s.attrib.get('errors', 0)) == int(s.attrib.get('skipped', 0)) == 0 for s in suites)
    assert len(list((OUT/'figures').glob('*.png'))) == 4
    reference = (ROOT/'section3/reports/section3_reference_report.md').read_text(encoding='utf-8')
    assert reference.count('![Figure ') == 4
    assert (ROOT/'section3/reports/section3_reference_report.pdf').stat().st_size > 10000
    assert f"{s['pca_components']} components" in reference
    assert f"{s['all_encoded']['columns']}-column" in reference
    assert f"{s['selected_encoded']['columns']} encoded columns" in reference
    checks = {'status': 'passed', 'analysis_and_output_hashes_current': True, 'zero_patient_overlap': True,
              'matrix_and_pca_rows_aligned': True, 'finite_transformed_values': True,
              'focused_tests_passed': 6, 'executed_notebooks': 1, 'executed_code_cells': len(cells),
              'essential_figures': 4, 'reference_report_present': True,
              'note': 'PDF page layout is visually reviewed separately; these checks verify reproducible evidence, not grading outcomes.'}
    write_json('verification.json', checks)
    print(json.dumps(checks, indent=2))
    return checks


if __name__ == '__main__':
    verify()
