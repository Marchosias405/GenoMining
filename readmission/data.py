"""Download and validate UCI data, define the eligible cohort, and persist patient-disjoint splits."""
import csv
import io
import zipfile
import urllib.request
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from .common import (ROOT, NUMERIC, IDS, TARGET, EXCLUDED_DISCHARGES, DATA_URL,
                     DATA_PAGE, config, path, sha256, write_json)


def load_raw():
    # keep_default_na=False is essential: 'None' means a test was not performed.
    return pd.read_csv(path('data/raw/diabetic_data.csv'), keep_default_na=False, na_values=['?'], low_memory=False)


def eligible_cohort(raw):
    if raw['encounter_id'].duplicated().any():
        raise ValueError('Duplicate encounter IDs; investigate source rather than silently drop rows.')
    if not set(raw[TARGET].unique()) <= {'<30', '>30', 'NO'}:
        raise ValueError('Unexpected readmission label.')
    cohort = raw.loc[~raw['discharge_disposition_id'].isin(EXCLUDED_DISCHARGES)].copy()
    cohort['target'] = cohort[TARGET].eq('<30').astype('int8')
    return cohort.reset_index(drop=True)


def assign_splits(frame, seed):
    # Five grouped, approximately stratified folds; three development, one calibration, one test.
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    assignments = np.full(len(frame), 'development', dtype=object)
    folds = np.full(len(frame), -1, dtype=int)
    for fold, (_, index) in enumerate(splitter.split(frame, frame['target'], frame['patient_nbr'])):
        folds[index] = fold
        if fold == 3:
            assignments[index] = 'calibration'
        elif fold == 4:
            assignments[index] = 'test'
    result = frame[IDS].copy()
    result['split'] = assignments
    result['partition_fold'] = folds
    if result.groupby('patient_nbr')['split'].nunique().max() != 1:
        raise AssertionError('Patient leakage in split assignment.')
    if (folds < 0).any():
        raise AssertionError('An encounter was not assigned.')
    return result


def load_cohort(split=None):
    frame = pd.read_csv(path('data/processed/cohort.csv'), keep_default_na=False, na_values=['?'], low_memory=False)
    return frame if split is None else frame.loc[frame['split'].eq(split)].reset_index(drop=True)


def xy(frame):
    return frame.drop(columns=IDS + [TARGET, 'target', 'split', 'partition_fold']), frame['target']


def dictionary(raw):
    definitions = {
        'encounter_id': 'Unique encounter ID; audit/join only, never a model input.',
        'patient_nbr': 'Patient grouping ID; used to prevent split overlap, never a model input.',
        'readmitted': 'Recorded readmission: <30 positive; >30 and NO negative.',
        'race': 'Recorded race category; missing values remain missing.',
        'gender': 'Recorded gender category; Unknown/Invalid retained as a category.',
        'age': 'Age in ten-year bands; categorical, not exact age.',
        'weight': 'Weight band; subject to training-only high-missingness exclusion.',
        'admission_type_id': 'Categorical admission type; see IDS_mapping.csv.',
        'discharge_disposition_id': 'Categorical discharge destination; used for cohort exclusions.',
        'admission_source_id': 'Categorical admission source; see IDS_mapping.csv.',
        'time_in_hospital': 'Length of the completed hospital stay in days.',
        'payer_code': 'Payer category; missingness retained as an explicit category.',
        'medical_specialty': 'Admitting physician specialty category.',
        'num_lab_procedures': 'Number of laboratory tests performed during this encounter.',
        'num_procedures': 'Number of non-laboratory procedures during this encounter.',
        'num_medications': 'Number of distinct generic medications during the encounter.',
        'number_outpatient': 'Outpatient encounters in the preceding year.',
        'number_emergency': 'Emergency encounters in the preceding year.',
        'number_inpatient': 'Inpatient encounters in the preceding year.',
        'diag_1': 'Primary ICD-9 diagnosis; deterministically grouped for modeling.',
        'diag_2': 'Secondary ICD-9 diagnosis; deterministically grouped for modeling.',
        'diag_3': 'Additional ICD-9 diagnosis; deterministically grouped for modeling.',
        'number_diagnoses': 'Number of diagnoses entered into the encounter record.',
        'max_glu_serum': 'Glucose test category; None means not measured, NOT missing.',
        'A1Cresult': 'HbA1c category; None means not measured, NOT missing.',
        'change': 'Whether a diabetes medication changed during the encounter.',
        'diabetesMed': 'Whether a diabetes medication was prescribed.'}
    rows = []
    for name in raw:
        role = 'identifier' if name in IDS else ('target' if name == TARGET else 'candidate input')
        rows.append({'column': name, 'role': role,
                     'type': 'integer' if name in NUMERIC else 'categorical',
                     'description': definitions.get(name, 'Medication status: No, Steady, Up, or Down; No is a valid category.'),
                     'missing_token': '?', 'source': DATA_PAGE})
    return pd.DataFrame(rows)


def run():
    archive = path('data/raw/uci-diabetes.zip')
    if not archive.exists():
        print('Downloading public UCI dataset...', flush=True)
        with urllib.request.urlopen(DATA_URL, timeout=90) as response:
            content = response.read()
        archive.write_bytes(content)
    with zipfile.ZipFile(archive) as z:
        for filename in ['diabetic_data.csv', 'IDS_mapping.csv']:
            member = next(n for n in z.namelist() if n.endswith(filename))
            path('data/raw/' + filename).write_bytes(z.read(member))
    raw = load_raw()
    if len(raw) != 101766 or len(raw.columns) != 50 or raw['patient_nbr'].nunique() != 71518:
        raise ValueError('Source differs from the proposed dataset; inspect before continuing.')
    frame = eligible_cohort(raw)
    splits = assign_splits(frame, config()['seed'])
    frame = frame.merge(splits, on=IDS, validate='one_to_one')
    frame.to_csv(path('data/processed/cohort.csv'), index=False, na_rep='?')
    splits.to_csv(path('data/processed/splits.csv'), index=False)
    dictionary(raw).to_csv(path('artifacts/tables/data_dictionary.csv'), index=False)
    counts = frame.groupby('split').agg(encounters=('target','size'), patients=('patient_nbr','nunique'), positives=('target','sum'), positive_rate=('target','mean')).reset_index()
    counts.to_csv(path('artifacts/tables/split_summary.csv'), index=False)
    audit = {'raw_encounters': len(raw), 'raw_patients': int(raw['patient_nbr'].nunique()),
             'candidate_features': len(raw.columns)-3, 'excluded_discharge_codes': sorted(EXCLUDED_DISCHARGES),
             'excluded_encounters': len(raw)-len(frame), 'eligible_encounters': len(frame),
             'eligible_patients': int(frame['patient_nbr'].nunique()),
             'positive_encounters': int(frame['target'].sum()), 'positive_rate': float(frame['target'].mean()),
             'raw_label_counts': raw[TARGET].value_counts().to_dict(),
             'patient_overlap': 0, 'source_url': DATA_URL, 'source_doi': '10.24432/C5230J',
             'archive_sha256': sha256(archive), 'csv_sha256': sha256(path('data/raw/diabetic_data.csv')),
             'note': '11.4% refers to encounters, not unique patients. None test values are retained.'}
    write_json('artifacts/data_audit.json', audit)
    print(counts.to_string(index=False), flush=True)
