"""Regression checks for target semantics, patient leakage, fold-local preprocessing, and frozen calibration."""
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from readmission.data import eligible_cohort,assign_splits,load_raw,load_cohort,xy
from readmission.features import ClinicalCleaner,RawMutualInformationSelector,make_pipeline,diagnosis_group
from readmission.evaluation import metrics


def test_cohort_excludes_encounters_not_whole_patients():
    raw = pd.DataFrame({'encounter_id':range(5),'patient_nbr':[1,1,2,3,4],
                        'discharge_disposition_id':[11,1,13,1,1],'readmitted':['NO','<30','NO','>30','NO']})
    result = eligible_cohort(raw)
    assert result.encounter_id.tolist()==[1,3,4]
    assert result.target.tolist()==[1,0,0]
    assert 1 in result.patient_nbr.values


def test_bad_targets_and_duplicate_encounters_fail():
    raw = pd.DataFrame({'encounter_id':[1,1],'patient_nbr':[1,1],'discharge_disposition_id':[1,1],'readmitted':['NO','<30']})
    with pytest.raises(ValueError,match='Duplicate'):
        eligible_cohort(raw)
    raw.loc[1,'encounter_id']=2
    raw.loc[1,'readmitted']='UNKNOWN'
    with pytest.raises(ValueError,match='Unexpected'):
        eligible_cohort(raw)


def test_groups_never_cross_splits_and_seed_is_repeatable():
    frame = pd.DataFrame({'encounter_id':range(200),'patient_nbr':np.repeat(np.arange(50),4),'target':np.tile([0,0,0,1],50)})
    first,second = assign_splits(frame,459),assign_splits(frame,459)
    pd.testing.assert_frame_equal(first,second)
    assert first.groupby('patient_nbr')['split'].nunique().eq(1).all()
    assert set(first.split)=={'development','calibration','test'}


def test_exclusions_are_learned_from_training_and_none_is_not_missing():
    train = pd.DataFrame({'A1Cresult':['None','Norm']*50,'weight':[np.nan]*98+['50','75'],
                          'constant':['No']*100,'time_in_hospital':np.arange(100)%14+1})
    cleaner = ClinicalCleaner().fit(train)
    validation = train.copy()
    validation['weight']='100'
    validation['constant']=np.arange(100).astype(str)
    result = cleaner.transform(validation)
    assert 'weight' not in result and 'constant' not in result
    assert result['A1Cresult'].eq('None').sum()==50


def test_identifiers_and_target_cannot_enter_training():
    for forbidden in ['patient_nbr','encounter_id','readmitted','target','split']:
        with pytest.raises(ValueError,match='Identifiers/labels'):
            ClinicalCleaner().fit(pd.DataFrame({forbidden:[0,1],'age':['[0-10)','[10-20)']}))


def test_diagnosis_grouping_preserves_missing_and_handles_v_codes():
    assert diagnosis_group('250.13')=='Diabetes'
    assert diagnosis_group('428')=='Circulatory'
    assert diagnosis_group('V58')=='Other'
    assert pd.isna(diagnosis_group(np.nan))


def test_mutual_information_selects_signal_and_transform_does_not_refit():
    y = np.tile([0,1],100)
    train = pd.DataFrame({'signal':y.astype(str),'noise':np.tile([0,0,1,1],50)})
    selector = RawMutualInformationSelector(k=1).fit(train,y)
    assert selector.columns_==['signal']
    before = selector.scores_.copy()
    assert list(selector.transform(train.iloc[:5]))==['signal']
    np.testing.assert_array_equal(before,selector.scores_)


def test_unknown_categories_and_independent_calibration_do_not_refit_base():
    X = pd.DataFrame({'age':['[20-30)','[60-70)']*60,'time_in_hospital':np.tile([1,5,2,7],30)})
    y = np.tile([0,1,0,1],30)
    model = make_pipeline(LogisticRegression(solver='liblinear')).fit(X,y)
    before = model.named_steps['model'].coef_.copy()
    calibration_X = X.copy()
    calibration_X.loc[:10,'age']='new category'
    calibrated = CalibratedClassifierCV(FrozenEstimator(model),method='sigmoid').fit(calibration_X,y)
    np.testing.assert_array_equal(before,model.named_steps['model'].coef_)
    p = calibrated.predict_proba(calibration_X)[:,1]
    assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all()


def test_majority_baseline_exposes_imbalance():
    result = metrics(np.array([0]*9+[1]),np.zeros(10))
    assert result['accuracy']==.9 and result['recall']==0
    assert result['roc_auc']==.5 and result['average_precision']==.1


@pytest.mark.parametrize('family',['random_forest','xgboost'])
def test_tree_models_support_frozen_calibration_with_mixed_inputs(family):
    from readmission.models import candidates
    X = pd.DataFrame({'age':['[20-30)','[60-70)']*60,'time_in_hospital':np.tile([1,5,2,7],30)})
    y = np.tile([0,1,0,1],30)
    classifier,_ = candidates({'seed':459,'threads':1})[family]
    model = make_pipeline(classifier).fit(X,y)
    before = model.predict_proba(X).copy()
    calibration_X = X.copy()
    calibration_X.loc[:10,'age']='previously unseen'
    calibrated = CalibratedClassifierCV(FrozenEstimator(model),method='sigmoid').fit(calibration_X,y)
    np.testing.assert_array_equal(before,model.predict_proba(X))
    probability = calibrated.predict_proba(calibration_X)[:,1]
    assert np.isfinite(probability).all() and ((probability>=0)&(probability<=1)).all()


def test_real_data_semantics_and_proposal_counts():
    raw = load_raw()
    assert raw['A1Cresult'].eq('None').any()
    assert raw['max_glu_serum'].eq('None').any()
    assert raw['weight'].isna().sum()==98569
    cohort = load_cohort()
    assert len(cohort)==99343 and cohort.patient_nbr.nunique()==69990
    assert cohort.target.sum()==11314
    assert cohort.groupby('patient_nbr')['split'].nunique().eq(1).all()
    inputs,_ = xy(cohort)
    assert len(inputs.columns)==47
    assert not {'patient_nbr','encounter_id','target','readmitted'} & set(inputs.columns)
