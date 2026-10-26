"""Check Section 3 leakage boundaries, missingness, outlier interpretation and reproducible matrix/PCA exports."""
import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from section3.analysis import preprocessing, iqr_screen, OUT
from readmission.common import NUMERIC
from readmission.data import load_cohort, xy


def test_median_and_scaler_learn_training_values_only():
    X = pd.DataFrame({'time_in_hospital': [1., 2., np.nan, 5.], 'age': ['young', 'old', 'young', 'old']})
    model = preprocessing().fit(X, np.array([0, 1, 0, 1]))
    numeric = model.named_steps['encode'].named_transformers_['numeric']
    np.testing.assert_allclose(numeric.named_steps['impute'].statistics_, [2.])
    np.testing.assert_allclose(numeric.named_steps['scale'].mean_, [2.5])
    before = numeric.named_steps['scale'].mean_.copy()
    validation = pd.DataFrame({'time_in_hospital': [1000., np.nan], 'age': ['new', 'old']})
    encoded = model.transform(validation)
    np.testing.assert_array_equal(before, numeric.named_steps['scale'].mean_)
    assert encoded.shape[0] == 2
    assert np.isfinite(encoded.toarray() if sparse.issparse(encoded) else encoded).all()


def test_mutual_information_selects_whole_input_before_onehot():
    y = np.tile([0, 1], 100)
    X = pd.DataFrame({'age': np.where(y, 'older', 'younger'), 'time_in_hospital': np.tile([1, 1, 2, 2], 50)})
    model = preprocessing(k=1).fit(X, y)
    assert model.named_steps['select'].columns_ == ['age']
    assert model.transform(X).shape == (200, 2)


def test_zero_iqr_flags_are_not_automatic_deletions():
    values = pd.DataFrame({'number_emergency': [0]*98+[1, 20]})
    original = values.copy()
    summary, flags = iqr_screen(values)
    assert summary.iloc[0].zero_iqr
    assert flags.number_emergency.sum() == 2
    pd.testing.assert_frame_equal(values, original)


def test_exported_matrix_preserves_rows_and_reproduces_transform():
    dev = load_cohort('development')
    X, _ = xy(dev)
    matrix = sparse.load_npz(OUT/'data/development_top20.npz')
    ledger = pd.read_csv(OUT/'data/development_row_index.csv')
    model = joblib.load(OUT/'transformers/preprocessor_top20.joblib')
    assert matrix.shape[0] == len(ledger) == len(dev)
    np.testing.assert_array_equal(ledger.encounter_id, dev.encounter_id)
    np.testing.assert_array_equal(ledger.target, dev.target)
    np.testing.assert_allclose(model.transform(X.iloc[:128]).toarray(), matrix[:128].toarray(), atol=1e-6, rtol=1e-6)
    assert np.isfinite(matrix.data).all()


def test_pca_export_reproduces_and_meets_variance_rule():
    X, _ = xy(load_cohort('development').iloc[:128])
    model = joblib.load(OUT/'transformers/preprocessor_all.joblib')
    pca = joblib.load(OUT/'transformers/pca.joblib')
    matrix = sparse.csr_matrix(model.transform(X), dtype=np.float32).toarray()
    saved = np.load(OUT/'data/development_pca.npz')['scores'][:128]
    np.testing.assert_allclose(pca.transform(matrix), saved, atol=1e-4, rtol=1e-4)
    assert pca.explained_variance_ratio_.sum() >= .95
    assert pca.explained_variance_ratio_[:-1].sum() < .95


def test_audits_match_development_and_no_outliers_are_removed():
    dev = load_cohort('development')
    X, _ = xy(dev)
    decisions = pd.read_csv(OUT/'tables/column_decisions.csv')
    numeric = pd.read_csv(OUT/'tables/numeric_transform_audit.csv')
    flags = pd.read_csv(OUT/'data/outlier_flags.csv')
    assert len(decisions) == 47 and decisions.retained.sum() == 42
    assert len(flags) == len(dev)
    np.testing.assert_allclose(numeric.mean_after, 0, atol=1e-12)
    np.testing.assert_allclose(numeric.population_std_after, 1, atol=1e-12)
    assert numeric.missing_before.sum() == X[NUMERIC].isna().sum().sum() == 0
    assert X['A1Cresult'].eq('None').any()
