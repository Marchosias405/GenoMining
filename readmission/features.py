"""Leakage-safe clinical grouping, training-only column filtering, and raw-feature mutual information."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted
from .common import NUMERIC


def diagnosis_group(value):
    """Fixed ICD-9 group rules; no outcome information is used to construct groups."""
    if pd.isna(value):
        return np.nan
    try:
        code = float(value)
    except (ValueError, TypeError):
        return 'Other'
    if 250 <= code < 251:
        return 'Diabetes'
    for low, high, extra, label in [
        (390, 460, 785, 'Circulatory'), (460, 520, 786, 'Respiratory'),
        (520, 580, 787, 'Digestive'), (580, 630, 788, 'Genitourinary'),
        (710, 740, -1, 'Musculoskeletal'), (800, 1000, -1, 'Injury'),
        (140, 240, -1, 'Neoplasms')]:
        if low <= code < high or int(code) == extra:
            return label
    return 'Other'


class ClinicalCleaner(TransformerMixin, BaseEstimator):
    """Learn high-missingness/constant exclusions from training data, preserving test-not-performed categories."""
    def __init__(self, missing_threshold=0.95):
        self.missing_threshold = missing_threshold

    def _convert(self, X):
        out = X.copy().replace('?', np.nan)
        for col in out:
            if col in NUMERIC:
                out[col] = pd.to_numeric(out[col], errors='raise').astype(float)
            else:
                out[col] = out[col].map(lambda v: str(v) if pd.notna(v) else np.nan)
        for col in ['diag_1', 'diag_2', 'diag_3']:
            if col in out:
                out[col] = out[col].map(diagnosis_group)
        return out

    def fit(self, X, y=None):
        forbidden = {'target', 'readmitted', 'encounter_id', 'patient_nbr', 'split', 'partition_fold'} & set(X.columns)
        if forbidden:
            raise ValueError(f'Identifiers/labels entered preprocessing: {sorted(forbidden)}')
        clean = self._convert(X)
        self.feature_names_in_ = np.array(X.columns, dtype=object)
        self.columns_ = [c for c in clean if clean[c].isna().mean() <= self.missing_threshold and clean[c].nunique(dropna=False) > 1]
        self.dropped_ = [c for c in clean if c not in self.columns_]
        return self

    def transform(self, X):
        check_is_fitted(self, 'columns_')
        return self._convert(X).loc[:, self.columns_]


class RawMutualInformationSelector(TransformerMixin, BaseEstimator):
    """Rank complete raw inputs, not separate one-hot levels; all current inputs are discrete counts/categories."""
    def __init__(self, k=None, random_state=459):
        self.k = k
        self.random_state = random_state

    def fit(self, X, y):
        self.feature_names_in_ = np.array(X.columns, dtype=object)
        # Integer-valued counts and nominal categories are both discrete here.
        # Factorization is fit locally to the training fold; no encoding is applied to validation data.
        coded = np.column_stack([pd.factorize(X[c], use_na_sentinel=True)[0] for c in X])
        self.scores_ = mutual_info_classif(coded, np.asarray(y), discrete_features=True, random_state=self.random_state)
        order = np.argsort(-self.scores_, kind='stable')
        selected = set(order if self.k is None else order[:min(self.k, len(order))])
        self.columns_ = [c for i, c in enumerate(X.columns) if i in selected]
        return self

    def transform(self, X):
        check_is_fitted(self, 'columns_')
        return X.loc[:, self.columns_]


def encoding():
    numeric = Pipeline([('impute', SimpleImputer(strategy='median')), ('scale', StandardScaler())])
    categorical = Pipeline([('impute', SimpleImputer(strategy='constant', fill_value='Missing')),
                            ('onehot', OneHotEncoder(handle_unknown='infrequent_if_exist', min_frequency=25, dtype=np.float32))])
    return ColumnTransformer([
        ('numeric', numeric, make_column_selector(dtype_include=np.number)),
        ('categorical', categorical, make_column_selector(dtype_exclude=np.number))], sparse_threshold=1.0)


def make_pipeline(classifier, k=None, seed=459, memory=None):
    return Pipeline([('clean', ClinicalCleaner()),
                     ('select', RawMutualInformationSelector(k=k, random_state=seed)),
                     ('encode', encoding()), ('model', classifier)], memory=memory)
