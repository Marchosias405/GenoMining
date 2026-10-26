"""Produce development-only EDA, auditable preprocessing, outlier screening, MI selection and PCA evidence."""
from pathlib import Path
from time import perf_counter
import json

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from readmission.common import ROOT, NUMERIC, config, read_json, sha256, environment
from readmission.data import load_cohort, load_raw, xy
from readmission.features import ClinicalCleaner, RawMutualInformationSelector, encoding

OUT = ROOT / 'section3' / 'outputs'
BLUE, ORANGE = '#246b83', '#c26b32'
LABELS = {'time_in_hospital': 'Length of stay (days)', 'num_lab_procedures': 'Laboratory procedures',
          'num_procedures': 'Other procedures', 'num_medications': 'Medications',
          'number_outpatient': 'Prior outpatient visits', 'number_emergency': 'Prior emergency visits',
          'number_inpatient': 'Prior inpatient visits', 'number_diagnoses': 'Diagnoses'}


def output(relative):
    """Resolve generated output paths in the dedicated Section 3 directory."""
    target = OUT / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def write_json(relative, value):
    output(relative).write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def save_table(frame, name, index=False):
    frame.to_csv(output('tables/' + name + '.csv'), index=index)


def save_figure(fig, name, dpi):
    fig.savefig(output('figures/' + name + '.png'), dpi=dpi, bbox_inches='tight')
    plt.close(fig)


def preprocessing(k=None):
    """A fitted transformer only: cleaning -> independent MI filtering -> imputation/scaling/encoding."""
    return Pipeline([('clean', ClinicalCleaner()),
                     ('select', RawMutualInformationSelector(k=k, random_state=config()['seed'])),
                     ('encode', encoding())])


def iqr_screen(values):
    """Flag observations outside 1.5 IQR fences for review, including explicit zero-IQR diagnostics."""
    rows, flags = [], pd.DataFrame(index=values.index)
    for name in values:
        q1, q3 = values[name].quantile([.25, .75])
        spread = q3 - q1
        low, high = q1 - 1.5 * spread, q3 + 1.5 * spread
        flag = values[name].lt(low) | values[name].gt(high)
        flags[name] = flag
        rows.append({'feature': name, 'q1': q1, 'q3': q3, 'iqr': spread,
                     'lower_fence': low, 'upper_fence': high, 'zero_iqr': bool(spread == 0),
                     'flagged_encounters': int(flag.sum()), 'flagged_percent': float(flag.mean() * 100)})
    return pd.DataFrame(rows), flags


def matrix_details(matrix):
    nnz = matrix.nnz if sparse.issparse(matrix) else np.count_nonzero(matrix)
    return {'rows': int(matrix.shape[0]), 'columns': int(matrix.shape[1]),
            'density': float(nnz / (matrix.shape[0] * matrix.shape[1])),
            'nonfinite_values': int((~np.isfinite(matrix.data if sparse.issparse(matrix) else matrix)).sum())}


def run():
    """Run the complete section once and return measured results; no classifier or held-out outcome is analyzed."""
    start = perf_counter()
    settings = json.loads((ROOT / 'section3/config.json').read_text(encoding='utf-8'))
    seed, dpi = config()['seed'], settings['figure_dpi']
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    raw = load_raw()
    cohort = load_cohort()
    dev = cohort.loc[cohort['split'].eq('development')].reset_index(drop=True)
    X, y = xy(dev)
    assert cohort.groupby('patient_nbr')['split'].nunique().max() == 1
    assert not raw['encounter_id'].duplicated().any()
    assert (dev[NUMERIC] >= 0).all().all()

    # EDA uses development encounters only. Full-cohort counts describe construction, not associations.
    missing = pd.DataFrame({'feature': X.columns, 'missing_count': X.isna().sum().values,
                            'missing_percent': X.isna().mean().values * 100})
    missing = missing.sort_values('missing_percent', ascending=False, kind='stable')
    save_table(missing, 'missingness')
    numeric = dev[NUMERIC].describe(percentiles=[.01, .25, .5, .75, .99]).T
    numeric['zero_percent'] = dev[NUMERIC].eq(0).mean() * 100
    numeric['skewness'] = dev[NUMERIC].skew()
    save_table(numeric.rename_axis('feature').reset_index(), 'numeric_summary')
    groups = dev.groupby('split').agg(encounters=('target', 'size'), patients=('patient_nbr', 'nunique'),
                                      early_readmissions=('target', 'sum'), positive_rate=('target', 'mean'))
    save_table(groups.reset_index(), 'analysis_population')
    split_sizes = cohort.groupby('split').agg(encounters=('encounter_id', 'size'), patients=('patient_nbr', 'nunique'))
    save_table(split_sizes.reset_index(), 'split_sizes')
    correlations = dev[NUMERIC].corr(method='spearman')
    save_table(correlations.rename_axis('feature').reset_index(), 'spearman_correlations')
    pairs = pd.DataFrame([{'feature_a': a, 'feature_b': b, 'spearman_rho': float(correlations.loc[a, b])}
                          for i, a in enumerate(NUMERIC) for b in NUMERIC[i+1:]])
    pairs['absolute_rho'] = pairs.spearman_rho.abs()
    pairs = pairs.sort_values('absolute_rho', ascending=False)
    save_table(pairs, 'correlation_pairs')
    categories = []
    for name in ['age', 'A1Cresult', 'discharge_disposition_id']:
        part = dev.groupby(name, dropna=False).agg(encounters=('target', 'size'), early_readmissions=('target', 'sum'),
                                                  observed_rate=('target', 'mean')).reset_index().rename(columns={name: 'category'})
        part.insert(0, 'feature', name)
        categories.append(part)
    save_table(pd.concat(categories, ignore_index=True), 'categorical_associations')
    lab_states = []
    for name in ['A1Cresult', 'max_glu_serum']:
        for value, count in X[name].fillna('Missing').value_counts().items():
            lab_states.append({'feature': name, 'category': value, 'encounters': int(count), 'percent': float(count / len(dev) * 100)})
    save_table(pd.DataFrame(lab_states), 'laboratory_categories')

    # These are standalone preprocessing artifacts, not preprocessed inputs for cross-validation.
    print('Fitting development-only preprocessing and mutual-information selection...', flush=True)
    full = preprocessing().fit(X, y)
    selected = preprocessing(settings['selected_features']).fit(X, y)
    clean = full.named_steps['clean'].transform(X)
    all_matrix = sparse.csr_matrix(full.transform(X), dtype=np.float32)
    selected_matrix = sparse.csr_matrix(selected.transform(X), dtype=np.float32)
    assert matrix_details(all_matrix)['nonfinite_values'] == 0
    assert matrix_details(selected_matrix)['nonfinite_values'] == 0
    cleaner, selector = full.named_steps['clean'], selected.named_steps['select']
    changes = []
    converted = cleaner._convert(X)
    for name in X:
        reason = 'retained'
        if name in cleaner.dropped_:
            reason = 'over 95% missing' if converted[name].isna().mean() > .95 else 'constant after diagnosis grouping'
        changes.append({'feature': name, 'retained': name in clean, 'reason': reason,
                        'raw_missing_percent': float(X[name].isna().mean()*100),
                        'distinct_after_grouping_including_missing': int(converted[name].nunique(dropna=False))})
    save_table(pd.DataFrame(changes), 'column_decisions')
    ranking = pd.DataFrame({'feature': selector.feature_names_in_, 'mutual_information_nats': selector.scores_})
    ranking['selected_top20'] = ranking.feature.isin(selector.columns_)
    ranking = ranking.sort_values('mutual_information_nats', ascending=False, kind='stable')
    ranking.insert(0, 'rank', np.arange(1, len(ranking)+1))
    save_table(ranking, 'mutual_information')

    enc = full.named_steps['encode']
    num_columns = list(enc.transformers_[0][2])
    numeric_pipe = enc.named_transformers_['numeric']
    scaled_numeric = numeric_pipe.transform(clean[num_columns])
    audit = pd.DataFrame({'feature': num_columns, 'missing_before': clean[num_columns].isna().sum().values,
                          'imputation_median': numeric_pipe.named_steps['impute'].statistics_,
                          'training_mean': numeric_pipe.named_steps['scale'].mean_,
                          'training_scale': numeric_pipe.named_steps['scale'].scale_,
                          'mean_after': scaled_numeric.mean(axis=0), 'population_std_after': scaled_numeric.std(axis=0)})
    save_table(audit, 'numeric_transform_audit')
    cat_columns = list(enc.transformers_[1][2])
    cat_encoder = enc.named_transformers_['categorical'].named_steps['onehot']
    categorical = []
    for name, known, rare in zip(cat_columns, cat_encoder.categories_, cat_encoder.infrequent_categories_):
        rare_count = 0 if rare is None else len(rare)
        categorical.append({'feature': name, 'training_categories_after_imputation': len(known),
                            'pooled_rare_categories': rare_count, 'encoded_columns': len(known)-rare_count+int(rare_count>0)})
    save_table(pd.DataFrame(categorical), 'categorical_encoding_audit')
    selected_names = selected.named_steps['encode'].get_feature_names_out()
    save_table(pd.DataFrame({'matrix_column': np.arange(len(selected_names)), 'encoded_feature': selected_names}), 'encoded_feature_names')
    # Matrix row order is explicit, so IDs and labels can never silently drift away from features.
    ledger = dev[['encounter_id', 'patient_nbr', 'target']].copy()
    ledger.insert(0, 'matrix_row', np.arange(len(dev)))
    ledger.to_csv(output('data/development_row_index.csv'), index=False)
    sparse.save_npz(output('data/development_top20.npz'), selected_matrix)
    joblib.dump(selected, output('transformers/preprocessor_top20.joblib'))
    joblib.dump(full, output('transformers/preprocessor_all.joblib'))
    # A compact example connects raw inputs with the numerical operations, without a huge wide preview.
    examples = []
    for row in range(3):
        for j, name in enumerate(num_columns):
            examples.append({'example_row': row, 'feature': name, 'raw_value': float(X.iloc[row][name]),
                             'imputed_value': float(numeric_pipe.named_steps['impute'].transform(clean[num_columns].iloc[[row]])[0, j]),
                             'standardized_value': float(scaled_numeric[row, j])})
    save_table(pd.DataFrame(examples), 'transformation_examples')

    print('Screening outliers without deleting encounters...', flush=True)
    fences, iqr_flags = iqr_screen(dev[NUMERIC])
    save_table(fences, 'outlier_iqr')
    anomaly_scaler = Pipeline([('impute', SimpleImputer(strategy='median')), ('scale', StandardScaler())])
    anomaly_matrix = anomaly_scaler.fit_transform(np.log1p(dev[NUMERIC]))
    detector = IsolationForest(n_estimators=settings['isolation_trees'], contamination=settings['outlier_flag_fraction'],
                               random_state=seed, n_jobs=config()['threads']).fit(anomaly_matrix)
    isolated = detector.predict(anomaly_matrix) == -1
    outlier_rows = ledger[['matrix_row', 'encounter_id']].copy()
    outlier_rows['iqr_any_flag'] = iqr_flags.any(axis=1)
    outlier_rows['isolation_forest_flag'] = isolated
    outlier_rows.to_csv(output('data/outlier_flags.csv'), index=False)
    profile = dev[NUMERIC].assign(flagged=isolated).groupby('flagged').median().reset_index()
    profile.insert(1, 'encounters', [int((isolated == flag).sum()) for flag in profile.flagged])
    save_table(profile, 'outlier_profiles')

    # PCA addresses the actual encoded dimensionality, rather than applying only to 8 numeric inputs.
    print('Fitting centered PCA on all development-encoded inputs...', flush=True)
    dense = all_matrix.toarray()
    pca = PCA(n_components=settings['pca_retained_variance'], svd_solver='full')
    scores = pca.fit_transform(dense)
    explained = pd.DataFrame({'component': np.arange(1, len(pca.explained_variance_ratio_)+1),
                              'variance_ratio': pca.explained_variance_ratio_,
                              'cumulative_variance_ratio': np.cumsum(pca.explained_variance_ratio_)})
    save_table(explained, 'pca_variance')
    names = full.named_steps['encode'].get_feature_names_out()
    save_table(pd.DataFrame(pca.components_[:2].T, index=names, columns=['PC1_loading', 'PC2_loading']).rename_axis('encoded_feature').reset_index(), 'pca_loadings')
    np.savez_compressed(output('data/development_pca.npz'), scores=scores)
    joblib.dump(pca, output('transformers/pca.joblib'))
    del dense

    # Four essential multi-panel figures keep the report focused on the supplied rubric.
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4), layout='constrained')
    counts = y.value_counts().reindex([0, 1])
    axes[0].bar(['No recorded early', 'Recorded <30'], counts, color=[BLUE, ORANGE])
    for i, count in enumerate(counts):
        axes[0].text(i, count+len(dev)*.018, f'{count:,}\n{count/len(dev):.2%}', ha='center')
    axes[0].set(title='Development outcome distribution', ylabel='Encounters', ylim=(0, len(dev)*1.05))
    top = missing.loc[missing.missing_count.gt(0)].iloc[::-1]
    axes[1].barh(top.feature, top.missing_percent, color=BLUE)
    axes[1].set(title='Missing values represented by ?', xlabel='Missing values (%)', xlim=(0, 108))
    for i, value in enumerate(top.missing_percent):
        axes[1].text(value+1, i, f'{value:.2f}', va='center', fontsize=9)
    save_figure(fig, '01_population_missingness', dpi)

    key = ['time_in_hospital', 'num_medications', 'number_inpatient', 'number_emergency']
    fig, axes = plt.subplots(2, 4, figsize=(12, 5.8), layout='constrained')
    for i, name in enumerate(key):
        axes[0, i].hist(dev[name], bins=25, color=BLUE, edgecolor='white')
        axes[0, i].set(title=LABELS[name], ylabel='Count (log scale)', yscale='log', xlabel='Observed value')
        axes[1, i].boxplot([dev.loc[y.eq(0), name], dev.loc[y.eq(1), name]], tick_labels=['No early', '<30'],
                           whis=(5, 95), showfliers=False, widths=.55)
        axes[1, i].set(ylabel='Observed value', xlabel='Recorded outcome')
    fig.suptitle('Development distributions and outcome comparisons\nBoxplots use 5th-95th percentile whiskers; all encounters remain in the data', fontsize=11)
    save_figure(fig, '02_distributions_boxplots', dpi)

    fig, ax = plt.subplots(figsize=(9.2, 6.2), layout='constrained')
    im = ax.imshow(correlations, vmin=-1, vmax=1, cmap='RdBu_r')
    labels = [LABELS[n] for n in NUMERIC]
    ax.set_xticks(range(8), labels, rotation=40, ha='right')
    ax.set_yticks(range(8), labels)
    for i in range(8):
        for j in range(8):
            value = correlations.iloc[i, j]
            ax.text(j, i, f'{value:.2f}', ha='center', va='center', color='white' if abs(value)>.65 else 'black', fontsize=9)
    ax.set_title('Development Spearman rank correlations')
    fig.colorbar(im, ax=ax, label='Spearman rho', shrink=.8)
    save_figure(fig, '03_correlations', dpi)

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 6), layout='constrained')
    displayed = ranking.head(settings['selected_features']).iloc[::-1]
    axes[0].barh(displayed.feature, displayed.mutual_information_nats, color=BLUE)
    axes[0].set(title='Top 20 raw inputs by mutual information', xlabel='Estimated mutual information (nats)')
    axes[1].plot(explained.component, explained.cumulative_variance_ratio, color=BLUE, linewidth=2)
    axes[1].axhline(settings['pca_retained_variance'], color=ORANGE, ls='--', label='95% retained variance')
    axes[1].scatter([pca.n_components_], [explained.cumulative_variance_ratio.iloc[-1]], color=ORANGE)
    axes[1].set(title=f'PCA of {all_matrix.shape[1]} encoded inputs', xlabel='Number of components',
                 ylabel='Cumulative explained variance', ylim=(0, 1.02))
    axes[1].legend(loc='lower right')
    save_figure(fig, '04_selection_pca', dpi)

    summary = {'scope': 'Section 3 only; development EDA and preprocessing; no classifier results',
               'data': read_json('artifacts/data_audit.json'), 'development_encounters': len(dev),
               'development_patients': int(dev.patient_nbr.nunique()), 'development_positives': int(y.sum()),
               'development_positive_rate': float(y.mean()), 'numeric_inputs': len(NUMERIC),
               'categorical_candidate_inputs': int(len(X.columns)-len(NUMERIC)),
               'retained_raw_inputs': len(clean.columns), 'selected_raw_inputs': len(selector.columns_),
               'dropped_columns': cleaner.dropped_, 'all_encoded': matrix_details(all_matrix),
               'selected_encoded': matrix_details(selected_matrix),
               'pca_components': int(pca.n_components_), 'pca_retained_variance': float(pca.explained_variance_ratio_.sum()),
               'pca_first_two_variance': float(pca.explained_variance_ratio_[:2].sum()),
               'strongest_numeric_correlation': pairs.iloc[0].drop('absolute_rho').to_dict(),
               'pairs_with_abs_rho_at_least_0_8': int(pairs.absolute_rho.ge(settings['strong_correlation_threshold']).sum()),
               'iqr_any_flagged': int(iqr_flags.any(axis=1).sum()), 'isolation_flagged': int(isolated.sum()),
               'outlier_rows_removed': 0, 'synthetic_rows_added': 0,
               'numeric_missing_cells': int(X[NUMERIC].isna().sum().sum()),
               'duplicate_encounter_ids': int(raw.encounter_id.duplicated().sum()),
               'settings': settings, 'environment': environment(), 'elapsed_seconds': perf_counter()-start,
               'fit_scope': 'Every learned transformation in this section fits development only. Refit inside each training fold for later CV.',
               'pca_scope': 'Centered numeric-standardized and binary-one-hot representation; labels are not inputs to PCA. Separate diagnostic alternative, not a change to the proposal classifier pipeline.',
               'outlier_scope': 'IQR screening plus one IsolationForest on the full development set. Flags identify unusual records, not confirmed errors. No labels enter the detector.',
               'augmentation_scope': 'None. Original class prevalence is retained. Oversampling before splitting would risk leakage; SMOTE is not required by this section.'}
    write_json('summary.json', summary)
    return summary
