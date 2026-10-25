"""Compare three clusterers and three outlier detectors using development data only."""
from time import perf_counter
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans, Birch
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from .common import NUMERIC, config, path, save_figure, write_json
from .data import load_cohort


def run():
    cfg = config()
    frame = load_cohort('development')
    prep = make_pipeline(SimpleImputer(strategy='median'), StandardScaler())
    Z = prep.fit_transform(np.log1p(frame[NUMERIC]))
    pca = PCA(n_components=2, random_state=cfg['seed']).fit(Z)
    coords = pca.transform(Z)
    rng = np.random.default_rng(cfg['seed'])
    draw = rng.choice(len(Z), min(6000, len(Z)), replace=False)
    scores, profiles, best_labels = [], [], {}
    fig, axes = plt.subplots(1,3, figsize=(14,4))
    factories = {
        'MiniBatchKMeans': lambda k: MiniBatchKMeans(n_clusters=k, n_init=5, batch_size=2048, random_state=cfg['seed']),
        'BIRCH': lambda k: Birch(n_clusters=k, threshold=1.5),
        'GaussianMixture': lambda k: GaussianMixture(n_components=k, covariance_type='diag', n_init=2, reg_covar=1e-4, random_state=cfg['seed'])}
    for ax, (name, factory) in zip(axes, factories.items()):
        best = None
        for k in cfg['cluster_counts']:
            start = perf_counter()
            estimator = factory(k)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                labels = estimator.fit_predict(Z)
            seconds = perf_counter()-start
            actual = len(np.unique(labels))
            if actual < 2 or actual >= len(Z):
                raise ValueError(f'{name} produced {actual} clusters; inspect configuration.')
            silhouette = silhouette_score(Z, labels, sample_size=min(cfg['silhouette_sample'],len(Z)), random_state=cfg['seed'])
            scores.append({'algorithm':name, 'requested_clusters':k, 'actual_clusters':actual,
                           'silhouette':silhouette, 'calinski_harabasz':calinski_harabasz_score(Z,labels),
                           'davies_bouldin':davies_bouldin_score(Z,labels), 'fit_seconds':seconds,
                           'encounters':len(Z), 'warnings':' | '.join(str(w.message) for w in caught)})
            if best is None or silhouette > best[0]:
                best = (silhouette, labels, k)
        _, labels, k = best
        best_labels[name] = labels
        scatter = ax.scatter(coords[draw,0], coords[draw,1], c=labels[draw], cmap='tab10', s=3, alpha=.35)
        handles,legend_labels = scatter.legend_elements(num=list(range(k)))
        ax.legend(handles,legend_labels,title='Cluster',fontsize=7,title_fontsize=8,loc='upper right')
        ax.set(title=f'{name}: k={k}', xlabel=f'PC1 ({pca.explained_variance_ratio_[0]:.0%} variance)', ylabel=f'PC2 ({pca.explained_variance_ratio_[1]:.0%} variance)')
        profile = frame[NUMERIC+['target']].assign(cluster=labels).groupby('cluster').mean()
        profile['encounters'] = pd.Series(labels).value_counts()
        profile['algorithm'] = name
        profiles.append(profile.reset_index().rename(columns={'target':'descriptive_readmission_rate'}))
        print(f'Clustering {name}: selected k={k}, silhouette={best[0]:.3f}', flush=True)
    fig.suptitle('Development clusters: 2D projection of log-scaled numerical inputs')
    fig.tight_layout()
    save_figure(fig, 'clusters_pca')
    pd.DataFrame(scores).to_csv(path('artifacts/tables/clustering_metrics.csv'), index=False)
    pd.concat(profiles,ignore_index=True).to_csv(path('artifacts/tables/cluster_profiles.csv'), index=False)
    labels_table = frame[['encounter_id','patient_nbr']].copy()
    for name, labels in best_labels.items():
        labels_table[name] = labels
    labels_table.to_csv(path('artifacts/tables/cluster_assignments.csv'), index=False)

    # A shared development subsample bounds LOF/covariance cost and keeps detector comparisons aligned.
    sample = frame.sample(n=min(cfg['outlier_sample'],len(frame)), random_state=cfg['seed']).reset_index(drop=True)
    S = make_pipeline(SimpleImputer(strategy='median'),StandardScaler()).fit_transform(np.log1p(sample[NUMERIC]))
    sample_pca = PCA(n_components=5, random_state=cfg['seed']).fit(S)
    reduced = sample_pca.transform(S)
    methods = {'IsolationForest': (IsolationForest(n_estimators=150, contamination=cfg['outlier_fraction'], random_state=cfg['seed'], n_jobs=cfg['threads']), S),
               'LocalOutlierFactor': (LocalOutlierFactor(n_neighbors=35, contamination=cfg['outlier_fraction'], n_jobs=cfg['threads']), S),
               'EllipticEnvelope': (EllipticEnvelope(contamination=cfg['outlier_fraction'], support_fraction=.8, random_state=cfg['seed']), reduced)}
    fig, axes = plt.subplots(1,3, figsize=(14,4))
    flags, overview, details = {}, [], []
    assignments = sample[['encounter_id','patient_nbr']].copy()
    for ax, (name, (estimator, values)) in zip(axes, methods.items()):
        start = perf_counter()
        flag = estimator.fit_predict(values) == -1
        seconds = perf_counter()-start
        flags[name] = flag
        assignments[name] = flag
        overview.append({'algorithm':name, 'sample_encounters':len(sample), 'flagged':int(flag.sum()),
                         'fraction':float(flag.mean()), 'fit_seconds':seconds,
                         'readmission_rate_flagged':float(sample.loc[flag,'target'].mean()),
                         'readmission_rate_unflagged':float(sample.loc[~flag,'target'].mean())})
        detail = sample[NUMERIC].assign(flagged=flag).groupby('flagged').median().reset_index()
        detail['algorithm'] = name
        details.append(detail)
        ax.scatter(reduced[~flag,0],reduced[~flag,1],s=3,alpha=.2,color='#26758c',label='Retained/unflagged')
        ax.scatter(reduced[flag,0],reduced[flag,1],s=6,alpha=.5,color='#cb5939',label='Flagged')
        ax.set(title=f'{name}: {flag.sum():,} flags',xlabel='PC1',ylabel='PC2')
        ax.legend(fontsize=7)
    fig.suptitle('Outlier candidates: all valid encounters remain in classifier training')
    fig.tight_layout()
    save_figure(fig, 'outliers_pca')
    pd.DataFrame(overview).to_csv(path('artifacts/tables/outlier_metrics.csv'), index=False)
    pd.concat(details,ignore_index=True).to_csv(path('artifacts/tables/outlier_profiles.csv'), index=False)
    assignments.to_csv(path('artifacts/tables/outlier_assignments.csv'), index=False)
    overlap = pd.DataFrame({a:{b:float((flags[a]&flags[b]).sum()/max(1,(flags[a]|flags[b]).sum())) for b in flags} for a in flags})
    overlap.to_csv(path('artifacts/tables/outlier_overlap.csv'))
    write_json('artifacts/unsupervised_design.json', {
        'partition':'development only', 'inputs':NUMERIC, 'scaling':'log1p -> median imputation -> standardization',
        'cluster_selection':'Maximum sampled silhouette among k=3,5,8, independently for each algorithm; labels unused.',
        'cluster_pca_explained_variance':pca.explained_variance_ratio_.tolist(),
        'outlier_sample':len(sample), 'outlier_fraction_target':cfg['outlier_fraction'],
        'elliptic_input':'Five PCA components to avoid singular covariance; approximate elliptical distribution assumption.',
        'outlier_decision':'Keep all eligible encounters. Flags are unusualness, not proven data error; no filtering of test/calibration.',
        'limitations':'Numeric-only exploratory clusters omit categorical structure; PCA is a visualization, not clinical validation. LOF is transductive on this sample.'})
