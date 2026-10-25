"""Development-only exploratory figures and tables, leaving held-out outcomes out of model design."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from .common import NUMERIC, path, save_figure, write_json
from .data import load_cohort, xy
from .features import ClinicalCleaner


def run():
    frame = load_cohort('development')
    X, y = xy(frame)
    cleaner = ClinicalCleaner().fit(X)
    clean = cleaner.transform(X)
    missing = X.isna().mean().sort_values(ascending=False).rename('missing_fraction').rename_axis('feature').reset_index()
    missing.to_csv(path('artifacts/tables/missingness.csv'), index=False)
    frame[NUMERIC].describe(percentiles=[.01, .25, .5, .75, .99]).T.to_csv(path('artifacts/tables/numeric_summary.csv'))
    clean.nunique(dropna=False).rename('distinct_values').to_csv(path('artifacts/tables/feature_cardinality.csv'))
    write_json('artifacts/preprocessing_audit.json', {
        'fit_partition': 'development', 'dropped': cleaner.dropped_, 'retained_input_count': len(clean.columns),
        'missing_policy': 'Only ? is a missing token. None in A1Cresult/max_glu_serum remains test not performed.',
        'age_policy': 'Ten-year bands remain categorical.',
        'diagnoses': 'Fixed broad ICD-9 groups; no outcome-derived mapping.',
        'outlier_policy': 'Retain unusual valid encounters; no automatic row deletion.',
        'note': 'Each CV training fold learns its own exclusions, imputation, MI and encoding.'})

    fig, ax = plt.subplots(figsize=(8,4))
    top = missing.head(12).iloc[::-1]
    ax.barh(top.feature, top.missing_fraction*100, color='#26758c')
    ax.set(xlabel='Missing values (%)', title='Development data: missingness (? only)')
    save_figure(fig, 'missingness')

    fig, axes = plt.subplots(2, 4, figsize=(14,7))
    for ax, name in zip(axes.flat, NUMERIC):
        ax.hist(frame[name], bins=30, color='#26758c', edgecolor='white')
        ax.set(title=name, ylabel='Encounters', yscale='log')
    fig.suptitle('Development data: numerical distributions (log count axes)')
    fig.tight_layout()
    save_figure(fig, 'numeric_distributions')

    fig, axes = plt.subplots(2, 4, figsize=(14,7))
    for ax, name in zip(axes.flat, NUMERIC):
        ax.boxplot([frame.loc[y.eq(0), name], frame.loc[y.eq(1), name]], tick_labels=['No early', '<30'], whis=(5,95), showfliers=False)
        ax.set(title=name, ylabel='Value')
    fig.suptitle('Development data: numerical features by outcome (5th-95th percentile whiskers)')
    fig.tight_layout()
    save_figure(fig, 'numeric_by_outcome')

    correlation = frame[NUMERIC].corr(method='spearman')
    correlation.to_csv(path('artifacts/tables/numeric_correlations.csv'))
    fig, ax = plt.subplots(figsize=(9,7))
    im = ax.imshow(correlation, cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_xticks(range(len(NUMERIC)), NUMERIC, rotation=65, ha='right')
    ax.set_yticks(range(len(NUMERIC)), NUMERIC)
    ax.set_title('Development data: Spearman correlations')
    for i in range(len(NUMERIC)):
        for j in range(len(NUMERIC)):
            ax.text(j, i, f'{correlation.iloc[i,j]:.2f}', ha='center', va='center', fontsize=8,
                    color='white' if abs(correlation.iloc[i,j])>.6 else 'black')
    fig.colorbar(im, ax=ax)
    save_figure(fig, 'correlations')

    fig, axes = plt.subplots(1,3, figsize=(14,4))
    tables = []
    for ax, name in zip(axes, ['age', 'A1Cresult', 'diag_1']):
        temp = clean[[name]].assign(target=y.values)
        stats = temp.groupby(name, dropna=False)['target'].agg(['size','sum','mean']).reset_index()
        stats['feature'] = name
        tables.append(stats.rename(columns={name:'category'}))
        labels = [f'{"Missing" if pd.isna(value) else value} (n={int(n)})' for value,n in zip(stats[name],stats['size'])]
        ax.bar(labels, stats['mean']*100, color='#26758c')
        ax.tick_params(axis='x', labelrotation=70)
        ax.set(title=name, ylabel='Recorded early readmission (%)')
    fig.suptitle('Development data: descriptive associations, not causal effects')
    fig.tight_layout()
    save_figure(fig, 'categorical_outcomes')
    pd.concat(tables, ignore_index=True).to_csv(path('artifacts/tables/categorical_outcomes.csv'), index=False)
    fig, ax = plt.subplots(figsize=(5,4))
    counts = y.value_counts().reindex([0,1])
    bars = ax.bar(['No recorded early readmission','Recorded <30'], counts, color=['#26758c','#d58042'])
    ax.bar_label(bars, labels=[f'{v:,}\n({v/len(y):.1%})' for v in counts])
    ax.set_ylim(0, counts.max()*1.18)
    ax.set(title='Development outcome imbalance', ylabel='Encounters')
    save_figure(fig, 'class_balance')
    print(f'EDA complete: {len(frame):,} development encounters; excluded columns: {cleaner.dropped_}', flush=True)
