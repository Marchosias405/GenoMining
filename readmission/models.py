"""Patient-grouped baseline/tuning experiments and training-only feature-selection comparisons."""
from time import perf_counter
import json
import warnings
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, cross_validate, RandomizedSearchCV
from xgboost import XGBClassifier
from .common import config, path, write_json
from .data import load_cohort, xy
from .features import make_pipeline


def candidates(cfg):
    seed, threads = cfg['seed'], cfg['threads']
    return {
        'logistic_regression': (LogisticRegression(solver='liblinear', max_iter=1500, random_state=seed),
            {'model__C':[.01,.1,1,10], 'model__class_weight':[None,'balanced']}),
        'random_forest': (RandomForestClassifier(n_estimators=120, max_depth=12, min_samples_leaf=10, n_jobs=threads, random_state=seed),
            {'model__max_depth':[8,12,18], 'model__min_samples_leaf':[5,15,30], 'model__class_weight':[None,'balanced']}),
        'xgboost': (XGBClassifier(n_estimators=180, max_depth=3, learning_rate=.05, tree_method='hist', eval_metric='logloss', n_jobs=threads, random_state=seed),
            {'model__max_depth':[2,3,5], 'model__learning_rate':[.03,.08], 'model__min_child_weight':[1,10], 'model__subsample':[.8,1.0]})}


def run():
    cfg = config()
    frame = load_cohort('development')
    X, y = xy(frame)
    groups = frame['patient_nbr'].to_numpy()
    cv = list(StratifiedGroupKFold(n_splits=cfg['cv_folds'], shuffle=True, random_state=cfg['seed']+1).split(X,y,groups))
    cv_audit = []
    for fold, (train, valid) in enumerate(cv):
        overlap = len(set(groups[train]) & set(groups[valid]))
        if overlap:
            raise AssertionError('Patient overlap inside cross-validation.')
        cv_audit.append({'fold':fold, 'training_encounters':len(train), 'validation_encounters':len(valid),
                         'training_positive_rate':float(y.iloc[train].mean()), 'validation_positive_rate':float(y.iloc[valid].mean()),
                         'patient_overlap':overlap})
    pd.DataFrame(cv_audit).to_csv(path('artifacts/tables/cv_split_audit.csv'), index=False)
    memory = joblib.Memory(path('.cache/preprocessing'), verbose=0)
    summaries, fold_rows, selections, models_index = [], [], [], []
    for family, (classifier, distribution) in candidates(cfg).items():
        for feature_set, k in [('all',None),('mi20',cfg['selected_features'])]:
            key = family+'__'+feature_set
            print(f'Training {key}: baseline + {cfg["search_iterations"]} randomized candidates x {len(cv)} folds', flush=True)
            pipeline = make_pipeline(clone(classifier), k, cfg['seed'], memory)
            start = perf_counter()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                baseline_cv = cross_validate(pipeline, X,y, cv=cv, scoring={'ap':'average_precision','roc_auc':'roc_auc'}, n_jobs=1, error_score='raise')
                search = RandomizedSearchCV(pipeline, distribution, n_iter=cfg['search_iterations'],
                    scoring='average_precision', cv=cv, n_jobs=1, random_state=cfg['seed'], refit=False, error_score='raise', return_train_score=False)
                search.fit(X,y)
            search_seconds = perf_counter()-start
            baseline_mean = float(np.mean(baseline_cv['test_ap']))
            best_params = search.best_params_ if search.best_score_ > baseline_mean else {}
            winner_score = float(max(search.best_score_,baseline_mean))
            cv_results = pd.DataFrame(search.cv_results_)
            cv_results.to_csv(path(f'artifacts/tables/search_{key}.csv'), index=False)
            for fold in range(len(cv)):
                fold_rows.append({'experiment':key, 'fold':fold, 'stage':'baseline', 'average_precision':float(baseline_cv['test_ap'][fold]),
                                  'fit_seconds':float(baseline_cv['fit_time'][fold])})
                selected_ap = float(cv_results.iloc[search.best_index_][f'split{fold}_test_score']) if best_params else float(baseline_cv['test_ap'][fold])
                fold_rows.append({'experiment':key, 'fold':fold, 'stage':'selected', 'average_precision':selected_ap,
                                  'fit_seconds':float(cv_results.iloc[search.best_index_]['mean_fit_time']) if best_params else float(baseline_cv['fit_time'][fold])})
            for stage, params in [('baseline',{}),('tuned',best_params)]:
                fitted = clone(pipeline).set_params(**params)
                # A fresh, uncached final fit makes before/after feature-selection runtime comparable.
                fitted.memory = None
                start = perf_counter()
                fitted.fit(X,y)
                fit_seconds = perf_counter()-start
                filename = f'artifacts/models/{key}__{stage}.joblib'
                joblib.dump(fitted, path(filename), compress=3)
                models_index.append({'experiment':key,'family':family,'feature_set':feature_set,'stage':stage,
                                     'file':filename,'fit_seconds':fit_seconds,'cv_ap':baseline_mean if stage=='baseline' else winner_score})
                if stage == 'tuned':
                    selector = fitted.named_steps['select']
                    for name, score in zip(selector.feature_names_in_,selector.scores_):
                        selections.append({'experiment':key,'feature':name,'mutual_information':float(score),'selected':name in selector.columns_})
                    summaries.append({'experiment':key,'family':family,'feature_set':feature_set,
                        'baseline_cv_ap':baseline_mean,'selected_cv_ap':winner_score,
                        'sampled_best_cv_ap':float(search.best_score_), 'cv_ap_gain':winner_score-baseline_mean,
                        'selected_candidate':'random_search' if best_params else 'baseline',
                        'raw_inputs_retained':len(selector.columns_),
                        'encoded_inputs':len(fitted.named_steps['encode'].get_feature_names_out()),
                        'fit_seconds':fit_seconds, 'search_seconds':search_seconds,
                        'best_params':json.dumps(best_params), 'warnings':' | '.join(sorted(set(str(w.message) for w in caught)))})
            pd.DataFrame(summaries).to_csv(path('artifacts/tables/model_selection.csv'),index=False)
            print(f'{key}: CV average precision {winner_score:.4f}; final fit {fit_seconds:.1f}s',flush=True)
    summary = pd.DataFrame(summaries)
    best = summary.sort_values(['selected_cv_ap','experiment'],ascending=[False,True]).iloc[0]
    write_json('artifacts/models/index.json',models_index)
    write_json('artifacts/model_choice.json', {'experiment':best['experiment'], 'selection_metric':'development grouped-CV average precision',
        'cv_ap':float(best['selected_cv_ap']), 'test_used_for_selection':False,
        'rule':'Best of baseline and randomized candidates for each experiment; highest development CV AP selects final experiment.'})
    pd.DataFrame(fold_rows).to_csv(path('artifacts/tables/cv_fold_scores.csv'),index=False)
    pd.DataFrame(selections).to_csv(path('artifacts/tables/feature_selection.csv'),index=False)
