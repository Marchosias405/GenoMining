"""Evaluate frozen models on held-out patients, calibrate independently, and export honest uncertainty estimates."""
from time import perf_counter
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, brier_score_loss, log_loss,
    confusion_matrix, roc_curve, precision_recall_curve, ConfusionMatrixDisplay)
from .common import config, path, read_json, write_json, save_figure
from .data import load_cohort, xy


def metrics(y, probability, threshold=.5):
    p = np.asarray(probability)
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError('Invalid model probabilities.')
    prediction = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y,prediction,labels=[0,1]).ravel()
    both = len(np.unique(y)) == 2
    return {'accuracy':accuracy_score(y,prediction), 'precision':precision_score(y,prediction,zero_division=0),
            'recall':recall_score(y,prediction,zero_division=0), 'f1':f1_score(y,prediction,zero_division=0),
            'roc_auc':roc_auc_score(y,p) if both else np.nan,
            'average_precision':average_precision_score(y,p) if both else np.nan,
            'brier_score':brier_score_loss(y,p), 'log_loss':log_loss(y,p,labels=[0,1]),
            'tn':int(tn),'fp':int(fp),'fn':int(fn),'tp':int(tp),'threshold':threshold}


def patient_bootstrap(y, p_raw, p_cal, patients, cfg):
    """Resample patients, retaining each patient's complete bundle of encounters through sample weights."""
    codes, uniques = pd.factorize(patients)
    rng = np.random.default_rng(cfg['seed']+20)
    records = []
    for repeat in range(cfg['bootstrap_repetitions']):
        multiplicities = np.bincount(rng.integers(0,len(uniques),size=len(uniques)),minlength=len(uniques))
        weights = multiplicities[codes]
        if len(np.unique(y[weights>0])) < 2:
            continue
        row = {'replicate':repeat}
        for label, p in [('uncalibrated',p_raw),('calibrated',p_cal)]:
            row[label+'_roc_auc'] = roc_auc_score(y,p,sample_weight=weights)
            row[label+'_average_precision'] = average_precision_score(y,p,sample_weight=weights)
            row[label+'_brier_score'] = np.average((p-y)**2,weights=weights)
        row['brier_change_calibrated_minus_uncalibrated'] = row['calibrated_brier_score']-row['uncalibrated_brier_score']
        records.append(row)
    table = pd.DataFrame(records)
    table.to_csv(path('artifacts/tables/patient_bootstrap_replicates.csv'),index=False)
    intervals = table.drop(columns='replicate').quantile([.025,.975]).T
    intervals.columns = ['lower_95','upper_95']
    intervals.index.name = 'metric'
    intervals.to_csv(path('artifacts/tables/patient_bootstrap_intervals.csv'))


def run():
    cfg = config()
    cal, test = load_cohort('calibration'), load_cohort('test')
    Xcal, ycal = xy(cal)
    Xtest, ytest = xy(test)
    models = read_json('artifacts/models/index.json')
    choice = read_json('artifacts/model_choice.json')['experiment']
    rows, probability_store, calibrated_files = [], {}, []
    predictions = test[['encounter_id','patient_nbr','target']].copy()
    for info in models:
        key, stage = info['experiment'], info['stage']
        model = joblib.load(path(info['file']))
        start = perf_counter()
        raw = np.asarray(model.predict_proba(Xtest)[:,1],dtype=np.float64)
        seconds = perf_counter()-start
        row = {'experiment':key,'stage':stage,'calibration':'none',**metrics(ytest,raw,cfg['threshold']),
               'training_seconds':info['fit_seconds'],'predict_seconds':seconds,'calibration_seconds':0.0}
        rows.append(row)
        predictions[key+'__'+stage+'__raw'] = raw
        if stage == 'tuned':
            # FrozenEstimator prevents refitting the base model on calibration data.
            calibrated = CalibratedClassifierCV(FrozenEstimator(model), method='sigmoid')
            start = perf_counter()
            calibrated.fit(Xcal,ycal)
            fit_seconds = perf_counter()-start
            start = perf_counter()
            probability = calibrated.predict_proba(Xtest)[:,1]
            seconds = perf_counter()-start
            rows.append({'experiment':key,'stage':stage,'calibration':'sigmoid',**metrics(ytest,probability,cfg['threshold']),
                         'training_seconds':info['fit_seconds'],'predict_seconds':seconds,'calibration_seconds':fit_seconds})
            predictions[key+'__'+stage+'__sigmoid'] = probability
            probability_store[key] = (raw,probability)
            target = f'artifacts/models/{key}__calibrated.joblib'
            joblib.dump(calibrated,path(target),compress=3)
            calibrated_files.append({'experiment':key,'file':target})
    dev = load_cohort('development')
    for name, p in [('always_negative',0.0),('training_prevalence',float(dev['target'].mean()))]:
        rows.append({'experiment':name,'stage':'dummy','calibration':'none',**metrics(ytest,np.full(len(ytest),p),cfg['threshold']),
                     'training_seconds':0.0,'predict_seconds':0.0,'calibration_seconds':0.0})
    results = pd.DataFrame(rows)
    results.to_csv(path('artifacts/tables/test_metrics.csv'),index=False)
    predictions.to_csv(path('artifacts/predictions/test_predictions.csv'),index=False)
    write_json('artifacts/models/calibrated_index.json',calibrated_files)

    fig, axes = plt.subplots(1,2,figsize=(12,5))
    for key, (raw, _) in probability_store.items():
        fpr,tpr,_ = roc_curve(ytest,raw)
        precision,recall,_ = precision_recall_curve(ytest,raw)
        label = key.replace('__',' / ').replace('_',' ')
        axes[0].plot(fpr,tpr,label=label)
        axes[1].plot(recall,precision,label=label)
    axes[0].plot([0,1],[0,1],'k--',linewidth=1)
    axes[1].axhline(ytest.mean(),color='black',linestyle='--',linewidth=1,label='Test prevalence')
    axes[0].set(title='Tuned models: held-out ROC',xlabel='False positive rate',ylabel='True positive rate')
    axes[1].set(title='Tuned models: held-out precision-recall',xlabel='Recall',ylabel='Precision')
    for ax in axes:
        ax.legend(fontsize=7)
        ax.set_xlim(0,1)
        ax.set_ylim(0,1)
    fig.tight_layout()
    save_figure(fig,'roc_precision_recall')

    fig, axes = plt.subplots(2,3,figsize=(14,9))
    calibration_rows = []
    for ax,(key,(raw,calibrated)) in zip(axes.flat,probability_store.items()):
        for name,p in [('Uncalibrated',raw),('Sigmoid',calibrated)]:
            observed,predicted = calibration_curve(ytest,p,n_bins=10,strategy='quantile')
            ax.plot(predicted,observed,'o-',markersize=4,label=name)
            calibration_rows.extend({'experiment':key,'calibration':name,'mean_prediction':float(a),'observed_rate':float(b)} for a,b in zip(predicted,observed))
        ax.plot([0,1],[0,1],'k--',linewidth=1)
        ax.set(title=key.replace('__',' / '),xlabel='Mean predicted probability',ylabel='Observed early readmission rate',xlim=(0,1),ylim=(0,1))
        ax.legend(fontsize=8)
    fig.suptitle('Reliability diagrams: quantile bins on held-out patients')
    fig.tight_layout()
    save_figure(fig,'calibration_comparison')
    pd.DataFrame(calibration_rows).to_csv(path('artifacts/tables/calibration_bins.csv'),index=False)

    fig, axes = plt.subplots(2,3,figsize=(12,8))
    for ax,(key,(_,probability)) in zip(axes.flat,probability_store.items()):
        ConfusionMatrixDisplay.from_predictions(ytest,probability>=cfg['threshold'],labels=[0,1],
            display_labels=['No early','<30'],ax=ax,colorbar=False,cmap='Blues',values_format='d')
        ax.set_title(key.replace('__',' / '),fontsize=10)
    fig.suptitle(f'Calibrated models: fixed threshold {cfg["threshold"]}; encounter counts')
    fig.tight_layout()
    save_figure(fig,'confusion_matrices')

    comparison = results.query("stage == 'tuned' and calibration == 'none'").copy()
    fig, axes = plt.subplots(1,3,figsize=(14,4))
    for ax,metric in zip(axes,['average_precision','brier_score','training_seconds']):
        pivot = comparison.assign(family=comparison.experiment.str.split('__').str[0],features=comparison.experiment.str.split('__').str[1]).pivot(index='family',columns='features',values=metric)
        pivot.plot.bar(ax=ax,rot=15,color=['#26758c','#d58042'])
        ax.set(title=metric.replace('_',' '),xlabel='')
    fig.suptitle('Feature selection comparison: tuned, uncalibrated models')
    fig.tight_layout()
    save_figure(fig,'feature_selection_comparison')

    raw,calibrated = probability_store[choice]
    patient_bootstrap(ytest.to_numpy(),raw,calibrated,test['patient_nbr'],cfg)
    subgroups = []
    for feature in ['age','race','gender']:
        for value,group in test.groupby(feature,dropna=False):
            idx = group.index.to_numpy()
            row = {'feature':feature,'group':str(value),'encounters':len(idx),'patients':int(test.loc[idx,'patient_nbr'].nunique()),
                   'positive_rate':float(ytest.iloc[idx].mean()),'small_group':len(idx)<100}
            if len(idx)>=100:
                row.update(metrics(ytest.iloc[idx],calibrated[idx],cfg['threshold']))
            subgroups.append(row)
    pd.DataFrame(subgroups).to_csv(path('artifacts/tables/subgroup_metrics.csv'),index=False)
    example = Xtest.head(3)
    example.to_csv(path('examples/encounters.csv'),index=False,na_rep='?')
    write_json('artifacts/evaluation_design.json',{
        'selected_experiment':choice,'selection_used_test':False,'calibration_method':'sigmoid on independent calibration patients',
        'base_models_refit_during_calibration':False,'threshold':cfg['threshold'],
        'test_encounters':len(test),'test_patients':int(test['patient_nbr'].nunique()),
        'bootstrap':'200 patient-cluster bootstrap replicates; percentile intervals conditional on fixed fitted models; not training uncertainty.',
        'subgroups':'Descriptive exploration only; groups below 100 encounters have metrics suppressed; not proof of fairness.',
        'limitations':'Historical US data, incomplete capture of readmission, no released hospital IDs or encounter dates; internal validation only.',
        'brier_note':'Brier score combines discrimination and calibration; use reliability diagrams alongside it.'})
    print(results[['experiment','stage','calibration','roc_auc','average_precision','brier_score']].to_string(index=False),flush=True)
