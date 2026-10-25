# Experiment design and implementation decisions

This technical specification connects the submitted proposal to executable experiments. It is not the team's academic report.

## Proposal questions and corresponding experiments

1. **Compare unseen-patient performance:** logistic regression, random forest and XGBoost use identical development folds and the same final held-out patients.
2. **Assess calibration:** each tuned model is frozen; sigmoid calibration is fitted on independent calibration patients; raw/calibrated probabilities are evaluated together.
3. **Assess feature selection:** compare all retained candidate inputs against the top 20 inputs ranked by mutual information. Record the number of raw and encoded inputs, development-CV average precision, test metrics, fresh final-fit time and prediction time.

## Cohort and data interpretation

The source contains 101,766 inpatient encounters. Apply exclusions to individual encounters, not all records of a patient. Discharge disposition codes 11/19/20/21 represent death and 13/14 represent hospice. The retained cohort is 99,343 encounters / 69,990 patients. There are 11,314 positive encounters. The proposal's phrase '11.4% of patients' is implemented correctly as '11.4% of encounters'.

The existing `<30` category defines the positive class; no exact 30-day outcome is reconstructed. `NO` means no recorded readmission, not proof of no readmission anywhere. The prediction is at discharge; length of stay and completed treatment information are allowed. The project makes no claim of an admission-time model or of preventing readmissions.

The unit of prediction/evaluation is an encounter, so patients with multiple encounters contribute more predictions. Patient separation prevents cross-split leakage but does not remove within-patient correlation. Bootstrap uncertainty resamples complete patients to reflect that correlation.

## Splits, tuning and final selection

- Seed 459. A five-fold StratifiedGroupKFold assignment creates three development folds, one calibration fold and one test fold. Fractions are approximate because patients are indivisible groups.
- A separate five-fold StratifiedGroupKFold within development is shared across all classifier comparisons. Each row of `cv_split_audit.csv` verifies no patient overlap.
- Four ParameterSampler configurations per experiment are evaluated with RandomizedSearchCV. An explicit untuned baseline is also evaluated. For each classifier/input set, retain the baseline if none of the sampled configurations has better mean development AP.
- The best mean development-CV AP across the six resulting experiments determines `model_choice.json`, before final test evaluation. CV scores used for selection may be optimistic; test scores are the final held-out assessment.
- No oversampling or SMOTE. Logistic regression and random forest may use balanced class weights as a training-only tuned option. XGBoost uses unweighted logistic loss. This is a deliberate bounded search, not an exhaustive comparison of imbalance techniques.
- The classification threshold is fixed at 0.5 before seeing test results. An operational hospital threshold would require a separately agreed recall/cost/resource objective and further validation.

Timing detail: `cv_fold_scores.csv` stores measured per-fold fit time for baseline rows; for selected rows its `fit_seconds` value is the selected candidate's mean CV fit time repeated across folds (RandomizedSearchCV does not expose individual fit times). Use the fresh, uncached `fit_seconds` in `model_selection.csv` and `training_seconds`/`predict_seconds` in `test_metrics.csv` for the feature-selection runtime comparison. Timings depend on hardware and concurrent system load.

## Preprocessing

All learned preprocessing is inside sklearn Pipelines, refitted in each development training fold. Numeric inputs are the eight integer counts/durations. Median imputation and scaling are applied there. Age bands and admission/discharge/source codes are categorical even when their raw representation is numeric.

Only `?` is parsed as missing. Literal `None` for A1Cresult/max_glu_serum means not tested, and medication `No` means not prescribed; these are preserved. Categories are imputed with `Missing`; one-hot encoding learns rare-category pooling (minimum training frequency 25), with unseen categories handled safely.

Columns over 95% missing or constant after fixed diagnosis grouping are excluded based only on the training fold. Weight and the constant medications are expected exclusions, but the audit records the actual decisions. Identifiers and the target are excluded from model inputs; patient IDs remain available for splitting and auditing.

Diagnosis codes are grouped with fixed rules: diabetes 250.xx; circulatory 390-459 or 785; respiratory 460-519 or 786; digestive 520-579 or 787; genitourinary 580-629 or 788; musculoskeletal 710-739; injury 800-999; neoplasms 140-239; other codes (including V/E) map to Other. Missing remains missing. This controls sparse diagnosis cardinality, at a cost of losing fine detail.

Mutual information is calculated on **whole candidate features before one-hot encoding**. Categorical variables and integer-valued counts are treated as discrete. Missing is represented by a distinct code for MI. Top-20 selection is learned from training labels only; validation/test labels are never used. MI is univariate and may miss useful interactions or favor high-cardinality features. It is an independent filter, not a classifier's embedded importance.

The all-input pipeline also calculates MI scores for the diagnostic feature table but retains every eligible input. Thus reported final-fit times include preprocessing and diagnostic MI computation for both variants; they are complete pipeline timings, not isolated classifier-fit timings.

## Clustering and outlier analysis

Use development data only. Transform the eight nonnegative numerical inputs with log1p, median imputation and standardization. Categorical attributes are intentionally excluded from distance/covariance calculations; clusters summarize utilization patterns, not comprehensive clinical phenotypes.

Compare MiniBatchKMeans, BIRCH and diagonal-covariance GaussianMixture at k=3,5,8. Within each method, choose k by sampled silhouette (2,000 fixed-seed encounters). Also report full-data Calinski-Harabasz, Davies-Bouldin and training time. PCA provides a 2D projection; the algorithms operate on all eight transformed inputs. Readmission rates are computed after clustering for description and never select k.

Compare IsolationForest, LocalOutlierFactor and EllipticEnvelope on the same fixed-seed 12,000-encounter development sample. Set a 3% target flag fraction. EllipticEnvelope uses five PCA components to avoid singular covariance and makes an approximate elliptical-distribution assumption. LOF is used transductively on the sample, not to predict new clinical cases. Inspect flagged/unflagged feature medians, prevalence and pairwise Jaccard overlaps. Rates/counts differ slightly from 3% if scores tie.

**Decision:** retain every eligible encounter in classifier training. High utilization may be genuine clinically relevant variation. This project does not have adjudicated data-error labels to justify automatic deletion. Training/test sets are not filtered using anomaly scores.

## Evaluation and interpretation

Report accuracy, precision, recall, F1, ROC-AUC, average precision, Brier score, log loss, TN/FP/FN/TP, ROC curves, precision-recall curves and reliability diagrams. Add always-negative and training-prevalence baselines. AP uses sklearn's average_precision_score (not trapezoidal PR-AUC).

Sigmoid calibration uses CalibratedClassifierCV with FrozenEstimator, preserving the development-trained base model. Brier score reflects both discrimination and calibration; it is interpreted with quantile-bin reliability plots. Calibration is not guaranteed to improve Brier score or class decisions.

For the development-selected experiment, obtain 95% percentile intervals from 200 patient-bootstrap replicates, including a paired calibrated-minus-uncalibrated Brier difference. These intervals are conditional on fitted models and do not include training/model-selection uncertainty. No repeated hypothesis tests are claimed across all models.

Age/race/gender subgroup metrics are descriptive checks on the final selected calibrated model. Suppress metrics below 100 encounters. Small positive counts, correlated encounters and historical coding remain limitations; these tables are not a clinical fairness certification.

## Scope limits

This is retrospective internal validation. The release lacks usable dates and hospital identifiers, preventing valid temporal and hospital holdouts. It cannot establish current Canadian/Fraser Health performance, intervention effects, clinical utility, or causal explanations. The code does not reproduce DERRI or Hai's longitudinal model because their inputs/design differ. Exact candidate searches, warnings, runtimes, package versions and data hashes are saved for inspection.

## Source references

- UCI dataset: https://doi.org/10.24432/C5230J
- Strack et al. (2014): https://doi.org/10.1155/2014/781670
- Rubin et al. (2016), DERRI development: https://doi.org/10.4158/E161391.OR
- Rubin et al. (2018), external validation: https://doi.org/10.4158/EP-2018-0035
- Hai et al., AMIA proceedings: https://pubmed.ncbi.nlm.nih.gov/37128461/
- Van Calster et al. (2019), calibration: https://doi.org/10.1186/s12916-019-1466-7
- sklearn grouped splitting: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html
- sklearn calibration: https://scikit-learn.org/stable/modules/generated/sklearn.calibration.CalibratedClassifierCV.html
- sklearn mutual information: https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.mutual_info_classif.html
- XGBoost parameters: https://xgboost.readthedocs.io/en/stable/parameter.html
