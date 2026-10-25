# File guide

Every portable file is listed below. Local environments, caches and Git internals are excluded because they are machine-specific. Source modules also have purpose docstrings. Generated outputs are evidence, not a written academic report.

| File | Purpose and why it is needed |
|---|---|
| `.gitignore` | Excludes environments, caches and large reproducible files from Git. |
| `artifacts/data_audit.json` | Machine-readable data audit metadata; records design, provenance or verification. |
| `artifacts/delivery_verification.json` | Machine-readable delivery verification metadata; records design, provenance or verification. |
| `artifacts/evaluation_design.json` | Machine-readable evaluation design metadata; records design, provenance or verification. |
| `artifacts/figures/calibration_comparison.png` | Generated calibration comparison figure; visual evidence for inspection and your report. |
| `artifacts/figures/categorical_outcomes.png` | Generated categorical outcomes figure; visual evidence for inspection and your report. |
| `artifacts/figures/class_balance.png` | Generated class balance figure; visual evidence for inspection and your report. |
| `artifacts/figures/clusters_pca.png` | Generated clusters pca figure; visual evidence for inspection and your report. |
| `artifacts/figures/confusion_matrices.png` | Generated confusion matrices figure; visual evidence for inspection and your report. |
| `artifacts/figures/correlations.png` | Generated correlations figure; visual evidence for inspection and your report. |
| `artifacts/figures/feature_selection_comparison.png` | Generated feature selection comparison figure; visual evidence for inspection and your report. |
| `artifacts/figures/missingness.png` | Generated missingness figure; visual evidence for inspection and your report. |
| `artifacts/figures/numeric_by_outcome.png` | Generated numeric by outcome figure; visual evidence for inspection and your report. |
| `artifacts/figures/numeric_distributions.png` | Generated numeric distributions figure; visual evidence for inspection and your report. |
| `artifacts/figures/outliers_pca.png` | Generated outliers pca figure; visual evidence for inspection and your report. |
| `artifacts/figures/roc_precision_recall.png` | Generated roc precision recall figure; visual evidence for inspection and your report. |
| `artifacts/integrity_tests.xml` | JUnit evidence from the 12 integrity tests covering patient isolation, target semantics and frozen calibration. |
| `artifacts/model_choice.json` | Machine-readable model choice metadata; records design, provenance or verification. |
| `artifacts/models/calibrated_index.json` | Machine-readable calibrated index metadata; records design, provenance or verification. |
| `artifacts/models/index.json` | Machine-readable index metadata; records design, provenance or verification. |
| `artifacts/models/logistic_regression__all__baseline.joblib` | Serialized fitted preprocessing/classifier (and calibration when named); reproduces saved predictions without retraining. |
| `artifacts/models/logistic_regression__all__calibrated.joblib` | Serialized fitted preprocessing/classifier (and calibration when named); reproduces saved predictions without retraining. |
| `artifacts/models/logistic_regression__all__tuned.joblib` | Serialized fitted preprocessing/classifier (and calibration when named); reproduces saved predictions without retraining. |
| `artifacts/models/logistic_regression__mi20__baseline.joblib` | Serialized fitted preprocessing/classifier (and calibration when named); reproduces saved predictions without retraining. |
| `artifacts/models/logistic_regression__mi20__calibrated.joblib` | Serialized fitted preprocessing/classifier (and calibration when named); reproduces saved predictions without retraining. |
| `artifacts/models/logistic_regression__mi20__tuned.joblib` | Serialized fitted preprocessing/classifier (and calibration when named); reproduces saved predictions without retraining. |
| `artifacts/models/random_forest__all__baseline.joblib` | Serialized fitted preprocessing/classifier (and calibration when named); reproduces saved predictions without retraining. |
| `artifacts/models/random_forest__all__calibrated.joblib` | Serialized fitted preprocessing/classifier (and calibration when named); reproduces saved predictions without retraining. |
| `artifacts/models/random_forest__all__tuned.joblib` | Serialized fitted preprocessing/classifier (and calibration when named); reproduces saved predictions without retraining. |
| `artifacts/models/random_forest__mi20__baseline.joblib` | Serialized fitted preprocessing/classifier (and calibration when named); reproduces saved predictions without retraining. |
| `artifacts/models/random_forest__mi20__calibrated.joblib` | Serialized fitted preprocessing/classifier (and calibration when named); reproduces saved predictions without retraining. |
| `artifacts/models/random_forest__mi20__tuned.joblib` | Serialized fitted preprocessing/classifier (and calibration when named); reproduces saved predictions without retraining. |
| `artifacts/models/xgboost__all__baseline.joblib` | Serialized fitted preprocessing/classifier (and calibration when named); reproduces saved predictions without retraining. |
| `artifacts/models/xgboost__all__calibrated.joblib` | Serialized fitted preprocessing/classifier (and calibration when named); reproduces saved predictions without retraining. |
| `artifacts/models/xgboost__all__tuned.joblib` | Serialized fitted preprocessing/classifier (and calibration when named); reproduces saved predictions without retraining. |
| `artifacts/models/xgboost__mi20__baseline.joblib` | Serialized fitted preprocessing/classifier (and calibration when named); reproduces saved predictions without retraining. |
| `artifacts/models/xgboost__mi20__calibrated.joblib` | Serialized fitted preprocessing/classifier (and calibration when named); reproduces saved predictions without retraining. |
| `artifacts/models/xgboost__mi20__tuned.joblib` | Serialized fitted preprocessing/classifier (and calibration when named); reproduces saved predictions without retraining. |
| `artifacts/predictions/example_predictions.csv` | Generated example predictions evidence table; supports reproducible comparisons or prediction auditing. |
| `artifacts/predictions/test_predictions.csv` | Generated test predictions evidence table; supports reproducible comparisons or prediction auditing. |
| `artifacts/preprocessing_audit.json` | Machine-readable preprocessing audit metadata; records design, provenance or verification. |
| `artifacts/status/data.json` | Completed phase status, source fingerprint, configuration, runtime and package versions; prevents stale reuse. |
| `artifacts/status/eda.json` | Completed phase status, source fingerprint, configuration, runtime and package versions; prevents stale reuse. |
| `artifacts/status/evaluation.json` | Completed phase status, source fingerprint, configuration, runtime and package versions; prevents stale reuse. |
| `artifacts/status/models.json` | Completed phase status, source fingerprint, configuration, runtime and package versions; prevents stale reuse. |
| `artifacts/status/unsupervised.json` | Completed phase status, source fingerprint, configuration, runtime and package versions; prevents stale reuse. |
| `artifacts/tables/calibration_bins.csv` | Generated calibration bins evidence table; supports reproducible comparisons or prediction auditing. |
| `artifacts/tables/categorical_outcomes.csv` | Generated categorical outcomes evidence table; supports reproducible comparisons or prediction auditing. |
| `artifacts/tables/cluster_assignments.csv` | Generated cluster assignments evidence table; supports reproducible comparisons or prediction auditing. |
| `artifacts/tables/cluster_profiles.csv` | Generated cluster profiles evidence table; supports reproducible comparisons or prediction auditing. |
| `artifacts/tables/clustering_metrics.csv` | Generated clustering metrics evidence table; supports reproducible comparisons or prediction auditing. |
| `artifacts/tables/cv_fold_scores.csv` | Generated cv fold scores evidence table; supports reproducible comparisons or prediction auditing. |
| `artifacts/tables/cv_split_audit.csv` | Generated cv split audit evidence table; supports reproducible comparisons or prediction auditing. |
| `artifacts/tables/data_dictionary.csv` | Generated data dictionary evidence table; supports reproducible comparisons or prediction auditing. |
| `artifacts/tables/feature_cardinality.csv` | Generated feature cardinality evidence table; supports reproducible comparisons or prediction auditing. |
| `artifacts/tables/feature_selection.csv` | Generated feature selection evidence table; supports reproducible comparisons or prediction auditing. |
| `artifacts/tables/missingness.csv` | Generated missingness evidence table; supports reproducible comparisons or prediction auditing. |
| `artifacts/tables/model_selection.csv` | Generated model selection evidence table; supports reproducible comparisons or prediction auditing. |
| `artifacts/tables/numeric_correlations.csv` | Generated numeric correlations evidence table; supports reproducible comparisons or prediction auditing. |
| `artifacts/tables/numeric_summary.csv` | Generated numeric summary evidence table; supports reproducible comparisons or prediction auditing. |
| `artifacts/tables/outlier_assignments.csv` | Generated outlier assignments evidence table; supports reproducible comparisons or prediction auditing. |
| `artifacts/tables/outlier_metrics.csv` | Generated outlier metrics evidence table; supports reproducible comparisons or prediction auditing. |
| `artifacts/tables/outlier_overlap.csv` | Generated outlier overlap evidence table; supports reproducible comparisons or prediction auditing. |
| `artifacts/tables/outlier_profiles.csv` | Generated outlier profiles evidence table; supports reproducible comparisons or prediction auditing. |
| `artifacts/tables/patient_bootstrap_intervals.csv` | Generated patient bootstrap intervals evidence table; supports reproducible comparisons or prediction auditing. |
| `artifacts/tables/patient_bootstrap_replicates.csv` | Generated patient bootstrap replicates evidence table; supports reproducible comparisons or prediction auditing. |
| `artifacts/tables/search_logistic_regression__all.csv` | Generated search logistic regression  all evidence table; supports reproducible comparisons or prediction auditing. |
| `artifacts/tables/search_logistic_regression__mi20.csv` | Generated search logistic regression  mi20 evidence table; supports reproducible comparisons or prediction auditing. |
| `artifacts/tables/search_random_forest__all.csv` | Generated search random forest  all evidence table; supports reproducible comparisons or prediction auditing. |
| `artifacts/tables/search_random_forest__mi20.csv` | Generated search random forest  mi20 evidence table; supports reproducible comparisons or prediction auditing. |
| `artifacts/tables/search_xgboost__all.csv` | Generated search xgboost  all evidence table; supports reproducible comparisons or prediction auditing. |
| `artifacts/tables/search_xgboost__mi20.csv` | Generated search xgboost  mi20 evidence table; supports reproducible comparisons or prediction auditing. |
| `artifacts/tables/split_summary.csv` | Generated split summary evidence table; supports reproducible comparisons or prediction auditing. |
| `artifacts/tables/subgroup_metrics.csv` | Generated subgroup metrics evidence table; supports reproducible comparisons or prediction auditing. |
| `artifacts/tables/test_metrics.csv` | Generated test metrics evidence table; supports reproducible comparisons or prediction auditing. |
| `artifacts/unsupervised_design.json` | Machine-readable unsupervised design metadata; records design, provenance or verification. |
| `config.json` | Fixed seeds, CV/search sizes, thresholds and sample sizes for reproducibility. |
| `data/processed/cohort.csv` | Eligible encounters with target and patient-disjoint split assignments. |
| `data/processed/splits.csv` | Explicit encounter-to-patient partition ledger for leakage auditing. |
| `data/raw/diabetic_data.csv` | Original encounter records; the immutable source for cohort construction. |
| `data/raw/IDS_mapping.csv` | Source meanings of admission/discharge/source codes; verifies cohort exclusions. |
| `data/raw/uci-diabetes.zip` | Original public UCI archive, retained for checksum/provenance and offline reproduction. |
| `docs/EXPERIMENT_DESIGN.md` | Technical choices, scope limits and source references supporting reproducibility. |
| `docs/TEAM_WORK.md` | Suggested review ownership and evidence checklist for your own report. |
| `examples/encounters.csv` | Three public feature-only rows demonstrating the CSV prediction interface. |
| `FILE_GUIDE.md` | This inventory explains the purpose of each portable project file. |
| `notebooks/01_data_preparation.ipynb` | Executed phase notebook with visible code, tables/figures and interpretation prompts; supports course submission. |
| `notebooks/02_exploratory_analysis.ipynb` | Executed phase notebook with visible code, tables/figures and interpretation prompts; supports course submission. |
| `notebooks/03_clustering_outliers.ipynb` | Executed phase notebook with visible code, tables/figures and interpretation prompts; supports course submission. |
| `notebooks/04_model_training.ipynb` | Executed phase notebook with visible code, tables/figures and interpretation prompts; supports course submission. |
| `notebooks/05_evaluation.ipynb` | Executed phase notebook with visible code, tables/figures and interpretation prompts; supports course submission. |
| `README.md` | Entry point: introduction, example use, setup, phases and folder structure. |
| `readme.me` | Requested-filename overview with a short introduction, usage example and folder structure. |
| `readmission/__init__.py` | Reproducible diabetes readmission experiments; shared code for CLI and notebooks. |
| `readmission/common.py` | Paths, configuration, and small output helpers shared across all experiment phases. |
| `readmission/data.py` | Download and validate UCI data, define the eligible cohort, and persist patient-disjoint splits. |
| `readmission/eda.py` | Development-only exploratory figures and tables, leaving held-out outcomes out of model design. |
| `readmission/evaluation.py` | Evaluate frozen models on held-out patients, calibrate independently, and export honest uncertainty estimates. |
| `readmission/features.py` | Leakage-safe clinical grouping, training-only column filtering, and raw-feature mutual information. |
| `readmission/models.py` | Patient-grouped baseline/tuning experiments and training-only feature-selection comparisons. |
| `readmission/pipeline.py` | Run resumable project phases and track source/configuration provenance for each completed phase. |
| `readmission/unsupervised.py` | Compare three clusterers and three outlier detectors using development data only. |
| `requirements-lock.txt` | Exact installed dependency versions for environment auditing. |
| `requirements.txt` | Pinned direct dependencies needed to reproduce the experiment. |
| `run.py` | Command-line entry point: reproduce phases or score a CSV using the development-selected calibrated model. |
| `scripts/build_notebooks.py` | Create five readable phase notebooks and optionally execute them with retained outputs in an isolated kernel. |
| `scripts/inventory.py` | Document every delivered file's purpose and package portable source, evidence, data and models without local environments. |
| `scripts/verify_delivery.py` | Check phase evidence, patient isolation, held-out metrics, serialized-model reproducibility and executed notebook outputs. |
| `tests/test_integrity.py` | Regression checks for target semantics, patient leakage, fold-local preprocessing, and frozen calibration. |
