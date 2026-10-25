# Team review and evidence checklist

This file proposes ownership for understanding, review and future work. It is not a record of who implemented the generated code. Each team member should verify their work, make their own commits, and accurately describe their contributions.

| Suggested reviewer | Primary classifier | Clustering | Outliers | Supporting responsibility |
|---|---|---|---|---|
| Member A | Logistic regression | MiniBatchKMeans | IsolationForest | Source audit, cohort, patient splits, data dictionary |
| Member B | Random forest | BIRCH | LocalOutlierFactor | EDA, missingness, diagnosis grouping, MI comparisons |
| Member C | XGBoost | GaussianMixture | EllipticEnvelope | Calibration, held-out metrics, reproducibility |

## Review in phases

1. **Data:** independently check the source counts, six excluded discharge codes, target mapping and no-overlap assertions in notebook 01.
2. **EDA:** inspect notebook 02 figures and explain meaningful missingness, imbalance, skew and categorical associations in your own words.
3. **Unsupervised:** inspect notebook 03 and compare algorithm assumptions, cluster profiles, outlier overlap, and why unusual encounters were retained.
4. **Models:** inspect notebook 04; understand how group folds and Pipeline prevent leakage; compare baseline/tuned AP, all/selected inputs, and actual runtime.
5. **Evaluation:** inspect notebook 05; distinguish discrimination, calibration and thresholded classification; interpret the bootstrap interval and limitations.

## Evidence for the report you will write

- Dataset/cohort facts: `artifacts/data_audit.json`, `tables/data_dictionary.csv`, `tables/split_summary.csv`.
- EDA: `tables/missingness.csv`, `numeric_summary.csv`, `numeric_correlations.csv`, `categorical_outcomes.csv` and their figures.
- Clusters: `tables/clustering_metrics.csv`, `cluster_profiles.csv`, `figures/clusters_pca.png`.
- Outliers: `tables/outlier_metrics.csv`, `outlier_profiles.csv`, `outlier_overlap.csv`, `figures/outliers_pca.png`.
- Feature selection: `tables/feature_selection.csv`, `model_selection.csv`, `figures/feature_selection_comparison.png`.
- Tuning: six `tables/search_*.csv` files, `cv_fold_scores.csv`, `model_selection.csv`.
- Final performance: `tables/test_metrics.csv`, ROC/PR, calibration and confusion-matrix figures.
- Uncertainty: `tables/patient_bootstrap_intervals.csv`, especially the paired Brier-change interval.
- Reproducibility: `config.json`, `requirements*.txt`, `artifacts/status/*.json`, `artifacts/delivery_verification.json`.

All table/figure paths above are relative to `artifacts/` unless explicitly prefixed. Use selected evidence that answers the course questions; do not paste every table. Explain results and limitations yourself, and keep the submitted notebooks' outputs visible. Verify the current course report/page-limit and access requirements before submission.
