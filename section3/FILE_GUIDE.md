# Section 3 file guide

Every file below is included in the focused ZIP. Large data, matrices and fitted transformers are generated locally and omitted from Git. This package has no classifier outputs. Read the root README for the workflow connecting the files.

| File | Purpose and why it is included |
|---|---|
| `.gitattributes` | Keeps source text in LF format and PDFs/images binary, preventing checkout corruption and hash drift. |
| `.gitignore` | Keeps caches, environments and large reproducible artifacts out of Git. |
| `artifacts/data_audit.json` | Source hashes, eligibility counts, exclusions and patient-separation evidence. |
| `artifacts/status/data.json` | Completed data-preparation status and fingerprint for reuse. |
| `artifacts/tables/data_dictionary.csv` | Original column definitions/types/roles, including identifiers and target. |
| `artifacts/tables/split_summary.csv` | Original partition counts/rates; Section 3 uses development only. |
| `config.json` | Original seed and settings, preserving the patient split and CPU limits. |
| `data/processed/cohort.csv` | Eligible encounters with binary target and patient-disjoint partition assignment. |
| `data/processed/splits.csv` | Encounter and patient partition ledger for checking overlap. |
| `data/raw/diabetic_data.csv` | Public encounter source; parsed with ? missing and literal None preserved. |
| `data/raw/IDS_mapping.csv` | Original code meanings used to verify death/hospice exclusions. |
| `data/raw/uci-diabetes.zip` | Original public UCI archive for provenance and offline reproduction. |
| `docs/FULL_PROJECT_GUIDE.md` | Original project instructions for context; later classifiers are outside this ZIP. |
| `FILE_GUIDE.md` | Branch-level pointer to this focused inventory and the original project guide. |
| `README.md` | Main walkthrough: commands, file relationships, measured results and explanations. |
| `readme.me` | Short overview preserving the requested filename and pointing to the complete README. |
| `readmission/__init__.py` | Reproducible diabetes readmission experiments; shared code for CLI and notebooks. |
| `readmission/common.py` | Paths, configuration, and small output helpers shared across all experiment phases. |
| `readmission/data.py` | Download and validate UCI data, define the eligible cohort, and persist patient-disjoint splits. |
| `readmission/features.py` | Leakage-safe clinical grouping, training-only column filtering, and raw-feature mutual information. |
| `readmission/pipeline.py` | Run resumable project phases and track source/configuration provenance for each completed phase. |
| `requirements-lock.txt` | Original environment record for auditing package versions. |
| `requirements.txt` | Pinned direct packages for analysis, notebook execution and testing. |
| `section3/__init__.py` | Standalone Section 3 analysis and submission helpers; no classifier training is required. |
| `section3/analysis.py` | Produce development-only EDA, auditable preprocessing, outlier screening, MI selection and PCA evidence. |
| `section3/build_notebook.py` | Generate the focused Section 3 teaching notebook and optionally execute it with retained outputs. |
| `section3/build_reference_report.py` | Build the Section 3 reference report as editable Markdown and a six-page PDF using measured evidence. |
| `section3/config.json` | Focused choices for MI size, anomaly screening and PCA variance retention. |
| `section3/FILE_GUIDE.md` | This per-file inventory, generated from the exact package contents. |
| `section3/outputs/data/development_pca.npz` | Development PCA scores under key scores; rows match the ledger. |
| `section3/outputs/data/development_row_index.csv` | Matrix-row ledger linking encounter/patient IDs and labels without using IDs as inputs. |
| `section3/outputs/data/development_top20.npz` | Compressed sparse 59,670 by 147 matrix; rows match the ledger. |
| `section3/outputs/data/outlier_flags.csv` | Encounter-level review flags; no observations are deleted. |
| `section3/outputs/figures/01_population_missingness.png` | Figure 1: class counts/proportions and true missingness. |
| `section3/outputs/figures/02_distributions_boxplots.png` | Figure 2: key histograms and outcome-wise boxplots with explicit whisker rules. |
| `section3/outputs/figures/03_correlations.png` | Figure 3: numeric Spearman heatmap with coefficients. |
| `section3/outputs/figures/04_selection_pca.png` | Figure 4: raw-input MI ranking and PCA cumulative variance. |
| `section3/outputs/manifest.json` | Source fingerprint and output hashes that detect stale or corrupted artifacts. |
| `section3/outputs/summary.json` | Measured counts, dimensions, findings and scope consumed by the notebook/report. |
| `section3/outputs/tables/analysis_population.csv` | Development counts and positive rate; fixes the EDA denominator. |
| `section3/outputs/tables/categorical_associations.csv` | Counts and descriptive outcome rates by age, HbA1c category and discharge code. |
| `section3/outputs/tables/categorical_encoding_audit.csv` | Category vocabularies, rare-category pooling and encoding widths. |
| `section3/outputs/tables/column_decisions.csv` | Every retained/dropped input and its measured reason. |
| `section3/outputs/tables/correlation_pairs.csv` | Unique pairs ordered by absolute correlation for redundancy checks. |
| `section3/outputs/tables/encoded_feature_names.csv` | Matrix column index to encoded feature-name mapping. |
| `section3/outputs/tables/laboratory_categories.csv` | Laboratory category counts explaining why None is retained. |
| `section3/outputs/tables/missingness.csv` | True missing-value counts and percentages for all original candidate inputs. |
| `section3/outputs/tables/mutual_information.csv` | All 42 MI ranks/scores and the top-20 selection before one-hot expansion. |
| `section3/outputs/tables/numeric_summary.csv` | All eight numeric distributions: quantiles, range, zeros and skewness. |
| `section3/outputs/tables/numeric_transform_audit.csv` | Learned imputation/scaling parameters and post-transform means/stds. |
| `section3/outputs/tables/outlier_iqr.csv` | Quartiles, fences, zero-IQR indicators and flagged counts. |
| `section3/outputs/tables/outlier_profiles.csv` | Flagged/unflagged Isolation Forest feature medians with sample sizes. |
| `section3/outputs/tables/pca_loadings.csv` | First two component weights for each encoded input. |
| `section3/outputs/tables/pca_variance.csv` | Variance per component and cumulative variance, checking the 95% target. |
| `section3/outputs/tables/spearman_correlations.csv` | Full numeric rank-correlation matrix used in the heatmap. |
| `section3/outputs/tables/split_sizes.csv` | Patient/encounter counts for reserved and development partitions. |
| `section3/outputs/tables/transformation_examples.csv` | Three public encounters traced through numeric transformation. |
| `section3/outputs/test_results.xml` | JUnit evidence from the six focused integrity tests. |
| `section3/outputs/transformers/pca.joblib` | Fitted centered PCA, applied after the all-input preprocessor. |
| `section3/outputs/transformers/preprocessor_all.joblib` | Fitted all-input preprocessing used before the PCA representation. |
| `section3/outputs/transformers/preprocessor_top20.joblib` | Fitted cleaner, MI selector and encoder for the selected-input representation. |
| `section3/outputs/verification.json` | Recorded artifact, notebook, row-alignment and test-evidence checks. |
| `section3/package.py` | Describe every focused deliverable and build a portable Section 3 ZIP without unrelated model experiments. |
| `section3/reports/section3_reference_report.md` | Editable narrative matching the reference PDF, with relative figure links. |
| `section3/reports/section3_reference_report.pdf` | Six-page methods/results reference containing four essential figures. |
| `section3/requirements-report.txt` | Optional PDF rebuilding dependency, separate from analysis installation. |
| `section3/REQUIREMENTS_COVERAGE.md` | Maps each screenshot requirement to code, evidence and discussion. |
| `section3/run.py` | Run or reuse the Section 3 analysis, recording source fingerprints and hashes for every generated output. |
| `section3/section3_eda_preprocessing.ipynb` | Executed teaching notebook with real outputs and transformation checks. |
| `section3/STUDY_GUIDE.md` | Exam-style questions and worked explanations of methods and leakage risks. |
| `section3/SUBMISSION_CHECKLIST.md` | Submission contents, verification commands and three-person review suggestions. |
| `section3/verify.py` | Verify focused artifacts, patient and row alignment, notebook outputs, tests and report evidence before packaging. |
| `tests/test_section3.py` | Check Section 3 leakage boundaries, missingness, outlier interpretation and reproducible matrix/PCA exports. |
