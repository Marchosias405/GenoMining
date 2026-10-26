# Section 3 requirements coverage

This map follows the user's screenshot of Section 3. It covers the requested EDA and preprocessing content; the screenshot does not specify a page limit, file naming convention, or upload format.

| Requirement | Implementation and saved evidence | Discussion location |
|---|---|---|
| Basic EDA of structure and distributions | `readmission/data.py`; `analysis_population.csv`, `split_sizes.csv`, the shared data dictionary, `numeric_summary.csv` | Notebook sections 1-3; reference pages 1-2 |
| Histograms and boxplots for key features | `02_distributions_boxplots.png`: four representative fields; complete statistics for all eight numeric fields | Notebook section 3; reference page 2 |
| Feature relationships and correlation heatmap | Spearman matrix/pair table, categorical outcome counts/rates, `03_correlations.png` | Notebook section 3; reference page 3 |
| Discuss insights and challenges | Measured imbalance, skew/zero concentration, missingness, moderate correlation, tiny categories and repeated patients | README results and examples; reference pages 1-3 and 6 |
| Handle missing values appropriately | Development-fitted >95% missing/constant exclusions, categorical Missing level, numeric median imputer; preserve literal None laboratory category | `column_decisions.csv`, `numeric_transform_audit.csv`, notebook section 5; reference page 4 |
| Handle outliers appropriately | IQR and multivariable Isolation Forest screens, profiles and flags; retain plausible unusual encounters, with zero rows removed | `outlier_iqr.csv`, `outlier_profiles.csv`, notebook section 4; reference page 4 |
| Normalize or standardize numerical inputs | StandardScaler after median imputation; mean/scale and post-transform mean/std recorded | `numeric_transform_audit.csv`, `transformation_examples.csv`, notebook section 5; reference page 4 |
| Encode categorical variables | One-hot encoding with training-learned rare-category pooling and unknown-category handling; treat numeric codes as categorical | `categorical_encoding_audit.csv`, `encoded_feature_names.csv`; notebook sections 5 and 7; reference page 4 |
| Independent feature selection | Mutual information filter on whole raw cleaned inputs; select 20 before one-hot expansion; no classifier importance used | `mutual_information.csv`, figure 4, notebook section 6; reference page 5 |
| Dimensionality reduction if dimensionality is high | PCA on 216 encoded inputs, retaining 65 components and 95.11% of variance; separate alternative to MI-selected inputs | `pca_variance.csv`, `pca_loadings.csv`, figure 4, notebook section 6; reference page 5 |
| Discuss missingness, outliers, normalization and augmentation methods | Explain actual actions and non-actions: no numeric missing cells to impute, no outlier deletion, no synthetic data, and training-only rules | Notebook sections 4-8; reference pages 4-6; README preprocessing example |
| Report methods and main results using only essential charts | Six-page reference report with four composite figures; detailed tables remain separate | `reports/section3_reference_report.pdf` and editable `.md`; team writes its final report |

## Decisions that need explanation rather than extra algorithms

- **Outlier handling can be retention.** The screenshot gives imputation/removal as examples; it does not require deleting observations. Zero-IQR visit counts and the detector profiles support retaining valid unusual encounters.
- **Augmentation is discussed but not applied.** The screenshot asks for a discussion of the method. There is no need to invent synthetic records to satisfy it. State clearly that no augmentation was used and why.
- **PCA is performed on the expanded representation.** This addresses actual encoding dimensionality. It is not claimed to improve prediction, and is not silently substituted into the original classifier experiments.
- **No highly correlated numeric pair is fabricated.** The observed maximum absolute Spearman correlation is 0.463; no pair exceeds 0.80. Discuss that result and the limits of pairwise correlation.
- **Train-only processing remains essential.** Patient grouping prevents overlapping patients; fold-local fitting would additionally prevent validation-label leakage when using MI in later cross-validation.

The reference contains measured evidence, not a promise of full marks. Before submitting, check the wider course handout for formatting and administrative requirements not visible in the screenshot.
