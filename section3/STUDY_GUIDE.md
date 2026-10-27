# Section 3 study guide

Use these questions after reading the README and executing the notebook. Try answering aloud first, then compare with the explanation. The numbers below come from this project's development data.

## What is one observation and what is the target

One row is one hospital encounter. A patient can have multiple encounters. The binary target is 1 for the source category `<30`, and 0 for `>30` or `NO`. It measures recorded early readmission in this dataset. It is not a freshly calculated 30-day follow-up event, nor proof that `NO` patients were never readmitted elsewhere.

There are 101,766 source encounters; eligibility exclusions leave 99,343 encounters from 69,990 patients. EDA uses the 59,670 development encounters only. Distinguish these denominators when explaining percentages.

## Why split by patient before exploring or fitting transforms

If encounters from the same person occur in both training and evaluation, the evaluation can benefit from repeated patient characteristics. Grouped splitting prevents that overlap. Reserving calibration/test patients also keeps exploratory decisions and learned transformations from adapting to their distributions.

Splitting by patient does not make all encounters independent within a partition. A person with many admissions still contributes multiple observations. These are descriptive encounter-level statistics.

## What does fit learn and what does transform do

`fit` learns the information needed to apply a rule: column exclusions, medians, means, scales, category vocabularies, MI ranks or PCA loadings. `transform` applies those learned values to a supplied table without relearning them. `fit_transform` combines the two for training data.

In this code, `preprocessing(k=20)` returns a sklearn Pipeline. Its stages are `clean`, `select`, and `encode`. `encode` is a ColumnTransformer with separate numeric and categorical pipelines. The saved joblib file preserves all these learned decisions.

## Why is None different from missing

`?` indicates a missing value. In the two laboratory fields, the text `None` means a test was not performed. It is a meaningful category. Converting that category into NaN would misrepresent the data and could cause an entire field to appear mostly missing. Medication `No` likewise means not prescribed.

Pandas' default CSV parsing can interpret the text `None` as missing. The project loads source data with `keep_default_na=False, na_values=['?']`. Read generated categorical evidence tables with `keep_default_na=False` too.

## Why drop weight but keep specialty and payer

The cleaner removes a variable when more than 95% of training values are missing. Weight is 96.81% missing. Specialty is 48.96% missing and payer is 39.59%; those remain, with an explicit Missing category. This is a documented, bounded choice rather than a universal missingness threshold.

Four medication columns are constant in development and are also dropped. A constant variable cannot distinguish rows there. That leaves 42 retained raw inputs. Exclusions are relearned from each training set, so a different cohort may have a different set of constant fields.

## Why use median imputation and what actually changed

The median is less affected by extreme numerical values than the mean. The pipeline includes median imputation so a future row with an absent numeric value can be transformed. In this development dataset, all eight numeric inputs have zero missing cells. Therefore the numerical imputer changes no observed values in this run. The categorical Missing fill does affect fields with missing entries.

Do not describe a method as an observed change when it merely exists as a precaution. Check `numeric_transform_audit.csv` for evidence.

## Calculate a standardized value

The rule is `z = (x - mean_training) / scale_training`. For length of stay, a four-day encounter gives approximately `(4 - 4.364) / 2.953 = -0.123`. A negative z-score means below the training mean; it does not mean an invalid negative hospital stay.

StandardScaler uses population variance (`ddof=0`). After fitting and transforming the same nonconstant training feature, the mean is approximately zero and the population standard deviation approximately one. A pandas `.std()` result uses `ddof=1` by default and can differ slightly. Standardization does not make a skewed variable Gaussian and does not remove outliers.

## Why use one-hot encoding rather than treat all codes as numbers

An admission-type code of 6 is not necessarily twice the quantity represented by code 3. Treating such codes as continuous introduces an arbitrary order and distance. One-hot encoding gives separate indicator columns. Age bands also remain categories because the source supplies ranges, not exact ages.

The encoder learns its categories from training data. Categories occurring fewer than 25 times are pooled. An unseen category goes into the learned pooled category if one exists; otherwise it becomes an all-zero block for that input. It does not change the number of columns during inference.

## How can 42 features become 216 columns

Raw features are variables such as age or payer. Encoded columns include individual category indicators. The eight numeric variables contribute one column each, and each retained categorical variable can contribute several. All 42 retained raw inputs produce 216 encoded columns. The 20 MI-selected raw inputs produce 147 columns. Neither count means there are 216 original clinical measurements.

## Explain a histogram and a boxplot from this project

The histogram shows counts across observed values. Its count axis is logarithmic so rare high values remain visible; it includes all data. The box spans the 25th to 75th percentiles, with the median inside. These particular boxplots use 5th-95th percentile whiskers and hide points outside them for clarity. Those plotting choices do not delete observations and are distinct from the 1.5-IQR screening rule.

The emergency-visit box often collapses at zero because most encounters have no prior emergency visits. A tiny box is evidence of a concentrated distribution, not a plotting failure.

## Why can the IQR rule overflag this dataset

The rule flags values below `Q1 - 1.5*IQR` or above `Q3 + 1.5*IQR`, where `IQR = Q3 - Q1`. If Q1 and Q3 are both zero, both fences equal zero. Every positive visit count is then flagged. For emergency visits that is 6,709 encounters, even though having a prior emergency visit is plausible.

Across eight variables, 20,162 encounters trigger at least one IQR flag. This is not a measured data-error rate. Isolation Forest offers a complementary multivariable view; it flags 1,791 encounters under a chosen 3% setting. Its setting does not discover that exactly 3% of records are wrong. We retain every eligible encounter because flags are not confirmed errors.

## Why is log1p used for outlier screening but not the main numeric matrix

`log1p(x)` means `log(1+x)`, which is defined at zero and compresses large positive counts. The Isolation Forest screening route uses log1p, imputation and scaling on numeric utilization features. The primary proposal-compatible preprocessing route uses the original counts followed by median imputation and standardization. These serve different purposes and should not be described as one universal transformation.

## What does Spearman correlation tell us

Spearman correlation measures association between ranks, making it useful for monotonic relationships among skewed counts. The largest numerical pair here is length of stay versus medication count, rho about 0.463. None exceeds absolute 0.80. A small pairwise coefficient does not prove independence, and a large coefficient does not establish causation. The heatmap omits categorical variables; it is not a complete redundancy analysis.

## Why is mutual information an independent selection method

The MI selector measures each raw input's estimated dependence on the target without fitting a classifier. It is a filter method rather than tree importance or a wrapper that repeatedly scores models. It uses labels, so it is supervised despite being independent of classifier fitting.

The selector treats counts/categories as discrete, ranks 42 eligible raw features and keeps 20. It ranks prior inpatient visits first. MI can miss interactions and can favor variables with many distinct values. A high score is not a causal effect and does not say how many accuracy points a feature adds.

## Why is PCA different from MI

| Property | MI filter here | PCA here |
|---|---|---|
| Uses outcome labels | Yes, development labels | No |
| Input | Cleaned raw features | All 216 encoded columns |
| Output | A subset of 20 existing variables | 65 new linear combinations |
| Criterion | Estimated dependence on the target | At least 95% of representation variance |
| Interpretation | Original variable meaning retained | Each component mixes multiple encoded inputs |

PCA centers the encoded representation. Numeric features have already been standardized; binary one-hot indicators are not scaled again. It retains 95.11% of that representation's variance in 65 components. The first two account for only 19.15%. PCA does not prove 95% predictive accuracy and might discard low-variance information useful for the target.

## Where would leakage occur if we reused the exported matrix in CV

The exported top-20 matrix was selected using every development label. If we then cross-validated a classifier on it, validation-fold labels would already have influenced the feature ranking. The correct procedure is to give raw inputs to a Pipeline and fit every learned stage on each training fold. The existing full-project classifier workflow does that. The exported matrices are for inspection and development-trained fits, not a shortcut around fold-local preprocessing.

## Why is no data augmentation performed

Augmentation or oversampling changes the data distribution and is not necessary to describe or clean this cohort. The screenshot asks us to discuss it; no synthetic data are added. If a later experiment uses resampling, it must happen only in training folds. Ordinary interpolation across one-hot clinical categories can produce meaningless combinations. Preserving class prevalence also avoids presenting synthetic counts as observed hospital utilization.

## What should we say about the HbA1c comparison

The observed rates are unadjusted descriptive associations. The None group has 49,453 encounters and an 11.69% early-readmission rate; the Norm group has 2,968 encounters and a 10.58% rate. That does not show that testing prevents readmission. Case mix, care decisions, missing follow-up and repeated patients can explain differences. Keep sample sizes and limitations next to any comparison.

## Practice tracing one file from input to report

`readmission/data.py` constructs the patient-separated cohort. `section3/analysis.py` reads development encounters, calculates missingness and writes `outputs/tables/missingness.csv`. It also draws `01_population_missingness.png`. The notebook displays those artifacts, and `build_reference_report.py` incorporates the figure and measured values into the report. `manifest.json` records hashes so `run_section()` can detect stale outputs.

The same pattern connects MI rankings, PCA variance and outlier profiles. Knowing this path helps you identify whether a mistaken claim requires a code correction, a rerun, or only a change in the team's prose.
