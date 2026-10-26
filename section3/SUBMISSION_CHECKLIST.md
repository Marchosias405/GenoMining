# Section 3 submission checklist

## Materials to use for this section

| Material | Role |
|---|---|
| Your team's final report | Discuss methods, measured findings, challenges and limitations in your own verified explanation. Add names and accurate contributions. |
| `section3_eda_preprocessing.ipynb` | Executed, error-free code walkthrough with saved tables and four figures. |
| Focused source and dependencies | Include `section3/`, the shared modules listed in the README, `config.json`, `requirements.txt` and `tests/test_section3.py`. The packaging command selects them. |
| Evidence tables and four PNGs | Supply exact values behind the discussion. Include only the figures needed in the report; CSVs can remain supporting material. |
| Data provenance and dictionary | Source URL/DOI, licence, data hashes, eligibility rules and original-column definitions. |
| Verification records | `outputs/test_results.xml` and `outputs/verification.json` record completed checks. |
| README and file guide | Explain installation, execution, file relationships, outputs and interpretation. |
| Reference report and study guide | Learning aids for the team. They are labelled separately from your final report. |

The generated ZIP is a working submission bundle, not a substitute for your final report. It includes public source data and generated matrices for offline review. Follow course portal limits if a smaller upload or repository link is required.

## Reproduce before submitting

Run from the project root with the project environment active:

```bash
python -m section3.run
python -m pytest tests/test_section3.py -q --junitxml=section3/outputs/test_results.xml
python -m section3.build_notebook --execute
python -m section3.verify
python -m section3.package --zip ../outputs/GenoMining-section3-submission.zip
```

If you change analysis code or settings, rerun the analysis and tests, refresh the notebook, and regenerate the reference report only if you want an updated reference. Check that your final report uses the new numbers. Preserve notebook/Markdown edits before using their generators.

## Three person review and writing plan

| Teammate | First review responsibility | Evidence to explain to the other two |
|---|---|---|
| Member A | Dataset, cohort and EDA | Encounter versus patient, patient grouping, class imbalance, missingness, histograms and boxplots. |
| Member B | Cleaning, encoding and outliers | None versus missing, exclusions, median/scaling parameters, one-hot expansion, IQR pitfalls and retention decision. |
| Member C | MI, PCA and reproducibility | Supervised filter versus unsupervised components, dimensions/variance, file hashes, matrix alignment and leakage risk in CV. |

These are suggested future review assignments, not claims about who implemented the existing work. Each member should run the notebook and explain another member's section before writing. Keep actual edits/reviews in your own Git commits; do not fabricate contribution history.

## Final content checks

- State the prediction time (discharge), label definition and unit of analysis (encounter).
- Distinguish full-cohort counts from development EDA counts and percentages.
- Explain numerical imputation as a configured rule that changed zero numerical cells in this run.
- Discuss why unusual values were retained and why no data augmentation was applied.
- Explain both raw feature counts and encoded column counts.
- Treat MI rankings as associations and PCA variance as a representation property, not model accuracy.
- Use exactly the figures that support your narrative; avoid adding unrelated classifier or clustering results to Section 3.
- Verify that all tables/figures have readable labels, captions, sample sizes where relevant and source attribution.
- Confirm page limits, filename conventions, required member details, allowed file formats and course rules in the full handout/portal; these are not specified in the screenshot.

The requirement map identifies what is covered. The final judgement remains with the instructor; the team is responsible for understanding and verifying the submitted report.
