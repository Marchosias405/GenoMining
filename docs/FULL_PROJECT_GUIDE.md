# GenoMining — Hospital Readmission Prediction

A three-person data-mining project investigating recorded early hospital readmission among patients with diabetes. It compares logistic regression, random forest, and XGBoost; checks whether mutual-information feature selection reduces inputs; and evaluates whether sigmoid calibration improves probability estimates. The code implements the submitted proposal and the course's clustering/outlier requirements. Your team writes the academic report separately.

**Example use:** a healthcare analyst can use the notebooks to compare how well models rank historical encounters and whether a predicted risk agrees with observed readmission rates. The command-line example scores three public, de-identified encounters. These are research estimates from historical US records, not a validated clinical decision tool.

## Start here

1. Read `docs/EXPERIMENT_DESIGN.md` for the experimental rules and limitations.
2. Open notebooks `01` through `05` in order. Saved outputs show the executed analysis.
3. See `artifacts/tables/test_metrics.csv` and `artifacts/figures/` for numerical evidence and figures.
4. Use `FILE_GUIDE.md` for a short description of each delivered file and why it exists.

## Dataset

[UCI Diabetes 130-US Hospitals for Years 1999-2008](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008), DOI [10.24432/C5230J](https://doi.org/10.24432/C5230J), CC BY 4.0, by Clore, Cios, DeShazo, and Strack.

- Source: 101,766 encounters, 71,518 patients, 47 candidate inputs, two identifiers, one target.
- Cohort: exclude encounters with death/hospice discharge codes 11, 13, 14, 19, 20, 21. This leaves 99,343 encounters from 69,990 patients.
- Positive: existing `readmitted == '<30'`. Negative: `'>30'` or `'NO'`. The positive proportion is 11,314 / 99,343 = 11.39% **of encounters**.
- The prediction time is discharge. Completed-stay variables are therefore eligible.
- `?` is missing; the literal `None` for laboratory tests means **not performed** and must be preserved.
- The download is automatic if the raw archive is absent. Its SHA-256 is stored in `artifacts/data_audit.json`.

## Run locally

Clone the project before following the setup instructions:

```bash
git clone https://github.com/Marchosias405/GenoMining.git
cd GenoMining
```

The GitHub repository includes source, executed notebooks, figures, and small result tables. Running `run.py all` downloads the public dataset and regenerates the data, trained models, and predictions omitted from Git.

Python **3.12** was used for the delivered run. From this project directory:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe run.py all
.\.venv\Scripts\python.exe -m pytest tests -q
```

On macOS/Linux:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python run.py all
.venv/bin/python -m pytest tests -q
```

The supplied source and artifacts are portable; the local `.venv/` is not included in the shareable archive. `requirements.txt` pins direct dependencies. `requirements-lock.txt` records the complete installed environment for audit; use `requirements.txt` for cross-platform setup.

### Run one phase

```bash
python run.py data
python run.py eda
python run.py unsupervised
python run.py models
python run.py evaluation
```

Commands reuse artifacts only when their required files and source/configuration fingerprints match. Add `--force` to recompute a requested phase, or run `python run.py all --force` for a complete reproduction. The modeling phase can take several minutes or longer on a CPU. It uses the full development cohort, five grouped folds, and four randomized configurations per classifier/feature setting, plus baselines. No GPU is needed. All settings are visible in `config.json`.

To refresh all notebooks with retained outputs after a run, use `python scripts/build_notebooks.py --execute`. Notebooks 01 and 02 are authored directly and their cells are preserved; the runner executes them without regenerating their content. Later notebooks are regenerated, so save personal edits to those elsewhere first. `python scripts/verify_delivery.py` checks the delivered evidence; `python scripts/inventory.py` refreshes the file guide.

### Example prediction

After the evaluation phase, run:

```bash
python run.py predict --input examples/encounters.csv --output artifacts/predictions/example_predictions.csv
```

The input has the same 47 candidate-feature columns as the original CSV, without identifiers or the target. Three public test encounters are supplied solely to demonstrate the interface. The output contains the input-row number, calibrated probability of recorded early readmission, and the predicted class at the predeclared 0.5 threshold. The actual example output is saved beside the test predictions. Neither the prediction command nor these examples retrain the model.

Only load the `.joblib` files produced by this project or another trusted source; they are Python serialized objects.

## Colab or Kaggle

Upload/extract the complete project archive (not just one notebook) into `/content/hospital-readmission` in Colab or `/kaggle/working/hospital-readmission` in Kaggle. Internet must be enabled if downloading the data or dependencies. Open any project notebook and run a setup cell appropriate to the platform:

```python
%cd /content/hospital-readmission
%pip install -r requirements.txt
```

For Kaggle, change the directory to `/kaggle/working/hospital-readmission`. Restart the runtime after installation if the platform already imported different NumPy/scikit-learn versions. Then run the notebook cells in order. A fresh environment may invalidate saved fingerprints and recompute phases as needed. Keep cell outputs when saving for submission.

## Folder structure

```text
hospital-readmission/
  README.md                    # Introduction, setup, usage example, and project map
  readme.me                    # Requested-filename overview, example and folder map
  FILE_GUIDE.md                # File-by-file purpose and evidence inventory
  config.json                 # Fixed experiment settings
  requirements.txt            # Pinned direct Python dependencies
  requirements-lock.txt       # Complete environment recorded for this run
  run.py                      # CLI for phases and CSV prediction
  readmission/
    common.py                 # Shared paths, settings, hashing, and output helpers
    data.py                   # Download, validation, cohort, dictionary, patient splits
    features.py               # Fold-local cleaning, mutual information, encoding
    eda.py                    # Development-only exploratory tables and figures
    unsupervised.py           # Three clustering and three outlier comparisons
    models.py                 # Grouped CV, tuning, feature-selection comparisons
    evaluation.py             # Independent calibration and held-out evaluation
    pipeline.py               # Resumable phases and provenance checks
  notebooks/
    01_data_preparation.ipynb
    02_exploratory_analysis.ipynb
    03_clustering_outliers.ipynb
    04_model_training.ipynb
    05_evaluation.ipynb
  data/
    raw/                      # Original UCI CSV, mapping, and download archive
    processed/                # Eligible cohort and immutable patient split assignment
  artifacts/
    figures/                  # PNGs for inspection and your own report
    tables/                   # EDA, CV, test metrics, runtime, and uncertainty evidence
    models/                   # Twelve base fits, six calibrated fits, and registries
    predictions/              # Held-out and example predictions
    status/                   # Phase completion records with hashes and environment
    *.json                    # Audits and experimental design metadata
  examples/encounters.csv     # Three public rows for the prediction example
  docs/
    EXPERIMENT_DESIGN.md       # Decisions, comparisons, references, limitations
    TEAM_WORK.md               # Suggested team review and report-evidence checklist
  scripts/
    build_notebooks.py         # Create/execute notebooks with visible outputs
    inventory.py               # Produce file descriptions and a portable project ZIP
    verify_delivery.py         # Check artifacts, splits, metrics, notebooks, predictions
  tests/test_integrity.py      # Meaningful regression tests for leakage/target semantics
```

## Phases and completion evidence

| Phase | What it answers | Evidence |
|---|---|---|
| 1. Data preparation | Is the source correct and are patients separated? | Data audit, dictionary, split summary, notebook 01 |
| 2. Exploration | What is missing, imbalanced, skewed, or associated? | EDA tables/figures, preprocessing audit, notebook 02 |
| 3. Unsupervised analysis | What groups and unusual encounters appear? | Nine clustering settings, three outlier detectors, notebook 03 |
| 4. Supervised comparison | Which classifier/input set performs best in development? | Fold scores, search tables, model choice, notebook 04 |
| 5. Calibration and validation | How do frozen models perform on unseen patients? | Test metrics, curves, patient-bootstrap intervals, notebook 05 |

`artifacts/status/` records only phases that actually completed. `artifacts/delivery_verification.json` records final verification. Do not infer completion from the checklist alone.

## Important interpretation rules

- Split by patient, not row. Training, calibration, and test patients never overlap.
- Hyperparameters, preprocessing and feature selection are learned inside development folds. The final family/input set is selected using development average precision, not the test leaderboard.
- Calibration fits an independent sigmoid while the base model stays frozen. Its benefit is measured, not assumed.
- The baseline and tuned models use the same held-out cohort. Report feature counts and measured fit/prediction time, not just scores.
- A classifier predicting no early readmissions can obtain high accuracy because of imbalance; examine recall, average precision and the dummy baselines.
- Clustering uses numerical features; readmission rates are added afterward for description only. Outlier flags are not proof of bad data, so eligible encounters remain in classifier training.
- Historical records, missing variables and incomplete capture of readmissions limit applicability. There are no released hospital IDs or usable visit dates for external-hospital or temporal validation. Subgroup analysis does not establish fairness.
- The report, clinical interpretation, and individual contribution claims are for your team to write and verify. This repository contains implementation and evidence, not a completed academic report.

## Git and team workflow

Use a shared repository and keep each member's own commits. The `.gitignore` omits environments, caches, large row-level data, serialized models and predictions; data/models can be reproduced with `run.py all`. Preserve executed notebooks, figures and small evidence tables in Git. `docs/TEAM_WORK.md` suggests ownership for review, without claiming anyone performed work they have not done. Give the instructor/TAs access according to the course instructions.
