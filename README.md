# A Hybrid-Based Diabetes Food Recommender System

Software implementation of the thesis system: a two-stage decision support
pipeline that runs diabetes detection first, then conditionally invokes a
Nigerian food recommendation stage only when detection is positive. Both
stages compare CART (pruned decision tree) and random forest classifiers,
matching Chapter Three's methodology and Section 3.7's architecture.

## What's implemented

- **Layer 1 — Application Layer**: Flask web app with three interfaces —
  Patient (submit clinical parameters, view results), Healthcare Worker
  (review logs, override outcomes, manage the food database), and Admin
  (compare CART vs random forest metrics, swap active classifiers, trigger
  retraining).
- **Layer 2 — Preprocessing Pipeline**: input validation, median imputation
  (stratified by outcome class for training; population median at
  inference), z-score scaling — two parallel branches, one per stage.
- **Layer 3 — Detection Model Layer**: CART and random forest trained on
  the real Pima Indians Diabetes Dataset (768 records, 8 clinical
  attributes), with cost-complexity pruning applied to CART.
- **Layer 4 — Recommendation Model Layer**: CART and random forest trained
  on a patient-food interaction dataset, gated so it only runs after a
  positive detection outcome. Suitable foods are ranked by the composite
  score from Section 3.7.2: `0.4×(1-normGI) + 0.35×normFibre + 0.25×normProtein`.
- **Layer 5 — Data Layer**: the Pima dataset, a 150-item Nigerian food
  nutrition database, and serialized model files.

## What to know before your supervisor meeting

- **The Pima dataset is real** — downloaded from the standard UCI-derived
  source, 768 records with the same 500/268 class split the thesis cites.
- **The Nigerian food nutrition database (150 items) is generated**, not
  sourced from a single published dataset. GI, fibre, protein, and other
  values are grounded in published glycaemic-index literature and typical
  food-composition figures, adapted to representative per-100g estimates —
  reasonable for a working system and a supervisor demo, but not a
  certified clinical nutrition database. If your program has access to a
  proper Nigerian food composition table (FAO/NFCT or a departmental
  dataset), swap `data/nigerian_food_nutrition.csv` for it — the pipeline
  doesn't care where the 8 columns come from.
- **The recommendation-stage training labels are rule-derived**, not
  clinician-annotated. `src/build_recommendation_dataset.py` generates
  synthetic patient profiles and labels food suitability using an
  ADA-aligned scoring rule (favouring low GI, high fibre, adequate
  protein). This is a standard approach when clinician-labelled ground
  truth isn't available, but it's worth being upfront about in your
  defense if asked how suitability was determined.
- All performance figures (accuracy, precision, recall, F1, tree depth,
  leaf counts) are real numbers from real training runs on this data —
  nothing in `models/*_metrics.json` is hand-typed.

## Setup

```bash
cd diabetes-food-recommender
pip install -r requirements.txt
```

## Train the models (already done, but to reproduce or retrain)

```bash
cd src
python3 train_models.py
```

This regenerates `models/*.pkl` and `models/*_metrics.json`.

## Run the app

```bash
python3 app.py
```

Visit `http://localhost:5000`. Three entry points:
- `/patient` — run a screening
- `/healthcare-worker` — review logs, override outcomes, manage food data
- `/admin` — compare classifiers, switch the active model, retrain

## Project structure

```
diabetes-food-recommender/
├── app.py                          # Flask app (Layer 1)
├── requirements.txt
├── data/
│   ├── pima_diabetes.csv           # Real Pima dataset, 768 records
│   ├── nigerian_food_nutrition.csv # 150-item generated food database
│   ├── recommendation_training.csv # Patient-food interaction training set
│   └── generate_food_data.py       # Regenerates the food database
├── models/                         # Serialized classifiers + metrics (generated)
├── src/
│   ├── preprocessing.py            # Layer 2
│   ├── train_models.py             # Layer 3 + Layer 4 training routines
│   ├── recommendation_engine.py    # Composite scoring + ranking logic
│   └── build_recommendation_dataset.py
├── templates/                      # Patient, Healthcare Worker, Admin UIs
└── static/css/style.css
```

## Extending this for your final submission

- Swap in a real Nigerian food composition dataset if your department has
  one on file — this is the single highest-value change for external
  validity.
- If you can get even a small set of clinician-reviewed food suitability
  labels, retraining the recommendation classifiers on those instead of
  the rule-derived labels would strengthen your methodology section.
- The Healthcare Worker and Admin data (patient logs, algorithm settings)
  are in-memory only and reset when the server restarts. For a persistent
  deployment, swap `PATIENT_LOG` in `app.py` for a real database — SQLite
  is a low-effort option that would fit your existing architecture
  description without changing Layer 5's conceptual role.
