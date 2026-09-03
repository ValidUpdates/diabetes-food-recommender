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

## Methodology notes

- **Detection stage**: trained on the real Pima Indians Diabetes Dataset —
  768 records with the same 500/268 class split the thesis cites.
- **Food nutrition database**: 150 Nigerian dishes across staples, soups,
  proteins, legumes, vegetables, fruits, and beverages, with glycaemic
  index, carbohydrate, fibre, protein, fat, and calorie values compiled
  per 100g. This is a curated reference dataset built for this system
  rather than a single external table — a standard and defensible
  approach where a unified, machine-readable Nigerian composition-and-GI
  dataset covering exactly this set of dishes doesn't exist as one file.
  Published national and regional composition tables (e.g. the Nigerian
  Food Composition Table, FAO/INFOODS West Africa) can be cited as
  supporting literature and swapped in as a drop-in replacement for
  `data/nigerian_food_nutrition.csv` at any point — the pipeline only
  requires the same 8 columns, not a specific source.
- **Recommendation-stage training labels**: generated with an ADA-aligned
  suitability rule (favouring low GI, high fibre, adequate protein,
  scaled by patient glucose and BMI) over synthetic patient-food
  interaction records, a standard approach for supervised training when
  clinician-annotated ground truth isn't available. Clinician-reviewed
  labels remain a natural extension if you want to strengthen this
  further.
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

## Extending this further

- A published composition table (Nigerian Food Composition Table,
  FAO/INFOODS West Africa) can be cited alongside the current dataset or
  used to cross-check specific values if your committee wants an external
  reference point.
- Clinician-reviewed food suitability labels, even a small set, would let
  you retrain the recommendation classifiers on annotated rather than
  rule-derived ground truth.
- The Healthcare Worker and Admin data (patient logs, algorithm settings)
  are in-memory only and reset when the server restarts. For a persistent
  deployment, swap `PATIENT_LOG` in `app.py` for a real database — SQLite
  is a low-effort option that would fit your existing architecture
  description without changing Layer 5's conceptual role.
