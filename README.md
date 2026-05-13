# Titanic Survival — Feature Engineering & Selection

A complete feature engineering and selection pipeline applied to the Titanic dataset, comparing a baseline Logistic Regression model against a feature-engineered version to evaluate the impact of thoughtful data preparation on predictive performance.

---

## Project Description

This project explores the full feature engineering workflow on the classic Titanic survival dataset. Starting from raw, messy data, the notebook walks through missing value imputation, categorical encoding, custom feature construction, and multiple feature selection strategies — culminating in a comparison of model performance before and after engineering.

The goal is not just to maximize accuracy, but to understand *why* certain features matter, how to represent domain knowledge in numerical form, and how to select a compact, interpretable feature set without sacrificing predictive power.

A baseline Logistic Regression model trained on minimally processed data is compared against a model trained on the full engineered feature set, and finally against a reduced model built from the top 8 features selected via Recursive Feature Elimination (RFE).

---

## Techniques Covered

| Stage | Method |
|---|---|
| Missing value imputation | Median (continuous), Mode (categorical) |
| Categorical encoding | One-hot encoding with `pd.get_dummies` |
| Feature engineering | `family_size`, `is_alone`, `fare_per_person` |
| Feature selection (Filter) | Mutual Information (`mutual_info_classif`) |
| Feature selection (Wrapper) | Recursive Feature Elimination (RFE) |
| Modeling | Logistic Regression |
| Evaluation | Accuracy, Precision, Recall, F1-score |

---

## Engineered Features

| Feature | Formula | Rationale |
|---|---|---|
| `family_size` | `sibsp + parch + 1` | Captures full social unit size; group size affects evacuation behaviour |
| `is_alone` | `1 if family_size == 1 else 0` | Isolates solo travellers, a strong survival predictor |
| `fare_per_person` | `fare / family_size` | Normalizes ticket cost to individual wealth rather than group spend |

---

## Results Summary

| Model | Accuracy |
|---|---|
| Baseline (no engineering) | ~0.80 |
| Feature-engineered model | ~0.80 (improved recall on survivors) |
| RFE-reduced model (top 8 features) | ~0.804 |

RFE selected: `pclass`, `age`, `sibsp`, `sex_male`, `embarked_Q`, `embarked_S`, `family_size`, `fare_per_person`

---

## Project Structure

```
├── Titanic_Feature_Engineering.ipynb   # Main notebook
└── requirements.txt                                        # Python dependencies
```

> **Note:** The Titanic dataset is loaded directly via `seaborn.load_dataset("titanic")` — no separate data file needed.

---

## Requirements

```bash
pip install -r requirements.txt
```

---

## Usage

Open the notebook in Jupyter or VS Code and run all cells:

```bash
jupyter notebook Titanic_Feature_Engineering.ipynb
```

All outputs and plots are generated inline within the notebook.

---

**Author:** Harmit Patel
# feature_engineering_ml
