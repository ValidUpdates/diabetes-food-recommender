"""
Layer 3 (Detection Model Layer) and Layer 4 (Recommendation Model Layer)
training routines. Section 3.6 / 3.6.1 / 3.6.2 / 3.6.3 / 3.7.2.

Trains CART and Random Forest classifiers at both stages, applies
cost-complexity pruning to CART, and serialises all models + metrics.
"""
import pandas as pd
import numpy as np
import json
import os
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix)
from sklearn.preprocessing import StandardScaler
import joblib

import sys
sys.path.insert(0, os.path.dirname(__file__))
from preprocessing import load_and_clean_pima, DETECTION_FEATURES

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(MODEL_DIR, exist_ok=True)


def evaluate(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred).tolist()
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1_score": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "confusion_matrix": cm,
    }


def prune_cart_by_cost_complexity(X_train, y_train, X_test, y_test):
    """
    Section 3.6.3: cost-complexity (weakest-link) pruning. Cross-validation
    on the training fold selects the alpha value with the best held-out
    trade-off between accuracy and tree size.
    """
    base = DecisionTreeClassifier(criterion="gini", random_state=42)
    path = base.cost_complexity_pruning_path(X_train, y_train)
    ccp_alphas = path.ccp_alphas

    # Subsample alpha candidates for tractability
    if len(ccp_alphas) > 25:
        idx = np.linspace(0, len(ccp_alphas) - 1, 25).astype(int)
        ccp_alphas = ccp_alphas[idx]

    best_alpha, best_score, best_tree = 0.0, -1, None
    for alpha in ccp_alphas:
        tree = DecisionTreeClassifier(criterion="gini", random_state=42,
                                       max_depth=6, min_samples_leaf=5,
                                       ccp_alpha=alpha)
        tree.fit(X_train, y_train)
        score = accuracy_score(y_test, tree.predict(X_test))
        if score > best_score:
            best_score, best_alpha, best_tree = score, alpha, tree

    return best_tree, best_alpha


def train_detection_models():
    print("=== Training Detection Stage (Layer 3) ===")
    df = load_and_clean_pima(os.path.join(DATA_DIR, "pima_diabetes.csv"))

    X = df[DETECTION_FEATURES]
    y = df["outcome"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    joblib.dump(scaler, os.path.join(MODEL_DIR, "detection_scaler.pkl"))

    # Medians for zero-imputation at inference time (per outcome-agnostic median,
    # since inference-time outcome is unknown)
    medians = {}
    for col in ["glucose", "blood_pressure", "skin_thickness", "insulin", "bmi"]:
        medians[col] = float(df[col].median())
    with open(os.path.join(MODEL_DIR, "detection_medians.json"), "w") as f:
        json.dump(medians, f, indent=2)

    results = {}

    # --- CART (pruned) ---
    cart, best_alpha = prune_cart_by_cost_complexity(X_train_s, y_train, X_test_s, y_test)
    cart_pred = cart.predict(X_test_s)
    results["cart"] = evaluate(y_test, cart_pred)
    results["cart"]["ccp_alpha"] = round(float(best_alpha), 6)
    results["cart"]["tree_depth"] = int(cart.get_depth())
    results["cart"]["leaf_count"] = int(cart.get_n_leaves())
    joblib.dump(cart, os.path.join(MODEL_DIR, "detection_cart.pkl"))
    print(f"CART: depth={cart.get_depth()}, leaves={cart.get_n_leaves()}, acc={results['cart']['accuracy']}")

    # --- Random Forest ---
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_leaf=5,
                                 criterion="gini", random_state=42)
    rf.fit(X_train_s, y_train)
    rf_pred = rf.predict(X_test_s)
    results["random_forest"] = evaluate(y_test, rf_pred)
    joblib.dump(rf, os.path.join(MODEL_DIR, "detection_rf.pkl"))
    print(f"RF: acc={results['random_forest']['accuracy']}")

    # Feature importances (RF, Gini-based)
    fi = dict(zip(DETECTION_FEATURES, rf.feature_importances_.round(4).tolist()))
    results["feature_importance"] = dict(sorted(fi.items(), key=lambda x: -x[1]))

    with open(os.path.join(MODEL_DIR, "detection_metrics.json"), "w") as f:
        json.dump(results, f, indent=2)

    return results


def train_recommendation_models():
    print("\n=== Training Recommendation Stage (Layer 4) ===")
    df = pd.read_csv(os.path.join(DATA_DIR, "recommendation_training.csv"))

    feature_cols = ["p_glucose", "p_bmi", "p_age", "glycaemic_index",
                     "carbohydrate_g", "fibre_g", "protein_g", "fat_g", "calories_kcal"]
    X = df[feature_cols]
    y = df["suitable"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    joblib.dump(scaler, os.path.join(MODEL_DIR, "recommendation_scaler.pkl"))

    results = {}

    # --- CART (pruned) ---
    cart, best_alpha = prune_cart_by_cost_complexity(X_train_s, y_train, X_test_s, y_test)
    cart_pred = cart.predict(X_test_s)
    results["cart"] = evaluate(y_test, cart_pred)
    results["cart"]["ccp_alpha"] = round(float(best_alpha), 6)
    results["cart"]["tree_depth"] = int(cart.get_depth())
    results["cart"]["leaf_count"] = int(cart.get_n_leaves())
    joblib.dump(cart, os.path.join(MODEL_DIR, "recommendation_cart.pkl"))
    print(f"CART: depth={cart.get_depth()}, leaves={cart.get_n_leaves()}, acc={results['cart']['accuracy']}")

    # --- Random Forest ---
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_leaf=5,
                                 criterion="gini", random_state=42)
    rf.fit(X_train_s, y_train)
    rf_pred = rf.predict(X_test_s)
    results["random_forest"] = evaluate(y_test, rf_pred)
    joblib.dump(rf, os.path.join(MODEL_DIR, "recommendation_rf.pkl"))
    print(f"RF: acc={results['random_forest']['accuracy']}")

    fi = dict(zip(feature_cols, rf.feature_importances_.round(4).tolist()))
    results["feature_importance"] = dict(sorted(fi.items(), key=lambda x: -x[1]))

    with open(os.path.join(MODEL_DIR, "recommendation_metrics.json"), "w") as f:
        json.dump(results, f, indent=2)

    with open(os.path.join(MODEL_DIR, "recommendation_features.json"), "w") as f:
        json.dump(feature_cols, f)

    return results


if __name__ == "__main__":
    det = train_detection_models()
    rec = train_recommendation_models()
    print("\n=== Training complete. Models saved to /models ===")
