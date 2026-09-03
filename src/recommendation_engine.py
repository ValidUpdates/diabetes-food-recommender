"""
Layer 4 output logic (Section 3.7.2): pairs a detected-positive patient's
health vector against every food item, classifies suitability, ranks
suitable items by the composite score, and returns the top 10.

Composite Score = 0.4 * (1 - normalised_GI) + 0.35 * normalised_fibre + 0.25 * normalised_protein
"""
import pandas as pd
import numpy as np
import os
import joblib
import json

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

RECOMMENDATION_FEATURES = ["p_glucose", "p_bmi", "p_age", "glycaemic_index",
                            "carbohydrate_g", "fibre_g", "protein_g", "fat_g", "calories_kcal"]


def load_food_database():
    return pd.read_csv(os.path.join(DATA_DIR, "nigerian_food_nutrition.csv"))


def composite_score(gi_series, fibre_series, protein_series):
    """0.4*(1-normGI) + 0.35*normFibre + 0.25*normProtein"""
    def normalise(s):
        rng = s.max() - s.min()
        if rng == 0:
            return pd.Series(0.5, index=s.index)
        return (s - s.min()) / rng

    norm_gi = normalise(gi_series)
    norm_fibre = normalise(fibre_series)
    norm_protein = normalise(protein_series)

    return 0.4 * (1 - norm_gi) + 0.35 * norm_fibre + 0.25 * norm_protein


def explain_recommendation(food_row, score):
    reasons = []
    if food_row["glycaemic_index"] <= 40:
        reasons.append("low glycaemic index")
    elif food_row["glycaemic_index"] <= 55:
        reasons.append("moderate glycaemic index")
    if food_row["fibre_g"] >= 3:
        reasons.append("high fibre content")
    if food_row["protein_g"] >= 8:
        reasons.append("good protein content")
    if not reasons:
        reasons.append("acceptable nutritional balance for the patient profile")
    return "Recommended for " + ", ".join(reasons) + f" (composite score: {score:.3f})"


def get_recommendations(patient_glucose, patient_bmi, patient_age, algorithm="random_forest", top_n=10):
    """
    Layer 4 core routine: pair patient vector against all 150 food items,
    classify each with the selected recommendation classifier, filter to
    suitable items, rank by composite score, return top N with explanations.
    """
    foods = load_food_database()
    scaler = joblib.load(os.path.join(MODEL_DIR, "recommendation_scaler.pkl"))

    model_file = "recommendation_cart.pkl" if algorithm == "cart" else "recommendation_rf.pkl"
    model = joblib.load(os.path.join(MODEL_DIR, model_file))

    rows = []
    for _, food in foods.iterrows():
        rows.append({
            "p_glucose": patient_glucose, "p_bmi": patient_bmi, "p_age": patient_age,
            "glycaemic_index": food["glycaemic_index"], "carbohydrate_g": food["carbohydrate_g"],
            "fibre_g": food["fibre_g"], "protein_g": food["protein_g"],
            "fat_g": food["fat_g"], "calories_kcal": food["calories_kcal"],
        })
    X = pd.DataFrame(rows)[RECOMMENDATION_FEATURES]
    X_scaled = scaler.transform(X)

    predictions = model.predict(X_scaled)
    foods = foods.copy()
    foods["suitable"] = predictions

    suitable_foods = foods[foods["suitable"] == 1].copy()

    if len(suitable_foods) == 0:
        # Fallback: if classifier finds none suitable, still surface the
        # lowest-GI/highest-fibre items so the clinician has something to review
        suitable_foods = foods.copy()

    suitable_foods["composite_score"] = composite_score(
        suitable_foods["glycaemic_index"], suitable_foods["fibre_g"], suitable_foods["protein_g"]
    )
    suitable_foods = suitable_foods.sort_values("composite_score", ascending=False)

    top = suitable_foods.head(top_n).copy()
    top["explanation"] = top.apply(lambda r: explain_recommendation(r, r["composite_score"]), axis=1)

    return top[["name", "category", "glycaemic_index", "fibre_g", "protein_g",
                "carbohydrate_g", "calories_kcal", "composite_score", "explanation"]].to_dict("records")
