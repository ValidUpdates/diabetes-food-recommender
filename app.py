"""
Layer 1 - Application Layer (Section 3.7.1)
Flask HTTP Router serving: Patient Web Interface, Healthcare Worker UI,
and Admin Model Management Panel.

Routes patient inputs through the detection pipeline first, and
conditionally routes onward to the recommendation pipeline only where
the detection outcome is positive (the Layer 3 -> Layer 4 conditional gate).
"""
import os
import sys
import json
import uuid
from datetime import datetime

import joblib
import pandas as pd
from flask import Flask, render_template, request, jsonify, session

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from preprocessing import validate_patient_input, DETECTION_FEATURES
from recommendation_engine import get_recommendations, load_food_database

BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")

app = Flask(__name__)
app.secret_key = "thesis-demo-secret-key-change-in-production"

# ---------------------------------------------------------------------------
# In-memory stores (Layer 5 - Data Layer surrogate for demo purposes).
# A production deployment would back these with a real database.
# ---------------------------------------------------------------------------
PATIENT_LOG = []          # Recommendation Log / Patient Clinical Profile Store
ACTIVE_DETECTION_ALGO = "random_forest"       # Admin-controlled config flag
ACTIVE_RECOMMENDATION_ALGO = "random_forest"  # Admin-controlled config flag

# ---------------------------------------------------------------------------
# Model loading (Layer 3 / Layer 4 serialised model store)
# ---------------------------------------------------------------------------
def load_models():
    models = {}
    models["detection_cart"] = joblib.load(os.path.join(MODEL_DIR, "detection_cart.pkl"))
    models["detection_rf"] = joblib.load(os.path.join(MODEL_DIR, "detection_rf.pkl"))
    models["detection_scaler"] = joblib.load(os.path.join(MODEL_DIR, "detection_scaler.pkl"))
    with open(os.path.join(MODEL_DIR, "detection_medians.json")) as f:
        models["detection_medians"] = json.load(f)
    with open(os.path.join(MODEL_DIR, "detection_metrics.json")) as f:
        models["detection_metrics"] = json.load(f)
    with open(os.path.join(MODEL_DIR, "recommendation_metrics.json")) as f:
        models["recommendation_metrics"] = json.load(f)
    return models

MODELS = load_models()


def run_detection(patient_data: dict, algorithm: str):
    """Layer 3: preprocess + classify via the selected detection classifier."""
    row = []
    for col in DETECTION_FEATURES:
        val = float(patient_data[col])
        if col in ["glucose", "blood_pressure", "skin_thickness", "insulin", "bmi"] and val == 0:
            val = MODELS["detection_medians"][col]
        row.append(val)

    X = pd.DataFrame([row], columns=DETECTION_FEATURES)
    X_scaled = MODELS["detection_scaler"].transform(X)

    model = MODELS["detection_cart"] if algorithm == "cart" else MODELS["detection_rf"]
    prediction = int(model.predict(X_scaled)[0])
    probability = float(model.predict_proba(X_scaled)[0][1])

    return prediction, probability


# ---------------------------------------------------------------------------
# Patient-facing routes
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/patient")
def patient_interface():
    return render_template("patient.html")


@app.route("/api/detect", methods=["POST"])
def api_detect():
    """
    Full pipeline entry point: validate -> detect -> conditional gate ->
    recommend (only if positive).
    """
    data = request.get_json()

    errors = validate_patient_input(data)
    if errors:
        return jsonify({"error": "validation_failed", "details": errors}), 400

    prediction, probability = run_detection(data, ACTIVE_DETECTION_ALGO)

    result = {
        "detection": {
            "outcome": "positive" if prediction == 1 else "negative",
            "probability": round(probability, 4),
            "algorithm_used": ACTIVE_DETECTION_ALGO,
        },
        "recommendations": None,
    }

    # Conditional gate: recommendation stage only invoked on positive detection
    if prediction == 1:
        recs = get_recommendations(
            patient_glucose=float(data["glucose"]),
            patient_bmi=float(data["bmi"]),
            patient_age=float(data["age"]),
            algorithm=ACTIVE_RECOMMENDATION_ALGO,
            top_n=10,
        )
        result["recommendations"] = recs
        result["recommendation_algorithm_used"] = ACTIVE_RECOMMENDATION_ALGO

    # Log to Patient Clinical Profile Store / Recommendation Log
    log_entry = {
        "id": str(uuid.uuid4())[:8],
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "input": data,
        "detection_outcome": result["detection"]["outcome"],
        "detection_probability": result["detection"]["probability"],
        "num_recommendations": len(result["recommendations"]) if result["recommendations"] else 0,
        "overridden": False,
    }
    PATIENT_LOG.append(log_entry)
    result["log_id"] = log_entry["id"]

    return jsonify(result)


# ---------------------------------------------------------------------------
# Healthcare Worker UI
# ---------------------------------------------------------------------------
@app.route("/healthcare-worker")
def healthcare_worker_interface():
    return render_template("healthcare_worker.html")


@app.route("/api/patient-logs")
def api_patient_logs():
    return jsonify(list(reversed(PATIENT_LOG)))


@app.route("/api/override/<log_id>", methods=["POST"])
def api_override(log_id):
    """Healthcare Worker use case: override a detection outcome."""
    data = request.get_json()
    new_outcome = data.get("outcome")
    reason = data.get("reason", "")

    for entry in PATIENT_LOG:
        if entry["id"] == log_id:
            entry["overridden"] = True
            entry["original_outcome"] = entry["detection_outcome"]
            entry["detection_outcome"] = new_outcome
            entry["override_reason"] = reason
            entry["override_timestamp"] = datetime.now().isoformat(timespec="seconds")
            return jsonify({"status": "ok", "entry": entry})

    return jsonify({"error": "log_id not found"}), 404


@app.route("/api/food-database")
def api_food_database():
    foods = load_food_database()
    return jsonify(foods.to_dict("records"))


@app.route("/api/food-database/add", methods=["POST"])
def api_add_food():
    """Healthcare Worker use case: update the food nutrition database."""
    data = request.get_json()
    required = ["name", "category", "glycaemic_index", "carbohydrate_g",
                "fibre_g", "protein_g", "fat_g", "calories_kcal"]
    if not all(k in data for k in required):
        return jsonify({"error": "missing_fields", "required": required}), 400

    path = os.path.join(DATA_DIR, "nigerian_food_nutrition.csv")
    foods = pd.read_csv(path)
    new_row = pd.DataFrame([{k: data[k] for k in required}])
    foods = pd.concat([foods, new_row], ignore_index=True)
    foods.to_csv(path, index=False)

    return jsonify({"status": "ok", "total_foods": len(foods)})


# ---------------------------------------------------------------------------
# Admin Model Management Panel
# ---------------------------------------------------------------------------
@app.route("/admin")
def admin_panel():
    return render_template("admin.html")


@app.route("/api/admin/status")
def api_admin_status():
    return jsonify({
        "active_detection_algorithm": ACTIVE_DETECTION_ALGO,
        "active_recommendation_algorithm": ACTIVE_RECOMMENDATION_ALGO,
        "detection_metrics": MODELS["detection_metrics"],
        "recommendation_metrics": MODELS["recommendation_metrics"],
        "total_patients_logged": len(PATIENT_LOG),
        "food_database_size": len(load_food_database()),
    })


@app.route("/api/admin/set-algorithm", methods=["POST"])
def api_set_algorithm():
    """Admin use case: swap active classifier (CART or RF) at either stage independently."""
    global ACTIVE_DETECTION_ALGO, ACTIVE_RECOMMENDATION_ALGO
    data = request.get_json()
    stage = data.get("stage")
    algorithm = data.get("algorithm")

    if algorithm not in ("cart", "random_forest"):
        return jsonify({"error": "invalid algorithm"}), 400

    if stage == "detection":
        ACTIVE_DETECTION_ALGO = algorithm
    elif stage == "recommendation":
        ACTIVE_RECOMMENDATION_ALGO = algorithm
    else:
        return jsonify({"error": "invalid stage"}), 400

    return jsonify({
        "status": "ok",
        "active_detection_algorithm": ACTIVE_DETECTION_ALGO,
        "active_recommendation_algorithm": ACTIVE_RECOMMENDATION_ALGO,
    })


@app.route("/api/admin/retrain", methods=["POST"])
def api_retrain():
    """
    Admin use case: trigger model retraining. Runs the training routine
    in-process and reloads models.
    """
    global MODELS
    import subprocess
    result = subprocess.run(
        [sys.executable, os.path.join(BASE_DIR, "src", "train_models.py")],
        capture_output=True, text=True, cwd=os.path.join(BASE_DIR, "src")
    )
    if result.returncode != 0:
        return jsonify({"status": "error", "log": result.stderr}), 500

    MODELS = load_models()
    return jsonify({"status": "ok", "log": result.stdout})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
