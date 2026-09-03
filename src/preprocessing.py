"""
Layer 2 - Preprocessing Pipeline (Section 3.5, Section 3.7.1)
Two parallel branches: detection preprocessing and recommendation preprocessing.
Each branch: Input Validator -> Imputer -> Scaler -> Encoder.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os

DETECTION_ZERO_AS_MISSING = ["glucose", "blood_pressure", "skin_thickness", "insulin", "bmi"]
DETECTION_FEATURES = ["pregnancies", "glucose", "blood_pressure", "skin_thickness",
                       "insulin", "bmi", "diabetes_pedigree", "age"]


def load_and_clean_pima(path):
    """
    Section 3.5, Step 1: Missing value treatment.
    Biologically implausible zeros in glucose/BP/skin/insulin/BMI are treated
    as missing and imputed with the median, stratified by outcome class.
    """
    df = pd.read_csv(path)

    for col in DETECTION_ZERO_AS_MISSING:
        df[col] = df[col].replace(0, np.nan)

    for outcome_val in [0, 1]:
        mask = df["outcome"] == outcome_val
        for col in DETECTION_ZERO_AS_MISSING:
            median_val = df.loc[mask, col].median()
            df.loc[mask & df[col].isna(), col] = median_val

    return df


def build_detection_pipeline(df, out_dir):
    """
    Section 3.5, Steps 2-3: Feature scaling (StandardScaler, z = (x-mu)/sigma)
    and 70:30 stratified train/test split.
    """
    X = df[DETECTION_FEATURES].copy()
    y = df["outcome"].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    os.makedirs(out_dir, exist_ok=True)
    joblib.dump(scaler, os.path.join(out_dir, "detection_scaler.pkl"))

    return {
        "X_train": X_train_scaled, "X_test": X_test_scaled,
        "y_train": y_train.values, "y_test": y_test.values,
        "X_train_raw": X_train, "X_test_raw": X_test,
        "feature_names": DETECTION_FEATURES,
    }


def validate_patient_input(data: dict):
    """
    Layer 2 Input Validator: checks required fields present and within
    acceptable numeric range before further processing.
    """
    required = DETECTION_FEATURES
    ranges = {
        "pregnancies": (0, 20),
        "glucose": (0, 300),
        "blood_pressure": (0, 200),
        "skin_thickness": (0, 100),
        "insulin": (0, 900),
        "bmi": (0, 80),
        "diabetes_pedigree": (0, 3),
        "age": (1, 120),
    }
    errors = []
    for field in required:
        if field not in data or data[field] in (None, ""):
            errors.append(f"Missing required field: {field}")
            continue
        try:
            val = float(data[field])
        except (ValueError, TypeError):
            errors.append(f"Field {field} must be numeric")
            continue
        lo, hi = ranges[field]
        if not (lo <= val <= hi):
            errors.append(f"Field {field} out of acceptable range ({lo}-{hi})")
    return errors


def impute_and_scale_single(raw_dict, scaler, medians):
    """
    Applies the same imputation logic (zero -> median) and scaling
    to a single new patient record at inference time.
    """
    row = []
    for col in DETECTION_FEATURES:
        val = float(raw_dict[col])
        if col in DETECTION_ZERO_AS_MISSING and val == 0:
            val = medians[col]
        row.append(val)
    row_arr = np.array(row).reshape(1, -1)
    return scaler.transform(row_arr)
