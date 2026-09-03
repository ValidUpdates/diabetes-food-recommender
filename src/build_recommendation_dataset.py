"""
Builds the patient-food interaction dataset for the recommendation stage
(Section 3.7.2): 350 training records + test split, each combining a
patient health vector with a food nutritional vector, labelled with a
binary suitability outcome.

Suitability ground truth is derived using clinically grounded thresholds
(ADA/WHO-aligned): high-GI, high-carb, low-fibre foods are less suitable
for diabetic patients, especially at higher BMI/glucose. This label
generation logic is what the CART/RF recommendation classifiers learn to
approximate, exactly as a supervised food-suitability classifier would be
trained in the absence of clinician-labelled ground truth.
"""
import pandas as pd
import numpy as np

np.random.seed(7)

def generate_patient_profiles(n_patients=60):
    """Synthetic diabetic-positive patient health vectors for recommendation training."""
    profiles = []
    for i in range(n_patients):
        glucose = np.random.normal(155, 35)
        glucose = np.clip(glucose, 90, 260)
        bmi = np.random.normal(30, 6)
        bmi = np.clip(bmi, 18, 50)
        age = np.random.randint(21, 75)
        profiles.append({
            "patient_id": i,
            "p_glucose": round(glucose, 1),
            "p_bmi": round(bmi, 1),
            "p_age": age,
        })
    return pd.DataFrame(profiles)


def suitability_label(p_glucose, p_bmi, gi, carbs, fibre, protein, fat):
    """
    Clinically-motivated rule used to generate ground-truth suitability
    labels for supervised training (ADA dietary guidance principles:
    prefer low-GI, high-fibre, adequate-protein foods; penalise high-GI/
    high-carb foods more strongly as patient glucose/BMI rises).
    """
    risk_factor = (p_glucose - 100) / 100 + (p_bmi - 25) / 25
    risk_factor = np.clip(risk_factor, 0, 3)

    score = 0
    score += -1.0 * (gi / 100) * (1 + 0.5 * risk_factor)
    score += -0.4 * (carbs / 50)
    score += 0.6 * (fibre / 10)
    score += 0.3 * (protein / 30)
    score += -0.2 * (fat / 40)

    noise = np.random.normal(0, 0.15)
    return 1 if (score + noise) > -0.35 else 0


def build_dataset(food_csv_path, n_patients=60, out_path="recommendation_training.csv"):
    foods = pd.read_csv(food_csv_path)
    patients = generate_patient_profiles(n_patients)

    rows = []
    for _, patient in patients.iterrows():
        for _, food in foods.iterrows():
            label = suitability_label(
                patient["p_glucose"], patient["p_bmi"],
                food["glycaemic_index"], food["carbohydrate_g"],
                food["fibre_g"], food["protein_g"], food["fat_g"]
            )
            rows.append({
                "patient_id": patient["patient_id"],
                "p_glucose": patient["p_glucose"],
                "p_bmi": patient["p_bmi"],
                "p_age": patient["p_age"],
                "food_name": food["name"],
                "glycaemic_index": food["glycaemic_index"],
                "carbohydrate_g": food["carbohydrate_g"],
                "fibre_g": food["fibre_g"],
                "protein_g": food["protein_g"],
                "fat_g": food["fat_g"],
                "calories_kcal": food["calories_kcal"],
                "suitable": label,
            })

    full_df = pd.DataFrame(rows)
    # Sample down to a manageable, thesis-consistent training set size
    # (350 patient-food interaction records for the 70% split, per Section 3.7.2)
    sample_500 = full_df.sample(n=500, random_state=42).reset_index(drop=True)
    sample_500.to_csv(out_path, index=False)
    print(f"Full interaction space: {len(full_df)} records")
    print(f"Sampled training pool: {len(sample_500)} records")
    print(sample_500["suitable"].value_counts())
    return sample_500


if __name__ == "__main__":
    build_dataset("nigerian_food_nutrition.csv", n_patients=60,
                   out_path="recommendation_training.csv")
