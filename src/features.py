"""Feature construction for AahaarTwin (Phase 2).

Documented feature list — every feature below is built ONLY from information
available at or before meal onset (no post-meal leakage), plus static
patient variables from bio.csv.

Feature groups:
  MEAL   : carbs, protein, fat, fiber, calories            (from the meal log)
  PRE    : g0, slope_60min, hr_mean_2h, act_kcal_2h        (pre-meal CGM/wearable context)
  TIME   : hour_sin, hour_cos, meal_type_{breakfast,lunch,dinner,snack}
           (hour-of-day is preserved under the dataset's day-shift anonymization)
  STATIC : age, gender_m, bmi, hba1c, fasting_glu, fasting_insulin,
           trig, chol, hdl, ldl                            (bio.csv, one row per patient)

Missing values are median-imputed with medians fit on TRAINING data only
(MedianImputer below); the imputer is part of the saved pipeline.
"""
import numpy as np
import pandas as pd

MEAL_FEATURES = ["carbs", "protein", "fat", "fiber", "calories"]
PRE_FEATURES = ["g0", "slope_60min", "hr_mean_2h", "act_kcal_2h"]
TIME_FEATURES = ["hour_sin", "hour_cos", "meal_type_breakfast", "meal_type_lunch",
                 "meal_type_dinner", "meal_type_snack"]
STATIC_FEATURES = ["age", "gender_m", "bmi", "hba1c", "fasting_glu", "fasting_insulin",
                   "trig", "chol", "hdl", "ldl"]

FEATURES = MEAL_FEATURES + PRE_FEATURES + TIME_FEATURES + STATIC_FEATURES

FEATURE_DOCS = {
    "carbs": "meal carbohydrates, grams (MyFitnessPal log)",
    "protein": "meal protein, grams",
    "fat": "meal fat, grams",
    "fiber": "meal fiber, grams",
    "calories": "meal calories, kcal (estimated)",
    "g0": "pre-meal glucose, mg/dL (nearest Dexcom reading within 15 min)",
    "slope_60min": "glucose trend over prior 60 min, mg/dL/min",
    "hr_mean_2h": "mean heart rate over prior 2 h, bpm (Fitbit)",
    "act_kcal_2h": "activity calories over prior 2 h (Fitbit)",
    "hour_sin": "sin(2*pi*hour/24) — circadian encoding",
    "hour_cos": "cos(2*pi*hour/24) — circadian encoding",
    "meal_type_breakfast": "one-hot meal type",
    "meal_type_lunch": "one-hot meal type",
    "meal_type_dinner": "one-hot meal type",
    "meal_type_snack": "one-hot meal type",
    "age": "years (bio.csv)",
    "gender_m": "1 if male else 0 (bio.csv)",
    "bmi": "kg/m^2 (bio.csv)",
    "hba1c": "glycated hemoglobin, % (bio.csv)",
    "fasting_glu": "fasting glucose, mg/dL (bio.csv)",
    "fasting_insulin": "fasting insulin, mcU/mL (bio.csv)",
    "trig": "triglycerides, mg/dL (bio.csv)",
    "chol": "total cholesterol, mg/dL (bio.csv)",
    "hdl": "HDL cholesterol, mg/dL (bio.csv)",
    "ldl": "LDL cholesterol, mg/dL (bio.csv, calculated)",
}

# bio.csv column mapping (after whitespace strip)
BIO_MAP = {
    "Age": "age", "Gender": "gender", "BMI": "bmi", "A1c PDL (Lab)": "hba1c",
    "Fasting GLU - PDL (Lab)": "fasting_glu", "Insulin": "fasting_insulin",
    "Triglycerides": "trig", "Cholesterol": "chol", "HDL": "hdl", "LDL (Cal)": "ldl",
}


def load_bio(bio_csv_path):
    """Load static patient variables, indexed by pid."""
    bio = pd.read_csv(bio_csv_path, low_memory=False)
    bio.columns = bio.columns.str.strip()
    out = pd.DataFrame({"pid": bio["subject"].astype(int)})
    for src, dst in BIO_MAP.items():
        out[dst] = pd.to_numeric(bio[src], errors="coerce")
    out["gender_m"] = (bio["Gender"].str.strip().str.upper() == "M").astype(int)
    return out.set_index("pid")


def build_feature_frame(meals, bio):
    """meals: usable meal table with ts parsed. Returns X (DataFrame, FEATURES order)."""
    df = meals.copy()
    df["hour"] = df["ts"].dt.hour + df["ts"].dt.minute / 60.0
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)
    for mt in ["breakfast", "lunch", "dinner", "snack"]:
        df[f"meal_type_{mt}"] = (df["meal_type"] == mt).astype(int)
    df = df.join(bio, on="pid")
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise KeyError(f"features missing after build: {missing}")
    return df[FEATURES].copy()


class MedianImputer:
    """Median imputation with medians fit on training data only."""

    def __init__(self):
        self.medians_ = None

    def fit(self, X):
        self.medians_ = X.median(numeric_only=True)
        return self

    def transform(self, X):
        return X.fillna(self.medians_)

    def fit_transform(self, X):
        return self.fit(X).transform(X)
