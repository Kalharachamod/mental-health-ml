import re
import pandas as pd
import numpy as np
from typing import Optional
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from .config import COLUMNS

# -------- Helpers --------
def _clean_colnames(cols):
    return [re.sub(r"\s+|-|/", "_", (c or "").strip()) for c in cols]

_NUMBER_RE = re.compile(r"[-+]?\d*\.?\d+")

def _to_float(x) -> Optional[float]:
    if pd.isna(x): 
        return np.nan
    if isinstance(x, (int, float, np.integer, np.floating)): 
        return float(x)
    s = str(x).strip().replace(",", "")
    if re.match(r"^\s*\d+(\.\d+)?\s*[-–]\s*\d+(\.\d+)?\s*$", s):
        a, b = re.split(r"[-–]", s)
        return (float(a) + float(b)) / 2
    m = _NUMBER_RE.search(s)
    return float(m.group()) if m else np.nan

def _norm_yes_no(s):
    if pd.isna(s): 
        return np.nan
    s = str(s).strip().lower()
    mapping = {
        "y": "Yes", "yes": "Yes", "true": "Yes", "1": "Yes",
        "ye": "Yes", "yeah": "Yes", "yep": "Yes", "yees": "Yes", "yess": "Yes",
        "n": "No", "no": "No", "false": "No", "0": "No",
        "nah": "No", "nope": "No", "noo": "No"
    }
    return mapping.get(s, s.title())

def _norm_category(s):
    if pd.isna(s): 
        return np.nan
    s = str(s).strip().lower()
    mapping = {
        # Health
        "healthy": "Healthy", "avg": "Average", "average": "Average", "aveeragee": "Average",
        "unhealthy": "Unhealthy", "poor": "Unhealthy",
        # Stress / Severity
        "low": "Low", "medium": "Medium", "med": "Medium", "meedium": "Medium",
        "high": "High", "none": "None", "nonee": "None",
        "mild": "Low", "moderate": "Medium", "severe": "High",
        # Smoking
        "non-smoker": "Non-Smoker", "occasional": "Occasional Smoker",
        "regular": "Regular Smoker", "heavy": "Heavy Smoker",
        # Drinking
        "non-drinker": "Non-Drinker", "occasional drinker": "Occasional Drinker",
        "regular drinker": "Regular Drinker",
    }
    return mapping.get(s, s.title())

def _clip_ranges(df):
    ranges = {
        "Age": (10, 100), "Sleep_Hours": (0, 16), "Work_Hours": (0, 120),
        "Physical_Activity_Hours": (0, 40), "Social_Media_Usage": (0, 24)
    }
    for col, (lo, hi) in ranges.items():
        if col in df:
            df[col] = df[col].clip(lo, hi)
    return df

def _iqr_winsorize(s, k=3.0):
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    lo, hi = q1 - k * iqr, q3 + k * iqr
    return s.clip(lo, hi)

# -------- Public --------
def load_dataset(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path, on_bad_lines="skip")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin1", on_bad_lines="skip")

def preprocess(df: pd.DataFrame, save_cleaned=True) -> pd.DataFrame:
    raw_snapshot_path = "data/raw_dataset_snapshot.csv"
    cleaned_path = "data/raw_dataset_cleaned.csv"
    df.to_csv(raw_snapshot_path, index=False)

    df = df.copy()
    df.columns = _clean_colnames(df.columns)

    # Convert numerics
    for col in COLUMNS.numeric: 
        if col in df: 
            df[col] = df[col].map(_to_float)

    # Normalize Yes/No fields
    for col in ["Consultation_History", "Medication_Usage", "Mental_Health_Condition"]:
        if col in df:
            df[col] = df[col].map(_norm_yes_no).fillna("No").astype(str)

    # Normalize categorical
    for col in ["Gender","Occupation","Country","Diet_Quality",
                "Smoking_Habit","Alcohol_Consumption","Stress_Level","Severity"]:
        if col in df: 
            df[col] = df[col].map(_norm_category)

    # Fill missing
    for col in COLUMNS.numeric:
        if col in df: 
            df[col] = df[col].fillna(df[col].median())
    for col in COLUMNS.categorical:
        if col in df: 
            mode = df[col].mode()
            df[col] = df[col].fillna(mode.iloc[0] if not mode.empty else "Unknown")

    # Winsorize and clip ranges
    for col in COLUMNS.numeric:
        if col in df: 
            df[col] = _iqr_winsorize(df[col])
    df = _clip_ranges(df)

    # Ensure ordered categories
    if "Severity" in df:
        df["Severity"] = pd.Categorical(df["Severity"], 
                                        categories=COLUMNS.severity_order, 
                                        ordered=True)
    if "Stress_Level" in df:
        df["Stress_Level"] = pd.Categorical(df["Stress_Level"], 
                                            categories=COLUMNS.stress_order, 
                                            ordered=True)

    # Save cleaned dataset
    if save_cleaned: 
        df.to_csv(cleaned_path, index=False)

    return df

def make_transformer(df):
    num = [c for c in COLUMNS.numeric if c in df]
    cat = [c for c in COLUMNS.categorical if c in df]
    return ColumnTransformer([
        ("num", StandardScaler(), num),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat)
    ])

def stratified_split(X, y, test_size=0.2, random_state=42):
    strat = y if getattr(y, "nunique", lambda: 1000)() < 20 else None
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=strat)

if __name__ == "__main__":
    df = preprocess(load_dataset("data/dataset.csv"))
    print("Columns:", df.columns.tolist())
    print("\nUnique values:")
    for col in ["Mental_Health_Condition", "Stress_Level", "Severity"]:
        if col in df:
            print(f"{col}: {df[col].unique()}")
            if col in ["Severity", "Stress_Level"]:
                print("Categories:", df[col].cat.categories)
    print(df.head())
