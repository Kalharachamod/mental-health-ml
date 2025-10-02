# src/config.py

from dataclasses import dataclass, field
from typing import List, Dict

@dataclass
class Columns:
    # ID column
    id: str = "User_ID"

    # Targets
    target_condition: str = "Mental_Health_Condition"
    target_severity: str = "Severity"
    target_stress: str = "Stress_Level"

    # Numeric features
    numeric: List[str] = field(default_factory=lambda: [
        "Age",
        "Sleep_Hours",
        "Work_Hours",
        "Physical_Activity_Hours",
        "Social_Media_Usage",
    ])

    # Categorical features
    categorical: List[str] = field(default_factory=lambda: [
        "Gender",
        "Occupation",
        "Country",
        "Consultation_History",
        "Diet_Quality",
        "Smoking_Habit",
        "Alcohol_Consumption",
        "Medication_Usage",
    ])

    # Ordered categories
    severity_order: List[str] = field(default_factory=lambda: ["None", "Low", "Medium", "High"])
    stress_order: List[str] = field(default_factory=lambda: ["Low", "Medium", "High"])

    # Probability thresholds
    risk_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "low": 0.33,
        "high": 0.66,
    })


COLUMNS = Columns()


# -------- Helper to categorize risk --------
def risk_category(prob: float) -> str:
    if prob < 0.2:
        return "Very Low Risk"
    elif prob < 0.4:
        return "Low Risk"
    elif prob < 0.6:
        return "Moderate Risk"
    elif prob < 0.8:
        return "High Risk"
    else:
        return "Very High Risk"
