import joblib
import pandas as pd
import numpy as np
from src.data import _to_float, _norm_category, _norm_yes_no, _clip_ranges
from src.config import COLUMNS

# Load models
condition_model = joblib.load("models/condition_model.pkl")
stress_model = joblib.load("models/stress_model.pkl")
severity_model = joblib.load("models/severity_model.pkl")

condition_encoder = joblib.load("models/condition_label_encoder.pkl")
stress_encoder = joblib.load("models/stress_encoder.pkl")
severity_encoder = joblib.load("models/severity_encoder.pkl")

print("Model information:")
print("Condition model classes:", condition_model.classes_)
print("Stress encoder classes:", stress_encoder.classes_)
print("Severity encoder classes:", severity_encoder.classes_)
print()

def preprocess_input_data(data):
    """Preprocess input data to match the format expected by the models"""
    # Create a DataFrame from the input data with correct column names
    mapped_data = {
        "Age": data["Age"],
        "Sleep_Hours": data["Sleep_Hours"],
        "Work_Hours": data["Work_Hours"],
        "Physical_Activity_Hours": data["Physical_Activity_Hours"],
        "Social_Media_Usage": data["Social_Media_Usage"],
        "Gender": data["Gender"],
        "Occupation": data["Occupation"],
        "Country": data["Country"],
        "Consultation_History": data["Consultation_History"],
        "Diet_Quality": data["Diet_Quality"],
        "Smoking_Habit": data["Smoking_Habit"],
        "Alcohol_Consumption": data["Alcohol_Consumption"],
        "Medication_Usage": data["Medication_Usage"],
    }
    
    # Create a DataFrame from the mapped data
    df = pd.DataFrame([mapped_data])
    
    # Apply the same preprocessing as in the training pipeline
    # Convert numerics
    for col in COLUMNS.numeric: 
        if col in df: 
            df[col] = df[col].map(_to_float)
    
    # Normalize Yes/No fields
    for col in ["Consultation_History", "Medication_Usage"]:
        if col in df:
            df[col] = df[col].map(_norm_yes_no).fillna("No").astype(str)
    
    # Normalize categorical fields
    categorical_fields = ["Gender", "Occupation", "Country", "Diet_Quality", 
                         "Smoking_Habit", "Alcohol_Consumption"]
    for col in categorical_fields:
        if col in df: 
            df[col] = df[col].map(_norm_category)
    
    # Clip ranges
    df = _clip_ranges(df)
    
    return df

def predict_mental_health(data):
    """Make predictions using the loaded models"""
    # Preprocess the input data
    input_df = preprocess_input_data(data)
    
    print("Input DataFrame:")
    print(input_df)
    print()
    
    # Make predictions
    cond_pred = condition_model.predict(input_df)[0]
    cond_probs = condition_model.predict_proba(input_df)[0]
    cond_prob = float(max(cond_probs))
    
    stress_pred = stress_model.predict(input_df)[0]
    sev_pred = severity_model.predict(input_df)[0]
    
    print("Raw predictions:")
    print("Condition prediction:", cond_pred, type(cond_pred))
    print("Condition probabilities:", cond_probs)
    print("Stress prediction:", stress_pred, type(stress_pred))
    print("Severity prediction:", sev_pred, type(sev_pred))
    print()
    
    # Decode predictions
    stress_label = stress_encoder.inverse_transform([int(stress_pred)])[0]
    severity_label = severity_encoder.inverse_transform([int(sev_pred)])[0]
    
    return {
        'condition': str(cond_pred),
        'condition_confidence': f"{cond_prob:.1%}",
        'stress_level': str(stress_label),
        'severity': str(severity_label)
    }

# Test different scenarios
print("Testing different input scenarios to see prediction changes:\n")

# Scenario 1: Healthy lifestyle
healthy_data = {
    "Age": 25,
    "Sleep_Hours": 8.0,
    "Work_Hours": 40.0,
    "Physical_Activity_Hours": 7.0,
    "Social_Media_Usage": 2.0,
    "Gender": "Female",
    "Occupation": "IT",
    "Country": "USA",
    "Consultation_History": "No",
    "Diet_Quality": "Healthy",
    "Smoking_Habit": "Non-Smoker",
    "Alcohol_Consumption": "Non-Drinker",
    "Medication_Usage": "No",
}

print("Scenario 1 - Healthy Lifestyle:")
result1 = predict_mental_health(healthy_data)
for key, value in result1.items():
    print(f"  {key}: {value}")
print()

# Scenario 2: Unhealthy lifestyle with high stress indicators
unhealthy_data = {
    "Age": 35,
    "Sleep_Hours": 4.0,
    "Work_Hours": 70.0,
    "Physical_Activity_Hours": 1.0,
    "Social_Media_Usage": 8.0,
    "Gender": "Male",
    "Occupation": "Sales",
    "Country": "UK",
    "Consultation_History": "No",
    "Diet_Quality": "Unhealthy",
    "Smoking_Habit": "Heavy Smoker",
    "Alcohol_Consumption": "Regular Drinker",
    "Medication_Usage": "Yes",
}

print("Scenario 2 - High Risk Lifestyle:")
result2 = predict_mental_health(unhealthy_data)
for key, value in result2.items():
    print(f"  {key}: {value}")
print()

# Scenario 3: Moderate risk
moderate_data = {
    "Age": 45,
    "Sleep_Hours": 6.0,
    "Work_Hours": 50.0,
    "Physical_Activity_Hours": 3.0,
    "Social_Media_Usage": 4.0,
    "Gender": "Non-Binary",
    "Occupation": "Healthcare",
    "Country": "Canada",
    "Consultation_History": "Yes",
    "Diet_Quality": "Average",
    "Smoking_Habit": "Occasional Smoker",
    "Alcohol_Consumption": "Social Drinker",
    "Medication_Usage": "No",
}

print("Scenario 3 - Moderate Risk Lifestyle:")
result3 = predict_mental_health(moderate_data)
for key, value in result3.items():
    print(f"  {key}: {value}")
print()