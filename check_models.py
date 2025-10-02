import joblib

# Load models
condition_model = joblib.load("models/condition_model.pkl")
stress_model = joblib.load("models/stress_model.pkl")
severity_model = joblib.load("models/severity_model.pkl")

stress_encoder = joblib.load("models/stress_encoder.pkl")
severity_encoder = joblib.load("models/severity_encoder.pkl")

print("=== CONDITION MODEL ===")
print("Model type:", type(condition_model.named_steps['clf']))
print("Model details:", condition_model.named_steps['clf'])
print()

print("=== STRESS MODEL ===")
print("Model type:", type(stress_model.named_steps['clf']))
print("Model details:", stress_model.named_steps['clf'])
print()

print("=== SEVERITY MODEL ===")
print("Model type:", type(severity_model.named_steps['clf']))
print("Model details:", severity_model.named_steps['clf'])
print()

print("=== ENCODERS ===")
print("Stress encoder classes:", stress_encoder.classes_)
print("Severity encoder classes:", severity_encoder.classes_)