from sklearn.ensemble import IsolationForest

def fit_anomaly_model(df_numeric, random_state=42):
    iso = IsolationForest(n_estimators=200, contamination="auto", random_state=random_state)
    iso.fit(df_numeric)
    return iso

def anomaly_flag(model, row_numeric, threshold=-0.25):
    score = model.decision_function([row_numeric.values])[0]
    return float(score), bool(score < threshold)
