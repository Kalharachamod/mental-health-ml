import os, json
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

def evaluate_and_save(model, X_test, y_test, out_json):
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred).tolist()
    metrics = {"classification_report": report, "confusion_matrix": cm}

    try:
        if len(set(y_test)) == 2 and hasattr(model, "predict_proba"):
            metrics["roc_auc"] = float(roc_auc_score(y_test, model.predict_proba(X_test)[:,1]))
    except Exception:
        pass

    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(metrics, f, indent=2)
    return metrics
