import joblib
from sklearn.model_selection import cross_val_score

def compare_models(build_fn, X, y, cv=5, scoring="f1_weighted"):
    """
    Compare different candidate models and return the best.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from xgboost import XGBClassifier

    models = {
        "logreg": LogisticRegression(max_iter=1000),
        "rf": RandomForestClassifier(n_estimators=200, random_state=42),
        "xgb": XGBClassifier(eval_metric="mlogloss", use_label_encoder=False)
    }

    results = {}
    best_model, best_score = None, -1

    for name, model in models.items():
        try:
            scores = cross_val_score(build_fn(model, X, y), X, y, cv=cv, scoring=scoring)
            mean_score = scores.mean()
            results[name] = mean_score
            print(f"{name}: {mean_score:.4f}")
            if mean_score > best_score:
                best_score = mean_score
                best_model = build_fn(model, X, y)
        except Exception as e:
            print(f"Model {name} failed: {e}")

    return {"results": results, "best_model": best_model, "best_score": best_score}


def save_model(model, path: str):
    """
    Save the trained model pipeline to disk.
    """
    joblib.dump(model, path)
    print(f"✅ Model saved to {path}")


def load_model(path: str):
    """
    Load a trained model pipeline from disk.
    """
    return joblib.load(path)
