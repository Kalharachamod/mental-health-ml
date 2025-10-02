import shap

def make_explainer(pipeline, X_background):
    clf = pipeline.named_steps["clf"]
    pre = pipeline.named_steps["preprocess"]
    X_bg = pre.transform(X_background.sample(min(500, len(X_background)), random_state=42))
    try:
        return shap.TreeExplainer(clf)
    except Exception:
        return shap.LinearExplainer(clf, X_bg)

def shap_for_instance(explainer, pipeline, X_row):
    pre = pipeline.named_steps["preprocess"]
    Xtr = pre.transform(X_row)
    sv = explainer.shap_values(Xtr)
    if isinstance(sv, list) and len(sv) == 2:
        return sv[1][0]  # class 1 for binary
    return sv[0] if hasattr(sv, "__len__") else sv
