import sys
import os
# Add the project root to the path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import joblib
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from src.data import load_dataset, preprocess, make_transformer
from src.config import COLUMNS

def evaluate_algorithms(X, y, transformer):
    """Evaluate multiple algorithms using 5-fold cross-validation"""
    # Define algorithms to evaluate
    algorithms = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Random Forest': RandomForestClassifier(random_state=42),
        'SVM': SVC(random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'XGBoost': XGBClassifier(
            eval_metric="mlogloss",
            use_label_encoder=False,
            random_state=42
        )
    }
    
    # Evaluate each algorithm
    results = {}
    print("🔍 Evaluating algorithms with 5-fold cross-validation...")
    
    for name, algorithm in algorithms.items():
        try:
            # Create pipeline with preprocessing
            pipeline = Pipeline([
                ("prep", transformer),
                ("clf", algorithm)
            ])
            
            # Perform 5-fold cross-validation
            cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring='accuracy')
            mean_score = np.mean(cv_scores)
            std_score = np.std(cv_scores)
            
            results[name] = {
                'mean_score': mean_score,
                'std_score': std_score,
                'pipeline': pipeline
            }
            
            print(f"  {name}: {mean_score:.4f} (+/- {std_score*2:.4f})")
        except Exception as e:
            print(f"  {name}: Error - {str(e)}")
            results[name] = {'mean_score': 0, 'std_score': 0, 'pipeline': None}
    
    return results

def main():
    # Load and preprocess
    df = preprocess(load_dataset("data/dataset.csv"))
    X = df.drop(columns=[COLUMNS.target_stress, COLUMNS.target_severity, COLUMNS.target_condition], errors="ignore")
    y = df[COLUMNS.target_stress]

    # Encode target
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    # Build pipeline
    transformer = make_transformer(X)

    # Evaluate multiple algorithms
    results = evaluate_algorithms(X, y_enc, transformer)
    
    # Select best algorithm
    best_algorithm = max(results.keys(), key=lambda k: results[k]['mean_score'])
    best_score = results[best_algorithm]['mean_score']
    
    print(f"\n🏆 Best Algorithm: {best_algorithm} (Accuracy: {best_score:.4f})")
    
    # Hyperparameter tuning for the best algorithm
    if best_algorithm == 'Logistic Regression':
        param_grid = {
            'clf__C': [0.1, 1, 10],
            'clf__penalty': ['l1', 'l2'],
            'clf__solver': ['liblinear']
        }
    elif best_algorithm == 'Random Forest':
        param_grid = {
            'clf__n_estimators': [100, 300, 500],
            'clf__max_depth': [3, 5, 7, None],
            'clf__min_samples_split': [2, 5, 10]
        }
    elif best_algorithm == 'SVM':
        param_grid = {
            'clf__C': [0.1, 1, 10],
            'clf__kernel': ['rbf', 'linear'],
            'clf__gamma': ['scale', 'auto']
        }
    elif best_algorithm == 'Gradient Boosting':
        param_grid = {
            'clf__n_estimators': [100, 300, 500],
            'clf__max_depth': [3, 5, 7],
            'clf__learning_rate': [0.01, 0.1, 0.2]
        }
    elif best_algorithm == 'XGBoost':
        param_grid = {
            "clf__learning_rate": [0.01, 0.1, 0.2],
            "clf__max_depth": [3, 5, 7],
            "clf__n_estimators": [100, 300, 500],
            "clf__subsample": [0.7, 1.0],
        }
    
    # Create pipeline for best algorithm
    best_pipeline = Pipeline([
        ("prep", transformer),
        ("clf", results[best_algorithm]['pipeline'].named_steps['clf'])
    ])
    
    print(f"\n🔍 Running hyperparameter tuning for {best_algorithm}...")
    grid = GridSearchCV(best_pipeline, param_grid, cv=5, scoring="accuracy", n_jobs=-1)
    grid.fit(X, y_enc)

    print(f"✅ Best Score: {grid.best_score_:.4f}")
    print(f"✅ Best Params: {grid.best_params_}")

    # Save model + encoder
    joblib.dump(grid.best_estimator_, "models/stress_model.pkl")
    joblib.dump(le, "models/stress_encoder.pkl")
    print("💾 Model saved to models/stress_model.pkl")
    print("💾 Encoder saved to models/stress_encoder.pkl")

if __name__ == "__main__":
    main()