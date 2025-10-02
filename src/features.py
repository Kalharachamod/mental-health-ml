import pandas as pd
from src.config import COLUMNS

def build_xy(df: pd.DataFrame, target: str):
    """
    Builds features (X) and target (y) for a given target column.
    Ensures that no target columns leak into features.
    """
    # Define all targets
    targets = [
        COLUMNS.target_condition,
        COLUMNS.target_stress,
        COLUMNS.target_severity,
    ]

    # Start from all numeric + categorical + target column
    keep = set([COLUMNS.id] + COLUMNS.numeric + COLUMNS.categorical + [target])
    cols = [c for c in df.columns if c in keep]
    d = df[cols].copy()

    # Drop ID and all targets from X
    drop_cols = [COLUMNS.id] + targets
    X = d.drop(columns=[c for c in drop_cols if c in d])
    y = d[target]

    return X, y
