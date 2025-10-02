import pandas as pd
import numpy as np
from src.data import preprocess

# Load the dataset
df = pd.read_csv('data/dataset.csv', encoding='latin1')

print("Dataset shape:", df.shape)
print("\nColumn names:")
for col in df.columns:
    print(f"  {col}")

print("\nTarget value distributions:")
print("\nMental_Health_Condition:")
print(df['Mental_Health_Condition'].value_counts())

print("\nStress_Level:")
print(df['Stress_Level'].value_counts())

print("\nSeverity:")
print(df['Severity'].value_counts())

# Preprocess the data to see clean values
print("\n\nAfter preprocessing:")
clean_df = preprocess(df, save_cleaned=False)

print("\nClean Mental_Health_Condition:")
print(clean_df['Mental_Health_Condition'].value_counts())

print("\nClean Stress_Level:")
print(clean_df['Stress_Level'].value_counts())

print("\nClean Severity:")
print(clean_df['Severity'].value_counts())

print("\nNumeric column statistics (after preprocessing):")
numeric_cols = ['Age', 'Sleep_Hours', 'Work_Hours', 'Physical_Activity_Hours', 'Social_Media_Usage']
for col in numeric_cols:
    if col in clean_df.columns:
        print(f"\n{col}:")
        print(f"  Min: {clean_df[col].min()}")
        print(f"  Max: {clean_df[col].max()}")
        print(f"  Mean: {clean_df[col].mean():.2f}")
        print(f"  Std: {clean_df[col].std():.2f}")

print("\nCategorical column value counts (after preprocessing):")
categorical_cols = ['Gender', 'Occupation', 'Country', 'Diet_Quality', 'Smoking_Habit', 'Alcohol_Consumption']
for col in categorical_cols:
    if col in clean_df.columns:
        print(f"\n{col}:")
        print(clean_df[col].value_counts())