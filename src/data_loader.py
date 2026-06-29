"""
data_loader.py
----------------
Loads the raw student lifestyle dataset, cleans it, and saves a
processed version to data/processed/ so training scripts never need
to repeat the cleaning steps.

Run this file directly to generate the processed CSV:
    python src/data_loader.py
"""

import numpy as np
import pandas as pd

RAW_PATH = "data/raw/student-lifestyle-and-stress-dataset.csv"
PROCESSED_PATH = "data/processed/student_data_clean.csv"


def load_raw_data(path: str = RAW_PATH) -> pd.DataFrame:
    """Load the untouched dataset exactly as downloaded."""
    df = pd.read_csv(path)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all cleaning steps from the original EDA notebook:
    - remove duplicates
    - fix invalid Study_Hours / Attendance values
    - fill missing values (mean/median/mode depending on column)
    - encode Student_Type
    """
    df = df.copy()

    # 1. Remove duplicate rows
    df.drop_duplicates(inplace=True)

    # 2. Fix invalid values by converting them to NaN first
    # Study_Hours can't be negative
    df.loc[df["Study_Hours"] < 0, "Study_Hours"] = np.nan

    # Attendance must be between 0 and 100
    df.loc[df["Attendance"] < 0, "Attendance"] = np.nan
    df.loc[df["Attendance"] > 100, "Attendance"] = np.nan

    # 3. Fill missing values
    # Categorical -> mode
    df["Student_Type"] = df["Student_Type"].fillna(df["Student_Type"].mode()[0])
    df["Month"] = df["Month"].fillna(df["Month"].mode()[0])

    # Numerical -> mean or median depending on distribution
    df["Sleep_Hours"] = df["Sleep_Hours"].fillna(df["Sleep_Hours"].mean())
    df["Study_Hours"] = df["Study_Hours"].fillna(df["Study_Hours"].median())
    df["Social_Media_Hours"] = df["Social_Media_Hours"].fillna(
        df["Social_Media_Hours"].mean()
    )
    df["Attendance"] = df["Attendance"].fillna(df["Attendance"].median())
    df["Exam_Pressure"] = df["Exam_Pressure"].fillna(df["Exam_Pressure"].mean())
    df["Family_Support"] = df["Family_Support"].fillna(df["Family_Support"].mean())

    # 4. Encode Student_Type (preserves a meaningful order)
    lifestyle_mapping = {
        "school": 1,
        "college": 2,
        "working_student": 3,
    }
    df["Student_Type_Encoded"] = df["Student_Type"].map(lifestyle_mapping)
    df.drop("Student_Type", axis=1, inplace=True)

    return df


def load_processed_data(path: str = PROCESSED_PATH) -> pd.DataFrame:
    """Load the already-cleaned data (use this in train.py)."""
    return pd.read_csv(path)


def run_pipeline():
    """Load raw -> clean -> save processed. Run once whenever raw data changes."""
    df = load_raw_data()
    df_clean = clean_data(df)
    df_clean.to_csv(PROCESSED_PATH, index=False)
    print(f"Processed data saved to {PROCESSED_PATH}")
    print(f"Shape: {df_clean.shape}")
    return df_clean


if __name__ == "__main__":
    run_pipeline()
