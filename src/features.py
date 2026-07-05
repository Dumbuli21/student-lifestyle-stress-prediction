"""
features.py
----------------
Feature engineering and transformation logic. This runs AFTER
data_loader.py (which only fixes/cleans raw data) and BEFORE train.py
(which trains models on the final feature set).
"""

from sklearn.preprocessing import StandardScaler

SCALE_FEATURES = [
    "Sleep_Hours",
    "Study_Hours",
    "Social_Media_Hours",
    "Attendance",
    "Exam_Pressure",
    "Family_Support",
]


def scale_features(df, feature_cols=None, scaler=None):
    """
    Scales numeric features using StandardScaler.
    Returns both the transformed dataframe AND the fitted scaler.
    """
    df = df.copy()
    feature_cols = feature_cols or SCALE_FEATURES

    if scaler is None:
        scaler = StandardScaler()
        df[feature_cols] = scaler.fit_transform(df[feature_cols])
    else:
        df[feature_cols] = scaler.transform(df[feature_cols])

    return df, scaler


def build_features(df):
    """Main entry point: apply all feature engineering steps in order."""
    df, scaler = scale_features(df)

    # Example of a future addition:
    # df['Study_Sleep_Balance'] = df['Study_Hours'] - df['Sleep_Hours']

    return df, scaler


def save_scaled_data(df, path="data/processed/student_data_scaled.csv"):
    """Save the scaled dataframe so you can actually SEE the result as a file."""
    df.to_csv(path, index=False)
    print(f"Scaled data saved to {path}")


if __name__ == "__main__":
    from data_loader import load_processed_data

    print("BEFORE scaling:")
    df = load_processed_data()
    print(df[SCALE_FEATURES].head())

    df_scaled, scaler = build_features(df)

    print("\nAFTER scaling:")
    print(df_scaled[SCALE_FEATURES].head())

    save_scaled_data(df_scaled)