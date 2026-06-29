"""
train.py
----------------
Loads the processed dataset, trains models, evaluates them, and logs
every run to experiments/results.csv so you can track performance
over time across different algorithms and settings.

Run with:
    python src/train.py
"""

import os
import pickle
from datetime import date

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
import pandas as pd

from data_loader import load_processed_data

MODEL_DIR = "models"
RESULTS_LOG = "experiments/results.csv"


def split_data(df, target_col="Stress_Level", test_size=0.2, random_state=42):
    x = df.drop([target_col], axis=1)
    y = df[target_col]
    return train_test_split(x, y, test_size=test_size, random_state=random_state)


def train_knn(x_train, y_train, n_neighbors=5):
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(x_train, y_train)
    return model


def train_logistic_regression(x_train, y_train):
    model = LogisticRegression(max_iter=1000)
    model.fit(x_train, y_train)
    return model


def train_random_forest(x_train, y_train, n_estimators=100, max_depth=None):
    model = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth, random_state=42
    )
    model.fit(x_train, y_train)
    return model


def evaluate_model(model, x_test, y_test, name="Model"):
    y_pred = model.predict(x_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    print(f"{name} -> Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f}")
    return {"accuracy": acc, "precision": prec, "recall": rec}


def log_result(model_name, metrics, notes=""):
    """Append one row to experiments/results.csv. Creates the file if it doesn't exist yet."""
    os.makedirs("experiments", exist_ok=True)

    new_row = {
        "model_name": model_name,
        "accuracy": round(metrics["accuracy"], 4),
        "precision": round(metrics["precision"], 4),
        "recall": round(metrics["recall"], 4),
        "date_run": date.today().isoformat(),
        "notes": notes,
    }

    if os.path.exists(RESULTS_LOG):
        df_log = pd.read_csv(RESULTS_LOG)
        df_log = pd.concat([df_log, pd.DataFrame([new_row])], ignore_index=True)
    else:
        df_log = pd.DataFrame([new_row])

    df_log.to_csv(RESULTS_LOG, index=False)


def save_model(model, filename):
    path = f"{MODEL_DIR}/{filename}"
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved to {path}")


def main():
    df = load_processed_data()
    x_train, x_test, y_train, y_test = split_data(df)

    # KNN
    knn_model = train_knn(x_train, y_train)
    knn_metrics = evaluate_model(knn_model, x_test, y_test, name="KNN")
    log_result("KNN", knn_metrics, notes="default n_neighbors=5")

    # Logistic Regression
    log_model = train_logistic_regression(x_train, y_train)
    log_metrics = evaluate_model(log_model, x_test, y_test, name="Logistic Regression")
    log_result("Logistic Regression", log_metrics, notes="default params")

    # Random Forest
    rf_model = train_random_forest(x_train, y_train, n_estimators=100)
    rf_metrics = evaluate_model(rf_model, x_test, y_test, name="Random Forest")
    log_result("Random Forest", rf_metrics, notes="n_estimators=100")

    # Save the best model — update this after comparing experiments/results.csv
    save_model(rf_model, "stress_model_v2.pkl")

    print(f"\nAll results logged to {RESULTS_LOG}")


if __name__ == "__main__":
    main()