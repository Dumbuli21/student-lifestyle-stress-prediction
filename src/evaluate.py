"""
evaluate.py
----------------
More detailed evaluation than a single accuracy score - useful once
you want to understand WHERE your model is going wrong (which stress
levels it confuses, precision/recall per class, etc).

Run with:
    python src/evaluate.py
"""

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
)
import matplotlib.pyplot as plt
import seaborn as sns


def full_report(y_test, y_pred, model_name="Model"):
    print(f"--- {model_name} ---")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))


def plot_confusion_matrix(y_test, y_pred, model_name="Model", save_path=None):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"{model_name} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    if save_path:
        plt.savefig(save_path)
        print(f"Saved to {save_path}")
    plt.show()


if __name__ == "__main__":
    print("Import the functions in this file from train.py once you have")
    print("y_test and y_pred available, e.g.:")
    print("  from evaluate import full_report, plot_confusion_matrix")
    print("  full_report(y_test, y_pred, 'Logistic Regression')")
