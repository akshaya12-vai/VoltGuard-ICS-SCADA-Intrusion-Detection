from pathlib import Path
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_FILE = DATA_DIR / "train_data.csv"
TEST_FILE = DATA_DIR / "test_data.csv"

FEATURES = [
    "Tank_Water_Level",
    "Pressure_Sensor",
    "Flow_Rate",
    "Valve_Status"
]
TARGET = "Attack_Label"

print("[+] Loading datasets...")
train_df = pd.read_csv(TRAIN_FILE)
test_df = pd.read_csv(TEST_FILE)

required_columns = FEATURES + [TARGET]

for name, df in [("Training", train_df), ("Testing", test_df)]:
    missing = set(required_columns) - set(df.columns)
    if missing:
        raise ValueError(f"{name} dataset missing columns: {sorted(missing)}")

    if df[required_columns].isnull().any().any():
        raise ValueError(f"{name} dataset contains missing feature/label values.")

    if not set(df[TARGET].unique()).issubset({0, 1}):
        raise ValueError(f"{name} dataset has unexpected labels.")

X_train = train_df[FEATURES]
y_train = train_df[TARGET].astype(int)
X_test = test_df[FEATURES]
y_test = test_df[TARGET].astype(int)

print(f"[+] Training rows: {len(X_train)}")
print(f"[+] Testing rows: {len(X_test)}")
print("[+] Training Random Forest model...")

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)

model.fit(X_train, y_train)

print("[+] Evaluating model...")
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions, zero_division=0)
recall = recall_score(y_test, predictions, zero_division=0)
f1 = f1_score(y_test, predictions, zero_division=0)
matrix = confusion_matrix(y_test, predictions, labels=[0, 1])
report = classification_report(
    y_test,
    predictions,
    labels=[0, 1],
    target_names=["Normal", "Simulated Attack"],
    zero_division=0
)

model_path = MODEL_DIR / "random_forest.joblib"
joblib.dump(model, model_path)

importance = pd.DataFrame({
    "Feature": FEATURES,
    "Importance": model.feature_importances_
}).sort_values("Importance", ascending=False)

report_text = f"""VoltGuard ICS/SCADA Intrusion Detection
========================================
Dataset: Simulated ICS/SCADA data
Model: Random Forest
Training rows: {len(X_train)}
Testing rows: {len(X_test)}
Features: {", ".join(FEATURES)}

Accuracy:  {accuracy:.4f}
Precision: {precision:.4f}
Recall:    {recall:.4f}
F1-score:  {f1:.4f}

Confusion Matrix (actual rows, predicted columns):
                 Predicted Normal  Predicted Attack
Actual Normal    {matrix[0, 0]:16d} {matrix[0, 1]:17d}
Actual Attack    {matrix[1, 0]:16d} {matrix[1, 1]:17d}

Classification Report:
{report}

Feature Importance:
{importance.to_string(index=False)}

Limitation:
These results measure performance on the supplied simulated dataset.
They do not establish performance on real industrial systems.
"""

report_path = REPORT_DIR / "model_evaluation.txt"
report_path.write_text(report_text, encoding="utf-8")

print("\n========== MODEL RESULTS ==========")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-score : {f1:.4f}")
print("\nConfusion Matrix:")
print(matrix)
print("\nClassification Report:")
print(report)
print("\nFeature Importance:")
print(importance.to_string(index=False))
print(f"\n[+] Model saved: {model_path}")
print(f"[+] Evaluation report saved: {report_path}")
print("[+] Training and evaluation completed.")
