
import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"

CLEANED_FILE = DATA_DIR / "cleaned_dataset.csv"
TRAIN_FILE = DATA_DIR / "train_data.csv"
TEST_FILE = DATA_DIR / "test_data.csv"

REQUIRED_COLUMNS = [
    "Timestamp",
    "Tank_Water_Level",
    "Pressure_Sensor",
    "Flow_Rate",
    "Valve_Status",
    "Attack_Label"
]

NUMERIC_COLUMNS = [
    "Tank_Water_Level",
    "Pressure_Sensor",
    "Flow_Rate",
    "Valve_Status",
    "Attack_Label"
]

def validate_dataset():

    print("\nVoltGuard - Dataset Validation")
    print("=" * 40)

    files = [CLEANED_FILE, TRAIN_FILE, TEST_FILE]

    for file in files:
        if not file.exists():
            raise FileNotFoundError(f"Missing file: {file.name}")

    cleaned = pd.read_csv(CLEANED_FILE)
    train = pd.read_csv(TRAIN_FILE)
    test = pd.read_csv(TEST_FILE)

    # 1. Column validation
    for name, df in [
        ("Cleaned", cleaned),
        ("Training", train),
        ("Testing", test)
    ]:
        if list(df.columns) != REQUIRED_COLUMNS:
            raise ValueError(f"{name} dataset columns are incorrect.")

    print("[PASS] Required columns validation")

    # 2. Missing value validation
    for name, df in [
        ("Cleaned", cleaned),
        ("Training", train),
        ("Testing", test)
    ]:
        if df.isnull().values.any():
            raise ValueError(f"Missing values found in {name} dataset.")

    print("[PASS] Missing value validation")

    # 3. Duplicate validation
    if cleaned.duplicated().any():
        raise ValueError("Duplicate rows found in cleaned dataset.")

    print("[PASS] Duplicate validation")

    # 4. Numeric and infinite value validation
    for name, df in [
        ("Cleaned", cleaned),
        ("Training", train),
        ("Testing", test)
    ]:
        for col in NUMERIC_COLUMNS:
            values = pd.to_numeric(df[col], errors="coerce")

            if values.isnull().any():
                raise ValueError(f"Invalid numeric values in {name}: {col}")

            if not np.isfinite(values).all():
                raise ValueError(f"Infinite values in {name}: {col}")

    print("[PASS] Numeric value validation")

    # 5. Attack label validation
    if not set(cleaned["Attack_Label"].unique()).issubset({0, 1}):
        raise ValueError("Invalid attack labels found.")

    print("[PASS] Attack label validation")

    # 6. Valve status validation
    if not set(cleaned["Valve_Status"].unique()).issubset({0, 1}):
        raise ValueError("Invalid valve status values found.")

    print("[PASS] Valve status validation")

    # 7. Timestamp validation
    cleaned["Timestamp"] = pd.to_datetime(
        cleaned["Timestamp"], errors="coerce"
    )

    if cleaned["Timestamp"].isnull().any():
        raise ValueError("Invalid timestamps found.")

    if not cleaned["Timestamp"].is_monotonic_increasing:
        raise ValueError("Timestamps are not in chronological order.")

    print("[PASS] Timestamp validation")

    # 8. Train/test split validation
    if len(cleaned) != len(train) + len(test):
        raise ValueError("Train/test record count mismatch.")

    train_times = set(train["Timestamp"])
    test_times = set(test["Timestamp"])

    if train_times.intersection(test_times):
        raise ValueError("Train/test timestamp overlap detected.")

    if not train["Timestamp"].max() < test["Timestamp"].min():
        raise ValueError("Train/test chronological order is incorrect.")

    print("[PASS] Train/test split validation")

    # 9. Class distribution
    normal = int((cleaned["Attack_Label"] == 0).sum())
    attacks = int((cleaned["Attack_Label"] == 1).sum())

    print("\nDataset Summary")
    print("-" * 40)
    print(f"Total records : {len(cleaned)}")
    print(f"Training rows : {len(train)}")
    print(f"Testing rows  : {len(test)}")
    print(f"Normal records: {normal}")
    print(f"Attack records: {attacks}")

    print("\n[+] All dataset validation checks passed!")


if __name__ == "__main__":
    validate_dataset()
