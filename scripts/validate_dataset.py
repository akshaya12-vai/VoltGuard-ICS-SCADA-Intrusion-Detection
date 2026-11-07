import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"
REPORT_DIR = BASE_DIR / "reports"

CLEANED_FILE = DATA_DIR / "cleaned_dataset.csv"
TRAIN_FILE = DATA_DIR / "train_data.csv"
TEST_FILE = DATA_DIR / "test_data.csv"

REPORT_FILE = REPORT_DIR / "dataset_validation_report.txt"

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

    results = []
    summary = []

    def run_check(name, function):
        try:
            function()
            results.append(f"[PASS] {name}")
            print(f"[PASS] {name}")
        except Exception as error:
            results.append(f"[FAIL] {name}: {error}")
            print(f"[FAIL] {name}: {error}")

    print("\nVoltGuard - Dataset Validation")
    print("=" * 45)

    # Check required files
    files = [CLEANED_FILE, TRAIN_FILE, TEST_FILE]

    for file in files:
        if not file.exists():
            raise FileNotFoundError(f"Missing file: {file.name}")

    cleaned = pd.read_csv(CLEANED_FILE)
    train = pd.read_csv(TRAIN_FILE)
    test = pd.read_csv(TEST_FILE)

    datasets = [
        ("Cleaned", cleaned),
        ("Training", train),
        ("Testing", test)
    ]

    # 1. Column validation
    def check_columns():
        for name, df in datasets:
            if list(df.columns) != REQUIRED_COLUMNS:
                raise ValueError(f"Incorrect columns in {name} dataset")

    run_check("Required column validation", check_columns)

    # 2. Missing values
    def check_missing():
        for name, df in datasets:
            if df.isnull().values.any():
                raise ValueError(f"Missing values in {name} dataset")

    run_check("Missing value validation", check_missing)

    # 3. Duplicate records
    def check_duplicates():
        if cleaned.duplicated().any():
            raise ValueError("Duplicate records found")

    run_check("Duplicate record validation", check_duplicates)

    # 4. Numeric values
    def check_numeric():
        for name, df in datasets:
            for col in NUMERIC_COLUMNS:
                values = pd.to_numeric(df[col], errors="coerce")

                if values.isnull().any() or not np.isfinite(values).all():
                    raise ValueError(f"Invalid numeric values: {name}/{col}")

    run_check("Numeric value validation", check_numeric)

    # 5. Attack labels
    def check_labels():
        if not set(cleaned["Attack_Label"].unique()).issubset({0, 1}):
            raise ValueError("Invalid attack labels")

    run_check("Attack label validation", check_labels)

    # 6. Valve status
    def check_valve():
        if not set(cleaned["Valve_Status"].unique()).issubset({0, 1}):
            raise ValueError("Invalid valve status")

    run_check("Valve status validation", check_valve)

    # 7. Timestamp validation
    def check_timestamps():
        timestamps = pd.to_datetime(cleaned["Timestamp"], errors="coerce")

        if timestamps.isnull().any():
            raise ValueError("Invalid timestamps")

        if not timestamps.is_monotonic_increasing:
            raise ValueError("Timestamps not chronological")

    run_check("Timestamp validation", check_timestamps)

    # 8. Train/test split
    def check_split():
        if len(cleaned) != len(train) + len(test):
            raise ValueError("Train/test record count mismatch")

        train_times = pd.to_datetime(train["Timestamp"], errors="coerce")
        test_times = pd.to_datetime(test["Timestamp"], errors="coerce")

        if train_times.isnull().any() or test_times.isnull().any():
            raise ValueError("Invalid train/test timestamps")

        if set(train_times).intersection(set(test_times)):
            raise ValueError("Train/test timestamp overlap")

        if not train_times.max() < test_times.min():
            raise ValueError("Incorrect chronological split")

    run_check("Train/test split validation", check_split)

    # Dataset summary
    normal = int((cleaned["Attack_Label"] == 0).sum())
    attacks = int((cleaned["Attack_Label"] == 1).sum())

    summary = [
        "",
        "VOLTGUARD DATASET VALIDATION REPORT",
        "=" * 45,
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        f"Total records : {len(cleaned)}",
        f"Training rows : {len(train)}",
        f"Testing rows  : {len(test)}",
        f"Normal records: {normal}",
        f"Attack records: {attacks}",
        "",
        "VALIDATION RESULTS",
        "-" * 45
    ]

    summary.extend(results)

    passed = sum("[PASS]" in result for result in results)
    failed = sum("[FAIL]" in result for result in results)

    summary.extend([
        "",
        f"Total checks: {len(results)}",
        f"Passed      : {passed}",
        f"Failed      : {failed}",
        "",
        "FINAL STATUS: " + ("PASSED" if failed == 0 else "FAILED")
    ])

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.write_text("\n".join(summary), encoding="utf-8")

    print("\n" + "=" * 45)
    print(f"Passed: {passed} | Failed: {failed}")
    print(f"Report saved: {REPORT_FILE}")

    if failed == 0:
        print("[+] All validation checks passed!")
    else:
        raise ValueError("Dataset validation failed. Check the report.")


if __name__ == "__main__":
    validate_dataset()