
import pandas as pd
import numpy as np
from pathlib import Path

# Automatically identify the project directory
BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
REPORTS_DIR = BASE_DIR / "reports"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

print("[-] Generating and preprocessing VoltGuard ICS/SCADA dataset...")

# Reproducible simulated dataset
np.random.seed(42)
n_samples = 5000

timestamps = pd.date_range(
    start="2026-10-13 00:00:00",
    periods=n_samples,
    freq="s"
)

tank_water_level = np.random.normal(50.0, 5.0, n_samples)
pressure_sensor = np.random.normal(30.0, 2.1, n_samples)
flow_rate = np.random.normal(12.5, 1.2, n_samples)

valve_status = np.random.choice(
    [0, 1], size=n_samples, p=[0.1, 0.9]
)

# Simulate 250 attack records
attack_indices = np.random.choice(
    n_samples, size=250, replace=False
)

tank_water_level[attack_indices] = np.random.uniform(
    75.0, 95.0, size=250
)
pressure_sensor[attack_indices] = np.random.uniform(
    45.0, 60.0, size=250
)

labels = np.zeros(n_samples, dtype=int)
labels[attack_indices] = 1

df = pd.DataFrame({
    "Timestamp": timestamps,
    "Tank_Water_Level": tank_water_level,
    "Pressure_Sensor": pressure_sensor,
    "Flow_Rate": flow_rate,
    "Valve_Status": valve_status,
    "Attack_Label": labels
})

# Introduce missing values and duplicate records for testing
df.loc[10:15, "Pressure_Sensor"] = np.nan
df = pd.concat([df, df.iloc[:5]], ignore_index=True)

raw_record_count = len(df)
missing_count_before = int(df.isna().sum().sum())
duplicate_count_before = int(df.duplicated().sum())

# Fill missing numeric values using their medians
numeric_columns = [
    "Tank_Water_Level",
    "Pressure_Sensor",
    "Flow_Rate"
]

for col in numeric_columns:
    df[col] = df[col].fillna(df[col].median())

# Remove duplicate records
df = df.drop_duplicates().reset_index(drop=True)

# Validate processed data
remaining_missing = int(df.isna().sum().sum())
remaining_duplicates = int(df.duplicated().sum())

if remaining_missing > 0:
    raise ValueError("Missing values remain after preprocessing.")

if remaining_duplicates > 0:
    raise ValueError("Duplicate records remain after preprocessing.")

if not set(df["Attack_Label"].unique()).issubset({0, 1}):
    raise ValueError("Unexpected attack label detected.")

# Split chronologically into training and testing datasets
train_size = int(0.8 * len(df))
train_df = df.iloc[:train_size].copy()
test_df = df.iloc[train_size:].copy()

# Save processed datasets
df.to_csv(PROCESSED_DIR / "cleaned_dataset.csv", index=False)
train_df.to_csv(PROCESSED_DIR / "train_data.csv", index=False)
test_df.to_csv(PROCESSED_DIR / "test_data.csv", index=False)

# Create feature dictionary
data_dict = pd.DataFrame({
    "Feature_Name": [
        "Timestamp",
        "Tank_Water_Level",
        "Pressure_Sensor",
        "Flow_Rate",
        "Valve_Status",
        "Attack_Label"
    ],
    "Data_Type": [
        "datetime",
        "float",
        "float",
        "float",
        "int",
        "int"
    ],
    "Description": [
        "Timestamp of the simulated sensor reading",
        "Simulated water level; attack records use elevated values",
        "Simulated process pressure in PSI",
        "Simulated liquid flow rate",
        "Valve state (0: Closed, 1: Open)",
        "Label (0: Normal, 1: Simulated attack)"
    ]
})

data_dict.to_csv(
    REPORTS_DIR / "data_dictionary.csv",
    index=False
)

# Generate preprocessing report
normal_count = int((df["Attack_Label"] == 0).sum())
attack_count = int((df["Attack_Label"] == 1).sum())

report_content = f"""VoltGuard Module 1 - Data Preprocessing Report
================================================
Dataset: Simulated ICS/SCADA Water Tank Process Dataset

Raw records: {raw_record_count}
Cleaned records: {len(df)}
Duplicate records removed: {duplicate_count_before}
Missing values filled: {missing_count_before}
Missing values remaining: {remaining_missing}
Duplicate records remaining: {remaining_duplicates}

Dataset split:
Training samples: {len(train_df)}
Testing samples: {len(test_df)}

Class distribution:
Normal records: {normal_count}
Simulated attack records: {attack_count}

Validation:
Missing-value check: PASS
Duplicate check: PASS
Label validation: PASS

Note: This is a synthetic dataset for educational testing.
Results from this dataset do not establish real-world
ICS/SCADA intrusion detection performance.
"""

with open(
    REPORTS_DIR / "preprocessing_report.txt",
    "w",
    encoding="utf-8"
) as report_file:
    report_file.write(report_content)

print("[+] Preprocessing completed successfully!")
print(f"[+] Raw records: {raw_record_count}")
print(f"[+] Cleaned records: {len(df)}")
print(f"[+] Training records: {len(train_df)}")
print(f"[+] Testing records: {len(test_df)}")
print(f"[+] Normal: {normal_count} | Simulated attacks: {attack_count}")
print(f"[+] Output directory: {PROCESSED_DIR}")
print(f"[+] Reports directory: {REPORTS_DIR}")
