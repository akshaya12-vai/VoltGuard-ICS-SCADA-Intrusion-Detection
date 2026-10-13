import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(r"D:\VoltGuard\VoltGuard-ICS-SCADA-Intrusion-Detection")
PROCESSED_DIR = BASE_DIR / "data" / "processed"
REPORTS_DIR = BASE_DIR / "reports"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

print("[-] Generating and Preprocessing VoltGuard ICS/SCADA Dataset...")

np.random.seed(42)
n_samples = 5000

timestamps = pd.date_range(start="2026-10-13 00:00:00", periods=n_samples, freq="S")
tank_water_level = np.random.normal(50.0, 5.0, n_samples)
pressure_sensor = np.random.normal(30.0, 2.1, n_samples)
flow_rate = np.random.normal(12.5, 1.2, n_samples)
valve_status = np.random.choice([0, 1], size=n_samples, p=[0.1, 0.9])

attack_indices = np.random.choice(n_samples, size=250, replace=False)
tank_water_level[attack_indices] = np.random.uniform(75.0, 95.0, size=250)
pressure_sensor[attack_indices] = np.random.uniform(45.0, 60.0, size=250)

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

df.loc[10:15, "Pressure_Sensor"] = np.nan
df = pd.concat([df, df.iloc[:5]], ignore_index=True)

missing_count_before = df.isna().sum().sum()
duplicate_count_before = df.duplicated().sum()

for col in ["Tank_Water_Level", "Pressure_Sensor", "Flow_Rate"]:
    df[col] = df[col].fillna(df[col].median())

df = df.drop_duplicates().reset_index(drop=True)

train_size = int(0.8 * len(df))
train_df = df.iloc[:train_size]
test_df = df.iloc[train_size:]

df.to_csv(PROCESSED_DIR / "cleaned_dataset.csv", index=False)
train_df.to_csv(PROCESSED_DIR / "train_data.csv", index=False)
test_df.to_csv(PROCESSED_DIR / "test_data.csv", index=False)

data_dict = pd.DataFrame({
    "Feature_Name": ["Timestamp", "Tank_Water_Level", "Pressure_Sensor", "Flow_Rate", "Valve_Status", "Attack_Label"],
    "Data_Type": ["datetime", "float", "float", "float", "int", "int"],
    "Description": [
        "Exact timestamp of the sensor reading",
        "Water level in meters (Normal: 40-60, Violation > 70)",
        "System pressure in PSI",
        "Liquid flow rate through pipes",
        "Valve state (0: Closed, 1: Open)",
        "Classification label (0: Normal, 1: Intrusion/Attack)"
    ]
})
data_dict.to_csv(REPORTS_DIR / "data_dictionary.csv", index=False)

report_content = f"""VoltGuard Module 1 - Data Preprocessing Report
==============================================
Dataset Name: Simulated ICS/SCADA Water Tank & Process Dataset
Total Records (Raw): {len(df) + duplicate_count_before}
Total Records (Cleaned): {len(df)}
Missing Values Cleaned: {missing_count_before}
Duplicate Rows Removed: {duplicate_count_before}

Dataset Statistics:
- Training Samples: {len(train_df)}
- Testing Samples: {len(test_df)}
- Normal Traffic Count: {int((df['Attack_Label'] == 0).sum())}
- Attack Traffic Count: {int((df['Attack_Label'] == 1).sum())}
"""

with open(REPORTS_DIR / "preprocessing_report.txt", "w") as f:
    f.write(report_content)

print("[+] Preprocessing completed successfully! All deliverables generated.")
