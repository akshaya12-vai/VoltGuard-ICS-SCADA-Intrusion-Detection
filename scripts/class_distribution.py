import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "data" / "processed" / "cleaned_dataset.csv"
REPORT_DIR = BASE_DIR / "reports"

def generate_class_report():
    df = pd.read_csv(DATA_FILE)

    counts = df["Attack_Label"].value_counts().sort_index()
    total = len(df)

    report = pd.DataFrame({
        "Class": ["Normal", "Simulated Attack"],
        "Label": [0, 1],
        "Record_Count": [
            counts.get(0, 0),
            counts.get(1, 0)
        ]
    })

    report["Percentage"] = (
        report["Record_Count"] / total * 100
    ).round(2)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    output = REPORT_DIR / "class_distribution_report.csv"
    report.to_csv(output, index=False)

    print("\nVoltGuard - Class Distribution Report")
    print("=" * 40)
    print(report.to_string(index=False))
    print(f"\nTotal records: {total}")
    print(f"[+] Report saved: {output}")

if __name__ == "__main__":
    generate_class_report()