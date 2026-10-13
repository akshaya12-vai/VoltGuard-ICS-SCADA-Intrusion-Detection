from pathlib import Path
import joblib
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "random_forest.joblib"

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        "Model not found. Run: python scripts/train_model.py"
    )

model = joblib.load(MODEL_PATH)

FEATURES = [
    "Tank_Water_Level",
    "Pressure_Sensor",
    "Flow_Rate",
    "Valve_Status"
]

print("\n=== VoltGuard ICS/SCADA Prediction Demo ===")
print("Enter simulated sensor readings.")
print("Type Ctrl+C to exit.\n")

try:
    while True:
        level = float(input("Tank water level: "))
        pressure = float(input("Pressure sensor: "))
        flow = float(input("Flow rate: "))
        valve = int(input("Valve status (0 or 1): "))

        if valve not in (0, 1):
            print("Valve status must be 0 or 1.\n")
            continue

        readings = pd.DataFrame(
            [[level, pressure, flow, valve]],
            columns=FEATURES
        )

        prediction = int(model.predict(readings)[0])
        probabilities = model.predict_proba(readings)[0]
        attack_index = list(model.classes_).index(1)
        score = probabilities[attack_index] * 100

        if prediction == 1:
            print("\n[ALERT] Model predicts: SIMULATED ATTACK")
        else:
            print("\n[OK] Model predicts: NORMAL")

        print(f"Model attack score: {score:.1f}%")
        print("Demo only: validate against real ICS/SCADA data before operational use.\n")

except (ValueError, EOFError):
    print("\nInvalid or incomplete input. Please enter numeric values.")
except KeyboardInterrupt:
    print("\nVoltGuard prediction demo stopped.")
