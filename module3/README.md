# VoltGuard – Module 3: Physics-Aware Analysis & Alert Dashboard

Module 3 adapts to Modules 1 and 2 (they are NOT changed):
- Module 1: `data/processed/*.csv` (water tank: Tank_Water_Level, Pressure_Sensor, Flow_Rate, Valve_Status, Attack_Label)
- Module 2: `models/random_forest.joblib` (scores those 4 columns)
- Module 3: water-tank physics rules (T1-T9) + the real Module 2 `ai_score` -> alerts, incidents, dashboard

## Setup (inside module3/)
    python3 -m venv .venv && source .venv/bin/activate
    python -m pip install -r requirements.txt

## Run (teammates' scripts first, from the repo root)
    python scripts/preprocessing.py   # Module 1 (unchanged)
    python scripts/train_model.py     # Module 2 (unchanged)
    cd module3
    streamlit run web_app.py          # Overview shows a connection check; default process = Water tank
    pytest -q                         # 28 tests

In the sidebar: "Water tank (Module 1/2 data)" uses their files; tick "Add Module 3 stress-test attacks"
to add 4 labelled, sustained attacks that a threshold model cannot see (physics rules catch them).
"11 kV feeder (simulation)" is the earlier simulated demo (also: `streamlit run app.py`).

## Files
    tank_rules.py   water-tank physics rules T1-T9 (RULES_TANK.md lists thresholds and rationale)
    tank_data.py    reads Module 1 files + Module 2 model (read-only), adds ai_score, optional stress tests
    alert_engine.py fusion AI + physics, incidents, evaluation   | incident_analysis.py attack type, MITRE ICS, response
    integration.py  paths + connection check                     | web_app.py multi-page dashboard
    physics_rules.py / simulate_process.py / app.py   11 kV feeder simulation profile

Offline/simulated data only. Do not connect to live control equipment.
