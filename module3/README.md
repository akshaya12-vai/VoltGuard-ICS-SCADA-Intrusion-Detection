# VoltGuard – Module 3: Physics-Aware Analysis & Alert Dashboard

## One-time setup (inside module3/)
    python3 -m venv .venv && source .venv/bin/activate
    python -m pip install -r requirements.txt

## Run order (everything connected)
    python generate_dataset.py       # 1. common 11 kV dataset  -> data/feeder/train.csv, test.csv
    python module2_train_predict.py  # 2. Module 2 model + ai_score -> data/feeder/module2_output.csv
    streamlit run web_app.py         # 3a. full site: Overview, Module 1, Module 2, Module 3 pages
    streamlit run app.py             # 3b. analyst dashboard (live replay, incident triage, ablation)
    python run_pipeline.py [process.csv [module2.csv]]   # CLI -> alerts/incidents/review_queue CSVs
    pytest -q                        # 20 tests

## How the files connect
    simulate_process.py  -> common dataset (process + traffic columns), hard attack, inject_attack
    generate_dataset.py  -> data/feeder/*.csv + data dictionary + dataset report (Module 1 view)
    module2_train_predict.py -> trains on traffic columns, writes module2_output.csv (Module 2)
    physics_rules.py     -> rules R1-R10, validate_input, data_quality
    alert_engine.py      -> fuse AI + physics, incidents, evaluation, review queue, join_module2
    incident_analysis.py -> attack type, MITRE ICS mapping, response steps, report
    integration.py       -> loads Module 1/2 files for the web_app pages
    web_app.py / app.py  -> the two dashboards

## Agreed data format (Module 2 output)
timestamp, voltage_kv, current_a, frequency_hz, power_mw, power_factor,
transformer_temp_c, breaker_status, ai_score   (extra columns such as true_attack are fine)

Offline/simulated data only. Do not connect to live control equipment.