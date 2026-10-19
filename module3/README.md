# VoltGuard – Module 3: Physics-Aware Analysis & Alert Dashboard
## Setup
    pip install -r requirements.txt
    python run_pipeline.py        # CLI test -> alerts.csv, incidents.csv
    streamlit run app.py          # dashboard
## Integrating with teammates
Module 2 must output a CSV with timestamp + ai_score (0-1) joined to process variables:
timestamp, voltage_kv, current_a, frequency_hz, power_mw, power_factor,
transformer_temp_c, breaker_status, ai_score.  Upload it in the dashboard sidebar
(or pass it: python run_pipeline.py module2_output.csv).
## Files
simulate_process.py (simulated process + mock AI) | physics_rules.py (rules R1-R9)
alert_engine.py (fusion, incidents, evaluation) | app.py (dashboard)
Offline/simulated data only. Do not connect to live control equipment.

# run command
cd module3
pip install -r requirements.txt
streamlit run app.py