# VoltGuard – Module 3: Physics-Aware Analysis & Alert Dashboard
Academic prototype. Offline/simulated data only. Do not connect to live control equipment.

## Setup
    pip install -r requirements.txt
    pytest -q                       # 17 rule/fusion/incident tests (or: python test_rules.py)
    python run_pipeline.py          # simulated data -> alerts.csv, incidents.csv, review_queue.csv
    streamlit run app.py            # dashboard

## Integrating with teammates
Option A: one CSV with the columns below plus `ai_score` (0-1).
Option B: process CSV + Module 2 CSV (`timestamp, ai_score`), joined by nearest timestamp:
    python run_pipeline.py process.csv module2_preds.csv
Required process columns: timestamp, voltage_kv, current_a, frequency_hz, power_mw,
power_factor, transformer_temp_c, breaker_status (0=open, 1=closed).
Input is validated (missing columns, bad values, unsorted rows are handled or reported).

## Files
simulate_process.py (simulated process + MOCK AI) | physics_rules.py (rules R1-R9, validation)
alert_engine.py (fusion, join, incidents, review queue, evaluation) | app.py (dashboard)
test_rules.py | RULES.md (rule table: threshold + rationale)

## Method
Rules: see RULES.md. Fusion: physics severity + AI flag -> final severity (matrix in alert_engine.py
and the dashboard Rules tab). AI-only alerts are "Low", go to a review queue, and do not open incidents.
Incidents = Medium+ alerts with gaps <= 5 samples merged.

## Limitations
- Simulated data; the mock `ai_score` is hand-crafted, so the "combined beats either alone" table
  demonstrates the design, not real accuracy. Re-run with Module 2's real scores.
- Rules were written knowing the injected attacks, so physics-only recall is optimistic.
- R9 (frozen sensor) needs a 20-sample window; the run is flagged retrospectively (offline),
  which would be a detection delay in live use.
- Metrics are per-sample, not per-incident. Not a validated industrial safety product.
