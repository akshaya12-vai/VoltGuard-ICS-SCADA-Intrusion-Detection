"""CLI: python run_pipeline.py [input.csv]  -> alerts.csv, incidents.csv"""
import sys
import pandas as pd
from simulate_process import simulate
from alert_engine import run_pipeline, build_incidents, evaluate

df = pd.read_csv(sys.argv[1], parse_dates=["timestamp"]) if len(sys.argv) > 1 else simulate()
res = run_pipeline(df)
res.to_csv("alerts.csv", index=False)
inc = build_incidents(res)
inc.to_csv("incidents.csv", index=False)
print(res["final_severity"].value_counts(), "\n")
print(inc.to_string(index=False), "\n")
ev = evaluate(res)
if ev is not None:
    print(ev.to_string(index=False))
