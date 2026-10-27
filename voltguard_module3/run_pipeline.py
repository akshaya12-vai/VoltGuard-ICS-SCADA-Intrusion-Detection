"""CLI: python run_pipeline.py [process.csv [module2_preds.csv]] -> alerts.csv, incidents.csv, review_queue.csv"""
import sys
import pandas as pd
from simulate_process import simulate
from alert_engine import (run_pipeline, build_incidents, evaluate, evaluate_by_attack,
                          join_module2, review_queue)

if len(sys.argv) > 1:
    df = pd.read_csv(sys.argv[1], parse_dates=["timestamp"])
    if len(sys.argv) > 2:
        df = join_module2(df, pd.read_csv(sys.argv[2], parse_dates=["timestamp"]))
else:
    df = simulate()
res = run_pipeline(df)
res.to_csv("alerts.csv", index=False)
inc = build_incidents(res)
inc.to_csv("incidents.csv", index=False)
review_queue(res).to_csv("review_queue.csv", index=False)
print(res["final_severity"].value_counts(), "\n")
print(inc.to_string(index=False), "\n")
for t in (evaluate(res), evaluate_by_attack(res)):
    if t is not None:
        print(t.to_string(index=False), "\n")
