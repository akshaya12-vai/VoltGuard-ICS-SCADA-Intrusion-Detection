"""Reference Module 2: trains on the TRAFFIC columns, compares with a 2-line threshold baseline,
writes the model, the evaluation report and module2_output.csv (the file Module 3 reads)."""
import os
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)
from simulate_process import PROCESS_COLS, TRAFFIC_COLS, ATTACKS

ROOT = Path(os.environ.get("VOLTGUARD_ROOT", Path(__file__).resolve().parent.parent))
tr = pd.read_csv(ROOT / "data/feeder/train.csv", parse_dates=["timestamp"])
te = pd.read_csv(ROOT / "data/feeder/test.csv", parse_dates=["timestamp"])
Xtr, ytr, Xte, yte = tr[TRAFFIC_COLS], tr["true_attack"], te[TRAFFIC_COLS], te["true_attack"]

# baseline: best single threshold on one feature (tuned on TRAIN only)
best = max(((f1_score(ytr, tr["write_requests_per_s"] >= th), th)
            for th in np.unique(tr["write_requests_per_s"])))
base_pred = te["write_requests_per_s"] >= best[1]

rf = RandomForestClassifier(n_estimators=200, min_samples_leaf=5, class_weight="balanced",
                            random_state=42, n_jobs=-1).fit(Xtr, ytr)
score = rf.predict_proba(Xte)[:, 1]
pred = score >= 0.5

lines = ["MODULE 2 EVALUATION (test split, different seed from train)", "",
         f"Baseline (write_requests_per_s >= {best[1]:.0f}): F1 {f1_score(yte, base_pred):.3f}",
         f"Random Forest: accuracy {accuracy_score(yte, pred):.3f} precision {precision_score(yte, pred):.3f} "
         f"recall {recall_score(yte, pred):.3f} F1 {f1_score(yte, pred):.3f} AUC {roc_auc_score(yte, score):.3f}",
         f"Confusion matrix [[TN FP][FN TP]]: {confusion_matrix(yte, pred).tolist()}", "",
         "Recall per attack type (network-only detection):"]
for k in ATTACKS:
    m = te["attack_type"] == k
    lines.append(f"  {k}: {pred[m].mean():.2f}")
lines += ["", "Low recall on false_data_injection / replay is expected: they have little network signature.",
          "That gap is what Module 3's physics rules are designed to close."]
report = "\n".join(lines)
print(report)

(ROOT / "models").mkdir(exist_ok=True); (ROOT / "reports").mkdir(exist_ok=True)
joblib.dump(rf, ROOT / "models/random_forest_feeder.joblib")
(ROOT / "reports/feeder_model_evaluation.txt").write_text(report)
out = te[["timestamp"] + PROCESS_COLS + TRAFFIC_COLS].copy()
out["ai_score"] = score
out["true_attack"] = te["true_attack"]; out["attack_type"] = te["attack_type"]
out.to_csv(ROOT / "data/feeder/module2_output.csv", index=False)
print("\nWrote data/feeder/module2_output.csv")
