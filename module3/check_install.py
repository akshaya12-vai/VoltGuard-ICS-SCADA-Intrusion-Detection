"""Checks that every Module 3 file has the right content.   Run inside module3/:  python check_install.py"""
import importlib
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
# file: (text it must contain, text it must NOT contain)
EXPECT = {
    "simulate_process.py": (["def simulate(", "def inject_attack("], ["import streamlit"]),
    "physics_rules.py": (["R10_POWER_DRIFT", "def apply_rules("], ["import streamlit"]),
    "tank_rules.py": (["T8_VALVE_CLOSED_FLOW", "def apply_rules("], ["import streamlit"]),
    "tank_data.py": (["def inject_stress(", "def load("], ["import streamlit"]),
    "integration.py": (["def compatibility_report(tank", "PATHS = {"], ["import streamlit"]),
    "alert_engine.py": (["def evaluate_incidents", "profile=None"], ["import streamlit"]),
    "incident_analysis.py": (["_classify_tank", "def generate_report("], ["import streamlit"]),
    "web_app.py": (["st.navigation(", "def page_about"], []),
}
bad = 0
print("File check")
for name, (need, forbid) in EXPECT.items():
    p = HERE / name
    if not p.exists():
        print(f"  ❌ {name}: missing"); bad += 1; continue
    t = p.read_text(errors="replace")
    miss, forb = [s for s in need if s not in t], [s for s in forbid if s in t]
    if miss or forb:
        first = t.strip().splitlines()[0][:70] if t.strip() else "(empty)"
        print(f"  ❌ {name}: wrong content (starts with: {first})"); bad += 1
    else:
        print(f"  ✅ {name}")
if bad:
    print(f"\n{bad} file(s) wrong. Re-extract module3.zip over this folder:  unzip -o ~/Downloads/module3.zip -d ..")
    sys.exit(1)
print("\nImport check")
for m in ["simulate_process", "physics_rules", "tank_rules", "integration", "alert_engine", "incident_analysis", "tank_data"]:
    try:
        importlib.import_module(m); print(f"  ✅ {m}")
    except Exception as e:
        print(f"  ❌ {m}: {type(e).__name__}: {e}"); bad += 1
print("\nAll good. Start the dashboard:  streamlit run web_app.py" if not bad else "\nFix the items marked ❌ above.")
sys.exit(1 if bad else 0)
