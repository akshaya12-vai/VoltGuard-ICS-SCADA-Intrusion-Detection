"""Run: pytest -q   (or: python test_rules.py)"""
import numpy as np
import pandas as pd
from physics_rules import apply_rules, validate_input, CONFIG
from alert_engine import run_pipeline, build_incidents, _fuse, join_module2


def base(n=60):
    return pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=n, freq="s"),
        "voltage_kv": 11.0 + np.random.default_rng(0).normal(0, 0.01, n),
        "current_a": 200.0 + np.random.default_rng(1).normal(0, 1, n),
        "frequency_hz": 50.0, "power_factor": 0.95, "transformer_temp_c": 70.0,
        "breaker_status": 1, "power_mw": 0.0, "ai_score": 0.0})


def fixed(n=60):
    d = base(n)
    d["power_mw"] = np.sqrt(3) * d.voltage_kv * d.current_a * d.power_factor / 1000
    return d


def rules_hit(d):
    return set(";".join(apply_rules(d)["rules_triggered"]).split(";")) - {""}


def test_normal_has_no_violations():
    assert rules_hit(fixed()) == set()

def test_r1_voltage():
    d = fixed(); d.loc[10, "voltage_kv"] = 12.5
    assert "R1_VOLTAGE_RANGE" in rules_hit(d)

def test_r2_frequency():
    d = fixed(); d.loc[10, "frequency_hz"] = 48.0
    assert "R2_FREQUENCY_RANGE" in rules_hit(d)

def test_r3_overcurrent():
    d = fixed(); d.loc[10, "current_a"] = 450.0
    assert "R3_OVERCURRENT" in rules_hit(d)

def test_r4_overtemp():
    d = fixed(); d.loc[10, "transformer_temp_c"] = 95.0
    assert "R4_OVERTEMP" in rules_hit(d)

def test_r5_pf():
    d = fixed(); d.loc[10, "power_factor"] = 0.5
    assert "R5_PF_RANGE" in rules_hit(d)

def test_r6_power_mismatch():
    d = fixed(); d.loc[10, "power_mw"] += 3.0
    assert "R6_POWER_MISMATCH" in rules_hit(d)

def test_r7_breaker_open_flow():
    d = fixed(); d.loc[10:12, "breaker_status"] = 0
    assert "R7_BREAKER_OPEN_FLOW" in rules_hit(d)

def test_r7_not_triggered_when_no_current():
    d = fixed(); d.loc[10, "breaker_status"] = 0; d.loc[10, "current_a"] = 0.0
    d["power_mw"] = np.sqrt(3) * d.voltage_kv * d.current_a * d.power_factor / 1000
    assert "R7_BREAKER_OPEN_FLOW" not in rules_hit(d)

def test_r8_rate_of_change():
    d = fixed(); d.loc[10, "current_a"] = 330.0
    assert "R8_RATE_OF_CHANGE" in rules_hit(d)

def test_r9_frozen_whole_run_flagged():
    d = fixed(); d.loc[20:44, ["current_a", "voltage_kv"]] = d.loc[20, ["current_a", "voltage_kv"]].values
    d["power_mw"] = np.sqrt(3) * d.voltage_kv * d.current_a * d.power_factor / 1000
    r = apply_rules(d)
    assert r.loc[20:44, "rules_triggered"].str.contains("R9").all()   # incl. first 19 samples
    assert not r.loc[0:15, "rules_triggered"].str.contains("R9").any()

def test_fusion_matrix():
    assert _fuse(4, False) == "Critical" and _fuse(3, True) == "Critical"
    assert _fuse(3, False) == "High" and _fuse(2, True) == "High"
    assert _fuse(2, False) == "Medium" and _fuse(0, True) == "Low" and _fuse(0, False) == "Normal"

def test_ai_only_is_low_and_not_an_incident():
    d = fixed(); d.loc[5, "ai_score"] = 0.9
    res = run_pipeline(d)
    assert res.loc[5, "final_severity"] == "Low" and build_incidents(res).empty

def test_incident_gap_merging():
    d = fixed(); d.loc[[10, 11, 14], "voltage_kv"] = 12.5   # gap of 2 samples
    d["power_mw"] = np.sqrt(3) * d.voltage_kv * d.current_a * d.power_factor / 1000
    assert len(build_incidents(run_pipeline(d), max_gap=5)) == 1
    assert len(build_incidents(run_pipeline(d), max_gap=0)) == 2

def test_validation_errors():
    for bad in (fixed().drop(columns=["current_a"]), fixed().assign(breaker_status=2)):
        try: validate_input(bad); assert False
        except ValueError: pass

def test_unsorted_input_is_sorted():
    d = fixed().sample(frac=1, random_state=1)
    assert run_pipeline(d)["timestamp"].is_monotonic_increasing

def test_join_module2():
    d = fixed(10).drop(columns="ai_score")
    m = pd.DataFrame({"timestamp": d.timestamp[:5], "ai_score": 0.9})
    j = join_module2(d, m, tolerance="500ms")
    assert (j.ai_score[:5] == 0.9).all() and (j.ai_score[5:] == 0).all()

if __name__ == "__main__":
    fns = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for f in fns: f(); print("ok", f.__name__)
    print(f"{len(fns)} passed")
