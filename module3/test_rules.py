"""Run: pytest -q   (or: python test_rules.py)"""
import numpy as np
import pandas as pd
from physics_rules import apply_rules, validate_input, data_quality, CONFIG
from alert_engine import (run_pipeline, build_incidents, _fuse, join_module2, likely_attack,
                          evaluate_incidents, threshold_sweep, playbook_for, SIGNATURES)
from simulate_process import simulate, inject_attack, ATTACK_KINDS


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

def test_custom_limits_change_result():
    d = fixed(); d.loc[10, "current_a"] = 300.0
    assert "R3_OVERCURRENT" not in ";".join(run_pipeline(d)["rules_triggered"])
    assert "R3_OVERCURRENT" in ";".join(run_pipeline(d, cfg=dict(CONFIG, i_max=250.0))["rules_triggered"])

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

def test_r10_catches_slow_drift_that_r6_misses():
    d = fixed(120); d.loc[40:99, "power_mw"] += 0.25      # under the 0.4 MW R6 limit
    r = apply_rules(d)
    assert r.loc[40:99, "rules_triggered"].str.contains("R10").any()
    assert not r["rules_triggered"].str.contains("R6").any()
    assert not r.loc[0:39, "rules_triggered"].str.contains("R10").any()

def test_r10_quiet_on_normal_noise():
    d = fixed(500); d["power_mw"] += np.random.default_rng(3).normal(0, 0.02, 500)
    assert "R10_POWER_DRIFT" not in rules_hit(d)

def test_likely_attack_names():
    assert likely_attack({"R7_BREAKER_OPEN_FLOW"}) == "breaker_manipulation"
    assert likely_attack({"R1_VOLTAGE_RANGE", "R4_OVERTEMP", "R6_POWER_MISMATCH"}) == "overvoltage_injection"
    assert likely_attack({"R9_FROZEN_SENSOR", "R6_POWER_MISMATCH"}) == "replay_frozen_sensor"
    assert likely_attack({"R6_POWER_MISMATCH", "R10_POWER_DRIFT"}) == "false_data_injection"
    assert likely_attack({"R10_POWER_DRIFT"}) == "stealthy_power_drift"
    assert likely_attack(set()) == "AI-only anomaly"

def test_risk_score_orders_by_severity():
    d = fixed(); d.loc[10, "voltage_kv"] = 12.5; d.loc[20, "breaker_status"] = 0
    d.loc[30, "ai_score"] = 0.9
    r = run_pipeline(d)
    assert r.loc[20, "risk_score"] > r.loc[10, "risk_score"] > r.loc[30, "risk_score"] > r.loc[5, "risk_score"]
    assert r["risk_score"].between(0, 100).all()

def test_incident_level_evaluation_default_and_hard():
    for hard, n_attacks in ((False, 4), (True, 5)):
        res = run_pipeline(simulate(hard=hard)); inc = build_incidents(res)
        seg, s = evaluate_incidents(res, inc)
        assert s["attacks"] == n_attacks and s["detected"] == n_attacks
        assert s["false_alarm_incidents"] == 0 and s["correctly_named"] == n_attacks

def test_disabled_rules_are_switched_off():
    d = fixed(); d.loc[10, "current_a"] = 450.0
    assert "R3_OVERCURRENT" in rules_hit(d)
    assert "R3_OVERCURRENT" not in ";".join(run_pipeline(d, disabled=["R3_OVERCURRENT"])["rules_triggered"])

def test_every_injected_attack_is_detected_and_named():
    for kind in ATTACK_KINDS:
        df = inject_attack(simulate(), kind, 300, 80, 0.5, ai_sees=False)
        res = run_pipeline(df); seg, s = evaluate_incidents(res, build_incidents(res))
        mine = seg[seg["start"] == df["timestamp"].iloc[300]].iloc[0]
        assert mine["detected"] == "Yes" and mine["named_correctly"] == "Yes", (kind, dict(mine))
        assert s["false_alarm_incidents"] == 0, kind

def test_inject_attack_labels_and_bounds():
    df = inject_attack(simulate(), "breaker_manipulation", 2990, 500)   # runs past the end
    assert df["true_attack"].iloc[2990:].eq(1).all() and len(df) == 3000

def test_threshold_sweep_shape_and_monotonic_ai_recall():
    sw = threshold_sweep(run_pipeline(simulate()))
    assert set(sw["method"]) == {"AI only", "Combined: any alert", "Combined: High and above"}
    r = sw[sw["method"] == "AI only"]["recall"].tolist()
    assert all(a >= b for a, b in zip(r, r[1:]))

def test_data_quality_counts():
    d = fixed(10); d.loc[3, "timestamp"] = d.loc[2, "timestamp"]; d = d.drop(index=[6, 7, 8])
    d.loc[0, "voltage_kv"] = np.nan
    q = data_quality(d)
    assert q["duplicate_timestamps"] == 1 and q["gaps"] == 1 and q["missing_values"] == 1

def test_playbook_covers_all_attack_names():
    names = {n for n, _ in SIGNATURES} | {"AI-only anomaly", "unclassified anomaly"}
    assert all(len(playbook_for(n)) >= 3 for n in names)

if __name__ == "__main__":
    fns = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for f in fns: f(); print("ok", f.__name__)
    print(f"{len(fns)} passed")