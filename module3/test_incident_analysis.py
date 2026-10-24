import pandas as pd
from incident_analysis import classify, risk_score, enrich_incidents
from simulate_process import simulate
from alert_engine import run_pipeline, build_incidents


def test_classify_signatures():
    assert classify({"R7_BREAKER_OPEN_FLOW"}) == "Breaker manipulation"
    assert classify({"R9_FROZEN_SENSOR", "R6_POWER_MISMATCH"}) == "Replay / frozen sensor"
    assert classify({"R6_POWER_MISMATCH"}) == "False data injection"
    assert classify({"R1_VOLTAGE_RANGE", "R6_POWER_MISMATCH"}) == "Process parameter manipulation"
    assert classify(set()).startswith("Suspicious traffic")


def test_risk_score_bounds():
    assert 0 <= risk_score("Critical", 500, 1.0) <= 100
    assert risk_score("Low", 1, 0.0) < risk_score("High", 1, 0.0)


def test_all_simulated_attacks_found():
    res = run_pipeline(simulate(), 0.5)
    ei = enrich_incidents(build_incidents(res), res)
    types = set(ei["attack_type"])
    assert {"False data injection", "Breaker manipulation",
            "Replay / frozen sensor", "Process parameter manipulation"} <= types
