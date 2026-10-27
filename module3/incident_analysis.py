"""Incident Analyst: attack-type inference, risk score, MITRE ATT&CK for ICS mapping,
recommended response and a downloadable report. Works on build_incidents() output."""
import pandas as pd

PLAYBOOK = {
    "Breaker manipulation": (
        "T0855 Unauthorized Command Message",
        "Verify the breaker position locally at the field device; review recent operator "
        "commands and engineering-station logins; isolate the control network segment if "
        "the command was not authorised."),
    "Replay / frozen sensor": (
        "T0856 Spoof Reporting Message",
        "Cross-check readings with an independent meter or historian; look for replayed or "
        "stale packets; use manual monitoring until the sensor feed is verified."),
    "False data injection": (
        "T0832 Manipulation of View",
        "Compare reported power with V x I x pf from an independent source; inspect the "
        "RTU/gateway for tampering; treat displayed values as untrusted until verified."),
    "Slow power drift": (
        "T0832 Manipulation of View",
        "Trend reported power against V x I x pf over a longer period; recalibrate or replace the "
        "power meter and check its firmware; treat the reading as untrusted until verified."),
    "Process parameter manipulation": (
        "T0836 Modify Parameter",
        "Check setpoint / tap-changer changes and configuration diffs; reduce load or trip "
        "per operating procedure if limits stay violated; escalate to the control engineer."),
    "Anomalous process behaviour": (
        "No direct mapping",
        "Review sensor health and recent operating changes; keep monitoring and compare "
        "with the AI detector output."),
    "Suspicious traffic (no physical impact)": (
        "No direct mapping",
        "Review the flagged network traffic; no physics rule was violated, so this may be a "
        "false positive. Tune the AI threshold if it repeats."),
}
BASE = {"Low": 20, "Medium": 40, "High": 65, "Critical": 85}


def classify(rules):
    """Rule signature -> attack type (checked in priority order)."""
    rules = set(rules)
    if "R7_BREAKER_OPEN_FLOW" in rules: return "Breaker manipulation"
    if "R9_FROZEN_SENSOR" in rules: return "Replay / frozen sensor"
    volt_or_heat = rules & {"R1_VOLTAGE_RANGE", "R4_OVERTEMP"}
    if "R6_POWER_MISMATCH" in rules and not volt_or_heat: return "False data injection"
    if "R10_POWER_DRIFT" in rules and not volt_or_heat: return "Slow power drift"
    if rules & {"R1_VOLTAGE_RANGE", "R2_FREQUENCY_RANGE", "R3_OVERCURRENT", "R4_OVERTEMP"}:
        return "Process parameter manipulation"
    if rules & {r for r in rules if r.startswith("R")}: return "Anomalous process behaviour"
    return "Suspicious traffic (no physical impact)"


def risk_score(severity, samples, avg_ai):
    """0-100: severity base + duration bonus (max 10) + AI confidence bonus (max 5)."""
    return int(min(100, BASE.get(severity, 0) + min(10, samples // 10) + round(5 * avg_ai)))


def enrich_incidents(inc, res):
    cols = ["attack_type", "risk_score", "mitre_ics", "duration_s", "evidence", "recommended_action"]
    if inc.empty:
        return inc.assign(**{c: [] for c in cols})
    out = []
    for _, r in inc.iterrows():
        seg = res[(res["timestamp"] >= r["start"]) & (res["timestamp"] <= r["end"])]
        rules = {x.strip() for x in str(r["rules"]).split(",") if x.strip().startswith("R")}
        atype = classify(rules)
        gap = (seg["power_mw"] - seg["power_calc_mw"]).abs().max()
        evidence = (f"I {seg.current_a.min():.0f}-{seg.current_a.max():.0f} A | "
                    f"V {seg.voltage_kv.min():.2f}-{seg.voltage_kv.max():.2f} kV | "
                    f"T max {seg.transformer_temp_c.max():.0f} C | power gap max {gap:.2f} MW | "
                    f"breaker open samples {int((seg.breaker_status == 0).sum())}")
        mitre, action = PLAYBOOK[atype]
        out.append({"attack_type": atype,
                    "risk_score": int(r["max_risk"]) if "max_risk" in inc.columns else
                    risk_score(r["max_severity"], int(r.get("alert_samples", r.get("samples", 1))), r["avg_ai_score"]),
                    "mitre_ics": mitre, "duration_s": int(r.get("duration_s", r.get("alert_samples", r.get("samples", 1)))),
                    "evidence": evidence, "recommended_action": action})
    base = inc.drop(columns=[c for c in cols if c in inc.columns]).reset_index(drop=True)
    return pd.concat([base, pd.DataFrame(out)], axis=1)


def generate_report(res, ei, source="", threshold=0.5):
    lines = ["# VoltGuard Incident Report", "",
             f"- Data source: {source or 'n/a'}",
             f"- Period: {res['timestamp'].min()} to {res['timestamp'].max()} ({len(res)} samples)",
             f"- AI threshold: {threshold}",
             f"- Incidents: {len(ei)} | Highest risk score: {int(ei['risk_score'].max()) if len(ei) else 0}",
             "", "## Severity counts", ""]
    for k, v in res["final_severity"].value_counts().items():
        lines.append(f"- {k}: {v}")
    lines += ["", "## Incidents", ""]
    for _, r in ei.sort_values("risk_score", ascending=False).iterrows():
        lines += [f"### Incident {r['incident']}: {r['attack_type']} (risk {r['risk_score']}/100)", "",
                  f"- Time: {r['start']} to {r['end']} ({r['duration_s']} s)",
                  f"- Severity: {r['max_severity']} | Average AI score: {r['avg_ai_score']}",
                  f"- Rules triggered: {r['rules']}",
                  f"- MITRE ATT&CK for ICS: {r['mitre_ics']}",
                  f"- Evidence: {r['evidence']}",
                  f"- Recommended response: {r['recommended_action']}", ""]
    lines += ["## Limitations", "",
              "Academic prototype on simulated/offline data; not a validated industrial safety "
              "product. Attack types are inferred from rule signatures and need analyst review."]
    return "\n".join(lines)