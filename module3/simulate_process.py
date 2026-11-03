"""Simulated 11 kV feeder + network-traffic features (offline, safe).
COMMON DATASET for Modules 1-3. Attack windows are placed randomly per seed, so train (seed A)
and test (seed B) differ in timing, length and noise.
Also provides hard=True (slow power drift) and inject_attack() used by the analyst dashboard."""
import numpy as np
import pandas as pd

PROCESS_COLS = ["voltage_kv", "current_a", "frequency_hz", "power_mw", "power_factor",
                "transformer_temp_c", "breaker_status"]
TRAFFIC_COLS = ["packets_per_s", "write_requests_per_s", "unique_src_ips",
                "avg_payload_bytes", "illegal_function_codes"]
ATTACKS = ["false_data_injection", "breaker_manipulation",
           "replay_frozen_sensor", "overvoltage_injection"]
MOCK_BANDS = {"false_data_injection": (.10, .40), "breaker_manipulation": (.70, .95),
              "replay_frozen_sensor": (.55, .85), "overvoltage_injection": (.40, .90),
              "slow_power_drift": (.05, .30)}
INTERNAL = {"False data injection": "false_data_injection", "Breaker manipulation": "breaker_manipulation",
            "Replay / frozen sensor": "replay_frozen_sensor", "Over-voltage injection": "overvoltage_injection",
            "Slow power drift": "slow_power_drift"}
ATTACK_KINDS = list(INTERNAL)
DRIFT_BIAS = 0.25   # MW, stays under the R6 limit (0.4) but R10 (CUSUM) catches it


def _schedule(rng, n, hard=False):
    """2-3 non-overlapping windows per attack type (40-120 s); hard adds one long drift window."""
    taken = np.zeros(n, bool)
    wins = []
    plan = [(k, int(rng.integers(2, 4)), (40, 121)) for k in ATTACKS]
    if hard:
        plan.append(("slow_power_drift", 1, (150, 251)))
    for atype, reps, (lo, hi) in plan:
        for _ in range(reps):
            for _try in range(300):
                d = int(rng.integers(lo, hi))
                a = int(rng.integers(120, n - d - 30))
                if not taken[max(0, a - 30):a + d + 30].any():
                    taken[a:a + d] = True
                    wins.append((a, a + d, atype))
                    break
    return sorted(wins)


def simulate(n=3000, seed=42, traffic=True, mock_ai=True, start="2026-10-13 08:00:00", hard=False):
    rng = np.random.default_rng(seed)
    x = np.arange(n)
    t = pd.date_range(start, periods=n, freq="s")
    phase = int(rng.integers(0, 1500))

    current = 200 + 60 * np.sin(2 * np.pi * (x + phase) / 1500) + rng.normal(0, 2, n)
    voltage = 11 + rng.normal(0, 0.03, n)
    freq = 50 + rng.normal(0, 0.02, n)
    pf = 0.95 + rng.normal(0, 0.005, n)
    power = np.sqrt(3) * voltage * current * pf / 1000 + rng.normal(0, 0.02, n)
    temp = pd.Series(55 + 0.08 * current + rng.normal(0, 0.3, n)).ewm(span=60).mean().to_numpy().copy()
    breaker = np.ones(n, dtype=int)

    pk = rng.normal(40, 8, n).clip(5)
    wr = rng.poisson(0.6, n).astype(float)
    src = 2 + rng.binomial(1, .08, n).astype(float)
    pay = rng.normal(64, 10, n).clip(16)
    ill = rng.poisson(0.05, n).astype(float)

    for _ in range(15):  # benign maintenance bursts: look suspicious on the network, NOT attacks
        d = int(rng.integers(5, 16)); a = int(rng.integers(0, n - d))
        wr[a:a + d] += rng.poisson(3.5, d); pk[a:a + d] += rng.normal(12, 5, d)

    attack = np.zeros(n, dtype=int)
    atype = np.array(["normal"] * n, dtype=object)
    wins = _schedule(rng, n, hard)
    for a, b, k in wins:
        d = b - a
        attack[a:b] = 1; atype[a:b] = k
        if k == "false_data_injection":
            power[a:b] += rng.uniform(2.0, 4.0)
            src[a:b] += rng.binomial(1, .12, d)
        elif k == "breaker_manipulation":
            breaker[a:b] = 0
            wr[a:b] += rng.poisson(3.0, d); ill[a:b] += rng.poisson(.4, d)
            src[a:b] += rng.binomial(1, .3, d)
        elif k == "replay_frozen_sensor":
            current[a:b] = current[a]; voltage[a:b] = voltage[a]
            pk[a:b] += rng.normal(6, 8, d); pay[a:b] += rng.normal(-4, 8, d)
        elif k == "overvoltage_injection":
            voltage[a:b] = rng.uniform(12.5, 14.0) + rng.normal(0, 0.03, d)
            temp[a:b] += np.linspace(0, rng.uniform(25, 40), d)
            wr[a:b] += rng.poisson(2.0, d); pay[a:b] += rng.normal(14, 12, d)
        elif k == "slow_power_drift":
            power[a:b] += np.minimum(1, np.arange(d) / 20) * DRIFT_BIAS

    out = pd.DataFrame({"timestamp": t, "voltage_kv": voltage, "current_a": current,
                        "frequency_hz": freq, "power_mw": power, "power_factor": pf,
                        "transformer_temp_c": temp, "breaker_status": breaker})
    if traffic:
        out["packets_per_s"] = pk; out["write_requests_per_s"] = wr
        out["unique_src_ips"] = src; out["avg_payload_bytes"] = pay
        out["illegal_function_codes"] = ill
    if mock_ai:  # placeholder until the real Module 2 model supplies ai_score
        ai = rng.beta(2, 12, n)
        for a, b, k in wins:
            ai[a:b] = rng.uniform(*MOCK_BANDS[k], b - a)
        fp = rng.choice(np.where(attack == 0)[0], 15, replace=False)
        ai[fp] = rng.uniform(.70, .90, 15)
        out["ai_score"] = ai
    out["true_attack"] = attack
    out["attack_type"] = atype
    return out


def inject_attack(df, kind, start, length, strength=0.5, ai=False):
    """Return a copy of df with a custom attack window added (used by the analyst dashboard)."""
    d = df.copy()
    n = len(d)
    a = int(max(0, min(start, n - 1))); b = int(min(n, a + max(1, length))); m = b - a
    rng = np.random.default_rng(a * 7 + m)
    key = INTERNAL[kind]

    def put(col, fn):
        v = d[col].to_numpy(dtype=float).copy()
        v[a:b] = fn(v[a:b])
        d[col] = v

    if key == "false_data_injection":
        put("power_mw", lambda v: v + 6.0 * strength)
    elif key == "breaker_manipulation":
        v = d["breaker_status"].to_numpy().copy(); v[a:b] = 0; d["breaker_status"] = v
    elif key == "replay_frozen_sensor":
        c0, v0 = d["current_a"].iloc[a], d["voltage_kv"].iloc[a]
        put("current_a", lambda v: np.full_like(v, c0)); put("voltage_kv", lambda v: np.full_like(v, v0))
    elif key == "overvoltage_injection":
        put("voltage_kv", lambda v: 11 + 4.0 * strength + rng.normal(0, 0.03, m))
        put("transformer_temp_c", lambda v: v + np.linspace(0, 40 * strength, m))
    elif key == "slow_power_drift":
        put("power_mw", lambda v: v + np.minimum(1, np.arange(m) / 20) * 0.3 * strength)

    if "true_attack" not in d: d["true_attack"] = 0
    if "attack_type" not in d: d["attack_type"] = "normal"
    ta = d["true_attack"].to_numpy().copy(); ta[a:b] = 1; d["true_attack"] = ta
    at = d["attack_type"].to_numpy(dtype=object).copy(); at[a:b] = key; d["attack_type"] = at
    if "ai_score" not in d: d["ai_score"] = 0.0
    sc = d["ai_score"].to_numpy(dtype=float).copy()
    sc[a:b] = rng.uniform(.6, .9, m) if ai else rng.uniform(.05, .3, m)
    d["ai_score"] = sc
    return d


if __name__ == "__main__":
    simulate().to_csv("process_data.csv", index=False)
    print("Saved process_data.csv")
