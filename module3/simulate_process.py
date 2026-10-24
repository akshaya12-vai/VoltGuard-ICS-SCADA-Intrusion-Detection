"""Simulates an 11 kV substation feeder (offline, safe) with injected attacks.
Also produces a MOCK Module 2 output (ai_score) so Module 3 runs standalone.
WARNING: ai_score is hand-crafted, so evaluation on it shows the design, not real accuracy."""
import numpy as np
import pandas as pd


def simulate(n=3000, seed=42, hard=False):
    rng = np.random.default_rng(seed)
    x = np.arange(n)
    t = pd.date_range("2026-10-13 08:00:00", periods=n, freq="s")

    current = 200 + 60 * np.sin(2 * np.pi * x / 1500) + rng.normal(0, 2, n)
    voltage = 11 + rng.normal(0, 0.03, n)
    freq = 50 + rng.normal(0, 0.02, n)
    pf = 0.95 + rng.normal(0, 0.005, n)
    power = np.sqrt(3) * voltage * current * pf / 1000 + rng.normal(0, 0.02, n)
    temp = pd.Series(55 + 0.08 * current + rng.normal(0, 0.3, n)).ewm(span=60).mean().to_numpy().copy()
    breaker = np.ones(n, dtype=int)
    ai = rng.beta(2, 12, n)  # normal traffic -> low AI score
    attack = np.zeros(n, dtype=int)
    atype = np.array(["normal"] * n, dtype=object)

    def mark(a, b, name):
        attack[a:b] = 1
        atype[a:b] = name

    # 1) False data injection on power reading (AI misses it: looks like normal traffic)
    power[800:900] += 3.0
    ai[800:900] = rng.uniform(0.10, 0.40, 100)
    mark(800, 900, "false_data_injection")

    # 2) Spoofed/unauthorised breaker-open command while load current still reported
    breaker[1500:1560] = 0
    ai[1500:1560] = rng.uniform(0.70, 0.95, 60)
    mark(1500, 1560, "breaker_manipulation")

    # 3) Replay attack: sensor values frozen
    current[2100:2200] = current[2100]
    voltage[2100:2200] = voltage[2100]
    ai[2100:2200] = rng.uniform(0.55, 0.85, 100)
    mark(2100, 2200, "replay_frozen_sensor")

    # 4) Over-voltage injection + heating
    voltage[2600:2650] = 13.5 + rng.normal(0, 0.03, 50)
    temp[2600:2650] += np.linspace(0, 35, 50)
    ai[2600:2650] = rng.uniform(0.40, 0.90, 50)
    mark(2600, 2650, "overvoltage_injection")

    if hard:  # 5) Stealthy slow drift on power reading: stays under the 0.4 MW R6 limit, AI misses it
        power[1100:1250] += np.linspace(0.15, 0.35, 150)
        ai[1100:1250] = rng.uniform(0.05, 0.30, 150)
        mark(1100, 1250, "stealthy_power_drift")

    # Benign AI false positives (physics is normal here)
    fp = rng.choice(np.where(attack == 0)[0], 15, replace=False)
    ai[fp] = rng.uniform(0.70, 0.90, 15)

    return pd.DataFrame({
        "timestamp": t, "voltage_kv": voltage, "current_a": current,
        "frequency_hz": freq, "power_mw": power, "power_factor": pf,
        "transformer_temp_c": temp, "breaker_status": breaker,
        "ai_score": ai,                       # <- replace with Module 2 output
        "true_attack": attack, "attack_type": atype,
    })


ATTACK_KINDS = ["false_data_injection", "stealthy_power_drift", "breaker_manipulation",
                "replay_frozen_sensor", "overvoltage_injection"]


def inject_attack(df, kind, start, length, strength=0.5, ai_sees=False, seed=0):
    """Return a copy of df with a custom attack injected (labels and a matching mock ai_score).
    strength 0.1-1.0 scales false_data_injection (6 MW x s), stealthy_power_drift (0.7 MW x s)
    and overvoltage_injection (11 + 5 x s kV); breaker and replay ignore it."""
    if kind not in ATTACK_KINDS:
        raise ValueError(f"Unknown attack kind: {kind}")
    d, rng = df.copy(), np.random.default_rng(seed)
    s = int(max(0, min(start, len(d) - 1))); e = int(min(len(d), s + max(1, int(length)))); L = e - s

    def add(col, vals):
        arr = d[col].to_numpy(dtype=float).copy(); arr[s:e] += vals; d[col] = arr

    def put(col, vals):
        arr = d[col].to_numpy().copy(); arr[s:e] = vals; d[col] = arr

    if kind == "false_data_injection":
        add("power_mw", 6.0 * strength)
    elif kind == "stealthy_power_drift":
        add("power_mw", np.linspace(0.5, 1.0, L) * 0.7 * strength)
    elif kind == "breaker_manipulation":
        put("breaker_status", 0)
    elif kind == "replay_frozen_sensor":
        for c in ("current_a", "voltage_kv"):
            put(c, d[c].iloc[s])
    else:  # overvoltage_injection
        put("voltage_kv", 11 + 5 * strength + rng.normal(0, 0.03, L))
        add("transformer_temp_c", np.linspace(0, 35, L))
    if "ai_score" not in d:
        d["ai_score"] = 0.0
    put("ai_score", rng.uniform(0.6, 0.95, L) if ai_sees else rng.uniform(0.05, 0.3, L))
    if "true_attack" not in d:
        d["true_attack"] = 0; d["attack_type"] = "normal"
    put("true_attack", 1); put("attack_type", kind)
    return d


if __name__ == "__main__":
    simulate().to_csv("process_data.csv", index=False)
    print("Saved process_data.csv")
