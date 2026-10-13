"""Simulates an 11 kV substation feeder (offline, safe) with injected attacks.
Also produces a MOCK Module 2 output (ai_score) so Module 3 runs standalone."""
import numpy as np
import pandas as pd


def simulate(n=3000, seed=42):
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


if __name__ == "__main__":
    simulate().to_csv("process_data.csv", index=False)
    print("Saved process_data.csv")
