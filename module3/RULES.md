# Physics rules (auto-generated)

| Rule | Severity | Threshold | Description | Rationale |
|---|---|---|---|---|
| R1_VOLTAGE_RANGE | 3 (High) | 10.45 - 11.55 kV | Voltage outside +-5% of nominal | Distribution voltage is normally held within +-5% of 11 kV. |
| R2_FREQUENCY_RANGE | 3 (High) | 49.5 - 50.5 Hz | Frequency outside 49.5-50.5 Hz | Grid frequency is tightly regulated around 50 Hz. |
| R3_OVERCURRENT | 3 (High) | > 400 A | Current above safe limit | Exceeds feeder/conductor rating (protection should trip). |
| R4_OVERTEMP | 3 (High) | > 85 C | Transformer temperature too high | Above typical winding/oil limit; risks insulation damage. |
| R5_PF_RANGE | 2 (Medium) | 0.80 - 1.00 | Power factor implausible | Power factor must be within 0.80-1.00 for this feeder. |
| R6_POWER_MISMATCH | 3 (High) | > 0.40 MW | Reported power inconsistent with V x I x pf | P = sqrt(3)*V*I*pf is a physical identity; a gap suggests data injection. |
| R7_BREAKER_OPEN_FLOW | 4 (Critical) | breaker=0 and I > 5 A | Breaker open but current still flowing (spoofed state/sensor) | An open breaker cannot carry load current. |
| R8_RATE_OF_CHANGE | 2 (Medium) | > 80 A/sample | Current changed faster than physically plausible | Load cannot jump by more than di_max in one sample. |
| R9_FROZEN_SENSOR | 2 (Medium) | std < 1e-6 over 20 samples | Sensor values frozen (possible replay attack) | Real measurements always carry noise; zero variance suggests replay. |
| R10_POWER_DRIFT | 3 (High) | CUSUM of |P - calc| > 1.5 MW (k = 0.1) | Reported power slowly drifting from V x I x pf (stealthy injection) | Small biases stay under R6's limit but add up; a CUSUM of the residual catches them. |