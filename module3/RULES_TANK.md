# Water-tank physics rules (auto-generated)

| Rule | Severity | Threshold | Description | Rationale |
|---|---|---|---|---|
| T1_LEVEL_HIGH | 3 (High) | > 70 | Tank level above the high alarm limit | Module 1 dictionary: level above 70 is a violation; the tank may overflow. |
| T2_LEVEL_LOW | 3 (High) | < 30 | Tank level below the low alarm limit | Very low level risks pump dry-run or means the sensor has failed. |
| T3_LEVEL_WARN | 2 (Medium) | 65 - 70 | Tank level in the warning band | Unusually high but not yet at the alarm limit. |
| T4_PRESSURE_HIGH | 3 (High) | > 40 PSI | Pressure above the alarm limit | Normal pressure is about 30 PSI; much higher stresses pipes and the tank. |
| T5_PRESSURE_LOW | 3 (High) | < 20 PSI | Pressure below the low limit | Loss of pressure: leak, pump failure or bad sensor. |
| T6_FLOW_RANGE | 2 (Medium) | outside 5 - 20 | Flow rate outside the plausible range | Normal flow is about 12.5; far outside is implausible. |
| T7_LEVEL_PRESSURE_MISMATCH | 2 (Medium) | |pressure - 0.6 x level| > 12 PSI | Pressure inconsistent with tank level | Bottom pressure follows water level (hydrostatic); a gap means one sensor is wrong or spoofed. |
| T8_VALVE_CLOSED_FLOW | 4 (Critical) | valve = 0 for 10+ samples and flow > 8 | Valve reported closed for 10+ s but flow continues | A closed valve cannot pass flow; valve status or flow reading is spoofed. |
| T9_FROZEN_SENSORS | 2 (Medium) | level and pressure unchanged for 20+ samples | All sensor readings frozen | Real readings always carry noise; zero variance suggests a replay attack. |
