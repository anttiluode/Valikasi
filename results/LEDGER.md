| gate | verdict |
|---|---|
| V0 | PASS |
| V4 | FAIL |
| V1 | VOID |
| V2 | VOID |
| V3 | VOID |
| V5 | PASS |
| V6 | not run (no --model-c) |

| set | n | text acc | hidden acc | delta [95% CI] | top-1 | text bits (Fano) |
|---|---|---|---|---|---|---|
| L1_all | 240 | 0.292 | - | +0.048 [+0.016, +0.085] | 0.312 | 0.006 |
| L1_clean | 176 | 0.244 | 0.244 | +0.013 [-0.016, +0.040] | 0.290 | 0.000 |
| L2_all | 240 | 0.263 | 0.263 | -0.004 [-0.033, +0.023] | 0.242 | 0.001 |
| L2_clean | 184 | 0.217 | 0.255 | -0.014 [-0.044, +0.014] | 0.234 | 0.004 |
| NOB_clean | 40 | 0.175 | 0.225 | +0.029 [-0.038, +0.094] | 0.325 | 0.023 |

leave-script-out accuracy (reported only, unseen B conversations): L1_clean: text 0.256, hidden 0.239; L2_all: text 0.254, hidden 0.263; L2_clean: text 0.250, hidden 0.255

survival (note has no lexicon word): {"L1": {"red": 0.75, "blue": 0.7333333333333333, "green": 0.7333333333333333, "yellow": 0.7666666666666667}, "L2": {"red": 0.8666666666666667, "blue": 0.8333333333333334, "green": 0.7666666666666667, "yellow": 0.8}, "NOB": 0.8166666666666667}
A ignored B entirely in L2 (identical note across all four Z): 0.05
empty notes: 0, notes mentioning B: 0.03
