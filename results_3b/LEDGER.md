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
| L1_all | 240 | 0.225 | - | +0.013 [-0.030, +0.056] | 0.263 | 0.002 |
| L1_clean | 168 | 0.226 | 0.244 | -0.007 [-0.039, +0.024] | 0.250 | 0.002 |
| L2_all | 240 | 0.275 | 0.283 | +0.027 [-0.013, +0.068] | 0.275 | 0.002 |
| L2_clean | 172 | 0.238 | 0.262 | +0.013 [-0.023, +0.048] | 0.267 | 0.001 |
| NOB_clean | 40 | 0.100 | 0.325 | +0.016 [-0.071, +0.097] | 0.275 | 0.105 |

leave-script-out accuracy (reported only, unseen B conversations): L1_clean: text 0.244, hidden 0.244; L2_all: text 0.342, hidden 0.317; L2_clean: text 0.308, hidden 0.302

survival (note has no lexicon word): {"L1": {"red": 0.7166666666666667, "blue": 0.7333333333333333, "green": 0.7, "yellow": 0.7}, "L2": {"red": 0.7333333333333333, "blue": 0.7166666666666667, "green": 0.75, "yellow": 0.75}, "NOB": 0.75}
A ignored B entirely in L2 (identical note across all four Z): 0.15
empty notes: 0, notes mentioning B: 0.00
