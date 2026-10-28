# Pattern Null Distribution Report

Diagnostic evaluation of pattern similarity on synthetic random-walk price series.

- **Series evaluated:** 2000
- **Timeframes:** 5m, 15m
- **Window lengths:** [30, 45, 60, 90]
- **Total window evaluations per template:** 16000
- **Random seed:** 42
- **Step volatility:** 1.5 pips (0.00015)

| Template | Direction | Type | >= 60 Sim (%) | >= 80 Sim (%) | Mean Sim | 95th Percentile |
|---|---|---|---|---|---|---|
| `double_bottom` | up | reversal | 10.57% | 0.57% | 23.62 | 66.42 |
| `double_top` | down | reversal | 10.53% | 0.35% | 23.80 | 65.94 |
| `support_bounce` | up | continuation | 26.74% | 9.81% | 29.91 | 85.82 |
| `resistance_bounce` | down | continuation | 26.86% | 10.16% | 30.03 | 86.06 |

## Key Observations
- Random-walk windows produce zero similarity for opposite-trend shapes due to negative correlation clipping.
- High-similarity false positives (>= 80%) on random walks remain rare across all built-in templates.
- Threshold >= 60% filters out the vast majority of unstructured noise windows.