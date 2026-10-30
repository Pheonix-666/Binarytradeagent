# Edge Decay Detection Power & Sample Size Reference

This reference documents the statistical power of the two-level edge decay detector in `adaptive/decay.py`.

## Theory & Detection Formula

Given:
- Break-even win rate $p_0 = \frac{1}{1 + \text{payout}}$ (with default payout $0.85$, $p_0 \approx 54.05\%$)
- Win-rate shortfall $\delta = p_0 - p_1$ (e.g. 5 points = $0.05$)
- Normal confidence critical value $z$ ($z=1.0$ for one-sided CAUTION, $z=1.96$ for 95% DEFENSIVE)

The minimum number of trades $N$ required to detect a degraded win rate $p_1$ below break-even is:

$$N \approx \left\lceil \frac{z^2 \cdot p_1 (1 - p_1)}{\delta^2} \right\rceil$$

---

## Trades to Detect Shortfall (Payout = 0.85, Break-Even = 54.05%)

| Shortfall ($\delta$) | Degraded Win Rate ($p_1$) | CAUTION ($z=1.0$, $w=60$) | DEFENSIVE ($z=1.96$, $w=100$) |
|:---:|:---:|:---:|:---:|
| **3 points (3.0%)** | 51.05% | **278 trades** | **1,067 trades** |
| **5 points (5.0%)** | 49.05% | **100 trades** | **384 trades** |
| **8 points (8.0%)** | 46.05% | **39 trades** | **149 trades** |
| **12 points (12.0%)** | 42.05% | **17 trades** | **65 trades** |

---

## Observations on Sensitivity

1. **CAUTION Sensitivity**:
   - With $z_{\text{caution}} = 1.0$ and rolling window $w_{\text{caution}} = 60$, any degradation exceeding $7$–$8$ percentage points triggers an immediate alert and escalation to `CAUTION` within $\le 40$ trades.
   - A $12$-point edge collapse triggers `CAUTION` in approximately $17$ trades.
2. **Defensive Persistence**:
   - `DEFENSIVE` requires edge decay to hold across $2$ consecutive non-overlapping windows of $100$ trades ($200$ trades total) at $z=1.96$, preventing premature over-reaction to short-term variance while catching structural regime shifts.
