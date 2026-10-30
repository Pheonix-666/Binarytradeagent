# Automated Fixed-Time Trading System: Project Spec

Version 1.0. Read this entire file before writing any code.

---

## 0. How to use this file

1. Put this file in the project root as `PROJECT_SPEC.md`.
2. Tell the coding agent: *"Read PROJECT_SPEC.md fully. Build one phase at a time (Section 12). Stop after each phase, summarize what you built, and wait for my approval."*
3. If your tool has a rules or instructions file, paste Section 1 into it as well.
4. Each phase in Section 12 has a ready prompt. Paste one at a time.

---

## 1. Rules for the coding agent (non-negotiable)

1. Build one phase at a time. Stop and summarize after each one.
2. **Default mode is paper.** Live trading requires `mode: live` in config AND the environment variable `LIVE_CONFIRM=YES`.
3. Never put secrets (passwords, API keys, tokens) in code or config. Use environment variables and a `.env` file listed in `.gitignore`.
4. Files in `safety/` and `execution/` are protected. Do not modify them in later phases unless the user explicitly asks. Show a diff for any change.
5. Every module ships with unit tests. `safety/` and `execution/` need a test for every rule, including tests proving forbidden orders are blocked.
6. **No look-ahead.** Any function that computes a signal at time `t` may only use data with timestamp `<= t`. Add tests that enforce this.
7. Every external call (platform, LLM, alerts) has a timeout and error handling. On any doubt, the system does nothing (SKIP) and sends an alert.
8. Do not add anti-detection, fake-human behavior, or fingerprint masking to the platform adapter. Only read and click what is needed.
9. All tunable numbers live in `config.yaml`. No magic numbers in logic.
10. Log every decision with its full inputs and outputs (JSONL).
11. Stack: Python 3.11+, type hints, numpy, pandas, pydantic, pyyaml, pytest, Playwright, SQLite.

---

## 2. Fixed design decisions

| Topic | Decision |
|---|---|
| Product | Fixed-time trades (CALL/PUT) on a binary-style platform |
| Expiry | 15, 30, or 60 minutes. Stored as a setting; the backtest decides which works best |
| Asset selection | Top 5 assets, re-picked every 2 hours (see 6.4) |
| OTC assets | Allowed, but an asset trades real money only after passing the paper-trading gate (Section 10) |
| Timeframes | Daily + 1h = direction, 15m = confirmation, 5m = check cadence, 1m = entry confirmation candle |
| Entry check | Every 5 minutes, right after the 5m candle closes |
| Trend rule | **Strict gate**: trade direction must match daily and 1h trend, else SKIP |
| Reversal patterns | Two modes built, backtest decides: `off` or `sr_bypass` (see 6.3) |
| Stake range | 1% to 20% of balance per trade, 35% total open exposure |
| Stake from risk | Higher risk means a lower maximum stake (formula in 6.5) |
| Multiple trades | Several trades on the same asset allowed, within the 35% cap. Never CALL and PUT on the same asset at the same time |
| Daily loss limit | Setting exists, **default off** (`null`) |
| Always-on protections | Malfunction breaker, kill switch, alerts |
| Decision maker | An LLM follows the process and decides; code validates and executes (Section 7) |
| LLM provider | Swappable (`anthropic` or `gemini`), one client file |
| Platform access | Browser automation (Playwright) behind a swappable adapter |
| Runs on | The user's own PC |
| Patterns | Plug-in folder of templates. A few built-in shapes until the user supplies pattern images |

---

## 3. Data flow

```
every 2h:  AssetSelector -> top 5 assets
every 5m:  for each asset in top 5:
             candles -> TrendAnalyzer (daily, 1h, 15m labels)
                     -> Behavior (volatility, ADX, S/R levels)
                     -> PatternLibrary (similarity scores)
                     -> StrictGate (pass / SKIP)
                     -> RiskEngine (risk 1-100, max stake %)
           candidates that passed -> LLM Decision (JSON)
                                  -> Validator (code, cannot be overridden)
                                  -> Safety checks
                                  -> Execution (paper / demo / live)
every 1m:  track open trades, record results, update stats
always:    malfunction checks, kill switch, alerts, JSONL logging
```

---

## 4. Folder structure

```
trading_system/
  PROJECT_SPEC.md
  config.yaml
  .env.example
  .gitignore
  main.py
  core/            config.py, models.py, logging_setup.py, clock.py
  adapters/        base.py, paper.py, playwright_adapter.py
  data/            recorder.py, candle_builder.py, store.py
  analysis/        indicators.py, trend.py, behavior.py, levels.py
  patterns/        library.py, similarity.py, builtin.py, templates/  (user adds .npy + meta)
  risk/            engine.py, stake.py
  selection/       asset_selector.py, stats.py
  decision/        llm_client.py, prompt.py, schema.py, validator.py
  safety/          limits.py, breaker.py, killswitch.py          (PROTECTED)
  execution/       trade_manager.py, reconcile.py, state.py      (PROTECTED)
  alerts/          telegram.py
  backtest/        engine.py, metrics.py, run_backtest.py
  tools/           discover_platform.py, digitize_patterns.py
  tests/
  logs/            (JSONL decision logs)
  state/           (state.json, trades.sqlite)
```

---

## 5. config.yaml (reference)

```yaml
mode: paper                 # record | paper | demo | live
live_requires_env: LIVE_CONFIRM

trading:
  expiry_minutes: 30        # 15 | 30 | 60
  allowed_expiries: [15, 30, 60]
  check_interval_minutes: 5
  asset_reselect_hours: 2
  top_assets: 5
  max_measured_slots: 3     # of the 5 slots, max filled by assets with proven stats
  min_trades_for_stats: 30

stake:
  min_pct: 1
  max_pct: 20
  total_cap_pct: 35
  daily_loss_limit_pct: null   # null = off
  min_amount: 1.0
  max_amount: 1000.0           # platform maximum single trade stake cap
  round_to: 1.0

gate:
  strict_trend_gate: true
  reversal_mode: off           # off | sr_bypass
  trend:
    ema_fast: 20
    ema_slow: 50
    adx_period: 14
    adx_min: 20
  reversal_bypass:
    min_similarity: 80
    max_risk_score: 40
    require_sr_level: true
    require_confirmation_candle: true
    max_volatility_percentile: 90

patterns:
  min_similarity: 60
  template_points: 64
  window_lengths: [30, 45, 60, 90]   # candles, resampled to template_points
  dtw_shortlist: 3
  allow_trade_without_pattern: false

risk:
  weights:                    # start equal, tune later via backtest
    volatility: 0.2
    pattern: 0.2
    win_rate: 0.2
    session: 0.2
    sr_distance: 0.2
  volatility_lookback_days: 30
  atr_period: 14

paper_gate:
  min_paper_trades: 200
  min_edge_margin: 0.02       # win rate must exceed break-even by this much

safety:
  max_orders_per_10min: 6
  stale_data_multiplier: 2    # data older than 2x check interval pauses trading
  max_consecutive_exec_errors: 3
  kill_file: KILL

llm:
  provider: anthropic         # anthropic | gemini
  model: claude-sonnet-5-5    # for gemini, set the model name you have access to
  temperature: 0
  max_tokens: 800
  timeout_seconds: 20
  max_decision_age_seconds: 30

alerts:
  telegram_enabled: true      # needs TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in env

backtest:
  default_payout: 0.85
  train_fraction: 0.7
```

---

## 6. Module specs

### 6.1 Platform adapter (`adapters/base.py`)

All platform access goes through this interface. Nothing else in the system knows which adapter is active.

```python
class PlatformAdapter(ABC):
    def get_assets(self) -> list[AssetQuote]: ...        # asset_id, name, is_otc, payout (0-1), is_open
    def get_candles(self, asset_id, timeframe, n) -> DataFrame: ...  # ts, open, high, low, close
    def get_balance(self) -> float: ...
    def place_trade(self, asset_id, direction, expiry_min, stake_amount) -> TradeReceipt: ...
    def get_open_trades(self) -> list[Trade]: ...
    def get_trade_history(self, since) -> list[Trade]: ...
    def is_healthy(self) -> bool: ...
```

- **PaperAdapter:** uses recorded or live-read candles. Simulates trades: settles by comparing the entry price to the price at expiry, using the payout recorded at entry. No clicks.
- **PlaywrightAdapter:** see Phase 1 (discovery, read-only) and Phase 10 (trading). The user logs in manually in a persistent browser profile; credentials are never in code.
- **Candle builder:** if the platform only exposes ticks, build 1m candles from ticks, then resample to 5m/15m/1h/1d. Only closed candles are used for signals.

### 6.2 Trend analyzer (`analysis/trend.py`)

For each of daily, 1h, 15m, label the trend as `up`, `down`, or `sideways`:
- `up`: close > EMA(slow) AND EMA(fast) > EMA(slow) AND ADX >= `adx_min`
- `down`: the inverse
- otherwise `sideways`

### 6.3 Strict gate (`analysis/trend.py`)

- A **CALL** passes only if daily = up AND 1h = up AND 15m is not down.
- A **PUT** passes only if daily = down AND 1h = down AND 15m is not up.
- Anything else: SKIP, with the reason logged.

**Reversal exception** (`reversal_mode: sr_bypass`): a reversal pattern (e.g. U-shape, double bottom, double top) may bypass the gate only if ALL are true:
- similarity >= `reversal_bypass.min_similarity`
- price is at a support level (for CALL) or resistance level (for PUT)
- volatility percentile <= `max_volatility_percentile`
- risk score <= `max_risk_score`
- the latest 1m or 15m confirmation candle points the same way

With `reversal_mode: off`, reversal patterns never trade. The backtest runs both modes (Section 11). Each reversal pattern is also auto-disabled if its measured win rate falls below break-even.

### 6.4 Asset selection (`selection/asset_selector.py`)

Runs every `asset_reselect_hours`.
- Universe: assets currently open for fixed-time trades, with live payout.
- `break_even = 1 / (1 + payout)`
- `EV per trade = win_rate * payout - (1 - win_rate)`
- **Tier 1:** assets with at least `min_trades_for_stats` recorded trades, ranked by EV.
- **Tier 2:** assets without enough stats, ranked by payout (these are paper-only until they pass the gate).
- Fill up to `max_measured_slots` from Tier 1, the remaining slots from Tier 2, so new assets keep building a record.
- Also compute a correlation map between assets on real-market data only. For OTC assets, correlation is informational and never used as a trading rule.

### 6.5 Risk engine (`risk/engine.py`, `risk/stake.py`)

Each input becomes a 0-100 **risk sub-score** (higher = riskier). The final risk score is the weighted average (weights from config), rounded and clamped to 1-100.

| Input | Sub-score |
|---|---|
| Volatility | Percentile of current ATR(5m) versus the asset's own last `volatility_lookback_days` |
| Pattern | `100 - best_similarity` for the trade direction |
| Win rate | `100 * clip((break_even + 0.15 - win_rate) / 0.20, 0.01, 1)`. If trades < `min_trades_for_stats`: 100 |
| Session | Risk by hour of day from the system's own historical results. Until enough data exists: 50 |
| S/R distance | Distance to the nearest opposing level in the trade direction, in ATR units: under 0.5 ATR = 90, 3 ATR or more = 10, linear between |

**Stake from risk:**

```
max_stake_pct = 20 - (risk - 1) / 99 * 19      # risk 1 -> 20%, risk 50 -> 10.60%, risk 100 -> 1%
```

(use `stake.max_pct` and `stake.min_pct` from config in place of 20 and 1). The LLM may choose any stake between `min_pct` and `max_pct_for_this_risk`, never above. Total open stakes may never exceed `total_cap_pct` of the balance. Position amounts are floored to `stake.round_to`, capped at platform `stake.max_amount`, and skipped if below `stake.min_amount`.

### 6.6 Pattern library (`patterns/`)

- A pattern is a template: a 1-D array of `template_points` values, z-normalized, stored as `.npy`, plus a metadata JSON: `name`, `direction` (up/down), `type` (continuation/reversal), `timeframe_hint`.
- At startup, load all templates once into one matrix.
- **Scoring (fast path):** take the latest candles at each window length, resample to `template_points`, z-normalize, and compute correlation against all templates in one vectorized operation. Convert to 0-100.
- **Refine:** run DTW only on the top `dtw_shortlist` matches.
- Output per asset: list of `{pattern, direction, type, similarity}` sorted best-first.
- **Built-in shapes** (generated by code, no images): double top, double bottom, support bounce, resistance bounce.
- With an empty template folder and no built-ins matching, there are simply no candidates.
- **Image conversion tool** (`tools/digitize_patterns.py`, later): converts candlestick screenshots into templates. Crop to the chart area, detect up/down candle colors, read open/high/low/close from pixel positions, normalize, save. It must also write an overlay image (extracted candles drawn on the source) so the user can verify each template visually. Known failure causes: drawn lines or text on the chart, grid lines in candle colors, low resolution.

### 6.7 State, logging, alerts

- **State:** open trades, today's results, per-asset stats, and breaker status are saved after every change using an atomic write (write temp file, then rename). Trades are also stored in SQLite.
- **Startup reconcile:** on start, read state, then read the platform's trade history, and fix any trade whose result happened while the system was offline. No new trades until reconcile finishes.
- **Logging:** one JSONL line per check cycle per asset (inputs, gate result, risk, LLM output, validator result, action).
- **Alerts (Telegram):** trade placed, trade result, breaker or kill switch fired, errors, startup and reconcile complete, system back online.

---

## 7. LLM decision spec

The LLM is called only when code has produced at least one candidate (gate passed and a pattern or reversal exception qualifies). It decides by following the process and returns JSON only. Code validates and executes.

### 7.1 System prompt (starting version)

```
You are the decision step of an automated fixed-time trading process.
You receive a JSON report produced by code. Follow these steps in order:

1. Market: confirm the asset is in the selected top list and note its payout.
2. Trend: confirm daily and 1h trend support the direction (the report already
   applied the gate; confirm, do not override it).
3. Behavior: check volatility and nearby support/resistance.
4. Pattern: check the best pattern match and its similarity.
5. Risk: use the provided risk_score and max_stake_pct. Never exceed max_stake_pct.
6. Decide: choose ONE candidate to trade, or SKIP.

Rules:
- Use only the numbers in the report. Never invent or estimate numbers.
- You may SKIP any time. SKIP is correct when evidence conflicts or is weak.
- Never choose an asset, expiry, or direction that is not in the candidates.
- stake_pct must be between min_stake_pct and the candidate's max_stake_pct.
- Output ONLY valid JSON matching the schema. No other text.
```

### 7.2 Input report (built by code)

```json
{
  "timestamp": "...",
  "balance": 1000.0,
  "open_exposure_pct": 12.0,
  "remaining_exposure_pct": 23.0,
  "min_stake_pct": 1,
  "candidates": [
    {
      "asset": "EURUSD",
      "direction": "CALL",
      "expiry_minutes": 30,
      "payout": 0.85,
      "trend": {"daily": "up", "1h": "up", "15m": "up"},
      "volatility_percentile": 55,
      "nearest_level": {"type": "support", "distance_atr": 1.8},
      "pattern": {"name": "double_bottom", "type": "reversal", "similarity": 82},
      "risk_score": 38,
      "max_stake_pct": 12.9,
      "asset_stats": {"trades": 240, "win_rate": 0.57, "mode": "live_eligible"}
    }
  ]
}
```

### 7.3 Output schema

```json
{
  "decision": "CALL | PUT | SKIP",
  "asset": "EURUSD",
  "expiry_minutes": 30,
  "stake_pct": 10.0,
  "evidence": {
    "risk_score": 38,
    "similarity": 82,
    "trend_daily": "up",
    "trend_1h": "up",
    "payout": 0.85
  },
  "reasoning": "max 60 words"
}
```

### 7.4 Validator (`decision/validator.py`), must pass ALL or the result is SKIP

1. Output parses as JSON and matches the schema exactly.
2. `decision` is one of CALL, PUT, SKIP.
3. For CALL/PUT: asset, direction, and expiry match one of the candidates.
4. `stake_pct` is within `[min_stake_pct, candidate.max_stake_pct]` and within `remaining_exposure_pct`.
5. Every number in `evidence` equals the real value in the report (catches misquoted or invented figures).
6. The decision is not older than `max_decision_age_seconds`, and current price has not moved against the signal beyond a configured tolerance.
7. No opposite-direction open trade exists on the same asset.

### 7.5 LLM client (`decision/llm_client.py`)

One abstract class with `AnthropicClient` and `GeminiClient`, chosen by `llm.provider`. Temperature 0. Timeout and retry once, then SKIP.

---

## 8. Safety layer (`safety/`, PROTECTED)

Plain code. The LLM cannot read, change, or bypass it.

- **Stake limits:** each stake between `min_pct` and `max_pct` of balance, total open stakes at most `total_cap_pct`.
- **No opposite trades** on the same asset at once.
- **Malfunction breaker:** pause trading and alert if any of these happen:
  - more than `max_orders_per_10min` orders in 10 minutes
  - price data older than `stale_data_multiplier` x check interval
  - `max_consecutive_exec_errors` execution errors in a row
  - balance read fails or changes in a way that doesn't match recorded trades
- **Kill switch:** the system halts new trades when a file named `KILL` exists in the project folder or a Telegram `/stop` command arrives. Open trades are left to settle.
- **Daily loss limit:** if `daily_loss_limit_pct` is not null, stop for the day when the day's loss reaches it. Default is off.
- **Mode guard:** `live` mode requires `LIVE_CONFIRM=YES`.
- On startup, the system refuses to trade until reconcile completes.

---

## 9. Execution (`execution/`, PROTECTED)

Pre-trade sequence, in order. Any failure means no trade plus an alert:

1. Kill switch not active, breaker not tripped, reconcile complete.
2. Validator passed (Section 7.4).
3. Safety limits passed (Section 8).
4. Asset mode check: real money only if the asset passed the paper gate (Section 10); otherwise record a **paper shadow trade** instead.
5. Place the trade through the adapter (one order per decision).
6. **Verify:** read back the open-trades list and confirm the trade exists with the right asset, direction, expiry, and stake. If it can't be confirmed: stop, alert, trip the breaker.
7. Save to state and SQLite. Alert.

Never retry an order automatically. After a result, update asset stats and the session statistics.

---

## 10. Paper-trading gate

- Every asset starts in `paper` mode. The system takes shadow trades on it using the same process.
- An asset becomes `live_eligible` only when: paper trades >= `min_paper_trades` (200) AND win rate > break-even + `min_edge_margin`.
- It drops back to paper mode if its rolling win rate (last 100 trades) falls below break-even.
- Always report the win rate together with a 95% confidence interval (Wilson). If the interval includes break-even, label the edge "unproven".
- In `record` mode, the system only collects candles and payouts.

---

## 11. Backtesting (`backtest/`)

- Replay historical candles in 5-minute steps; at each step the system sees only data up to that time (tests must enforce this).
- Use the same code path as live: trend, gate, patterns, risk, stake. For the LLM step, support two options: a deterministic stand-in (take the best candidate) for fast tests, and optionally the real LLM on a sample of steps.
- Settlement: compare the entry price at the step to the price at expiry. Win pays `stake * payout`, loss costs `stake`.
- Use `default_payout` or recorded payouts; include a spread/slippage assumption.
- **Variants to compare:** `reversal_mode` off vs `sr_bypass`, and expiry 15 vs 30 vs 60. Keep the variant count small to limit lucky results.
- **Split:** choose rules on the first `train_fraction` of the data, then report results on the remaining data the rules never saw. Walk-forward if enough data.
- **Report:** trades, win rate with 95% CI, break-even rate, EV per trade, equity curve under the stake rules, max drawdown, and breakdowns by asset, pattern, session, and risk bucket.
- For OTC assets, backtest only on the platform's own recorded candles after enough recording time.

---

## 11.5. Adaptive layer (`adaptive/`)

A pure regulatory layer that dynamically adjusts system aggressiveness based on real-time performance telemetry.

### Core Principles
1. **Tightening Only:** The adaptive layer may ONLY tighten risk, switch to pre-approved profiles, or pause. It never increases a stake after a loss, never loosens a threshold relative to the base configuration, and never touches hard safety limits (stake 1–20%, total open exposure 35%).
2. **Dual Speed Dynamics:**
   - **Fast Reactions (size reduction & pausing):** Triggered immediately by peak drawdown and consecutive loss streaks.
   - **Slow Reactions (profile switching & context suspension):** Require statistically significant evidence and mandatory cooldown periods. No adaptation is performed on fewer trades than configured sample minimums.
3. **Pure Logic & Determinism:** Explicit inputs (`equity`, `closed_trades`, `now`, `prev_state`) and outputs, ensuring identical execution in backtesting and live operations without disk or network I/O inside decision loops.
4. **Serialization:** `AdaptiveController` and `AdaptiveState` implement `to_dict()` and `from_dict()`.

### Modules
- **`adaptive/monitor.py`:** State machine operating across four tiers: `NORMAL` $\to$ `CAUTION` $\to$ `DEFENSIVE` $\to$ `PAUSED`. Escalation is immediate upon breaching drawdown or streak thresholds. Recovery moves down strictly ONE tier at a time with a configured hysteresis fraction. `PAUSED` persists until requalification passes ($\ge 30$ shadow trades beating break-even), followed by a 20-trade ramp-up at half stakes.
- **`adaptive/profiles.py`:** Pre-approved profile overrides limited to a strict whitelist of keys with a clear monotonic tightening order: `reversal_mode` (can only tighten to "off"), `patterns.min_similarity` (can only increase), `max_risk_cap` (can only decrease), and `stake_scale` (can only decrease $\le 1.0$). Shipped profiles: `standard` and `conservative`. Computes `effective_config` guaranteeing every parameter is at least as strict as the base config, and hard safety limits (stake 1–20%, total cap 35%, max_amount) are strictly untouched.
- **`adaptive/champion.py`:** Champion-challenger shadow tracker. Expensive per-step operations (indicators, trend detection, DTW pattern matching) are computed once and shared; each profile applies its filters post-hoc. The live champion switches only when a challenger achieves $\ge 50$ shadow trades, Wilson 95% CI lower bound $>$ break-even, $\ge 0.02$ EV advantage, and cooldown $\ge 3$ days.
- **`adaptive/contexts.py`:** Tags every trade across dimensions `[asset, hour_bucket, pattern, risk_bucket, volatility_bucket, expiry]`. Calculates Wilson 99% CI upper bound ($z=2.58$). If $n \ge 30$ and upper bound $<$ break-even, the context is suspended (capped at at most 30% of observed contexts). Candidates matching suspended contexts are skipped with explicit reasons.
- **`adaptive/postmortem.py`:** Diagnostic Markdown generator analyzing recent losses, tier transitions, and context health. Never mutates state or config.

---

## 12. Build phases and prompts

Paste one prompt at a time. After each phase: run the tests, read the summary, and only then continue.

### Phase 0: Skeleton

> Read PROJECT_SPEC.md fully. Build Phase 0: the folder structure from Section 4, `config.yaml` from Section 5 with a pydantic config loader that validates every value, JSONL logging, a `.env.example`, a `.gitignore` that excludes `.env`, `logs/` and `state/`, and a pytest setup. Include tests for config validation (reject out-of-range values like `min_pct` > `max_pct`). Follow all rules in Section 1. Stop and summarize when done.

### Phase 1: Data, adapters, and platform discovery

> Build Phase 1. Implement the `PlatformAdapter` base class from 6.1, the `PaperAdapter`, a candle builder (ticks to 1m, resample to 5m/15m/1h/1d, closed candles only), and a recorder that saves candles and payouts to SQLite. Then write `tools/discover_platform.py`: it opens the trading platform in a persistent Playwright browser profile, waits for me to log in manually, and then READ-ONLY logs which network or websocket messages and which page elements carry prices, the asset list, payouts, and balance. It writes a `discovery_report.md`. It must never click any trade button. Add tests for the candle builder, including partial-candle handling.

### Phase 2: Indicators, trend, and gate

> Build Phase 2: EMA, ATR, ADX, swing-based support/resistance levels, the trend labeler from 6.2, and the strict gate with the reversal exception from 6.3. Add tests: known synthetic series for each indicator and each trend label, every gate case (CALL/PUT with matching and conflicting trends, sideways), and a no-look-ahead test proving the result at time t does not change when later candles are added.

### Phase 3: Pattern library

> Build Phase 3 from 6.6: template loading, the vectorized similarity function, DTW refinement for the top matches, the built-in shapes, and the output format. Add tests: a template matches itself at ~100, a time-stretched copy scores high, noise scores low, and a loop over 4 assets x 4 window lengths x 20 templates runs in under 50 ms. Leave `tools/digitize_patterns.py` for later.

### Phase 4: Risk engine and stake

> Build Phase 4 from 6.5: each sub-score, the weighted average from config, and the stake formula. Add tests for boundary values (risk 1 gives 20%, risk 100 gives 2%, risk 50 gives about 11%), unknown-data defaults (win rate with too few trades gives 100), and weight handling. Keep all numbers in config.

### Phase 5: Asset selector and stats

> Build Phase 5 from 6.4 and Section 10's stats: per-asset win rate, EV, the two tiers with `max_measured_slots`, break-even from payout, and Wilson confidence intervals. Add tests for ranking, tiers, and EV math with known numbers.

### Phase 6: Backtester

> Build Phase 6 from Section 11. Include a test that deliberately introduces look-ahead and confirms the engine catches it. Provide `run_backtest.py` that compares reversal_mode off vs sr_bypass and expiry 15/30/60 on a train/test split, and prints the report. Use a deterministic stand-in for the LLM step.

### Phase 7: LLM decision and validator

> Build Phase 7 from Section 7: the report builder, the prompt, `AnthropicClient` and `GeminiClient` behind one interface selected by config, JSON schema parsing, and the validator with all seven checks. Add tests that feed bad outputs (wrong asset, stake too high, misquoted evidence, invalid JSON, stale decision, opposite-direction conflict) and confirm each one results in SKIP.

### Phase 8: Safety, execution, state, alerts (PROTECTED files)

> Build Phase 8 from Sections 8, 9 and 6.7: stake and exposure limits, the malfunction breaker, the kill switch (file and Telegram), the optional daily loss limit, the mode guard, the pre-trade sequence with read-back verification, atomic state saving, startup reconcile, and Telegram alerts. Write a test for every single rule, including tests that an oversized order, a duplicate order, and a trade during reconcile are all blocked. Mark these files as protected.

### Phase 9: Orchestrator and the record + paper run

> Build Phase 9: `main.py` with the scheduler from Section 3 (2-hour asset selection, 5-minute entry checks, 1-minute trade tracking). Support `record` and `paper` modes. Run end to end with the PaperAdapter. Produce a daily summary file with trades, win rate with CI, EV, and per-asset status.

*Then run record + paper mode for days or weeks, until assets accumulate their trades. Review the summaries before moving on.*

### Phase 10: Playwright trading adapter (demo account first)

> Build Phase 10: implement the `PlaywrightAdapter` using `discovery_report.md`: reading prices, payouts, and balance, and `place_trade`, `get_open_trades`, `get_trade_history`. It uses my manually logged-in persistent profile. After every click, read back the open trades to verify. Pause and alert if the page looks different from what is expected (missing element, popup, logged out). Test on the platform's DEMO account only, with minimal stakes, in `demo` mode. No anti-detection behavior.

### Phase 11: Go-live checklist (human)

Do not run `live` mode until all of these are true:
- Backtest on unseen data shows win rate above break-even, with the confidence interval reviewed
- Assets have passed the paper gate (Section 10)
- Demo-account run completed with correct trade verification
- Every file in Section 13 reviewed by you
- Telegram alerts and the kill switch tested
- Start with a small balance

---

## 13. Human review checklist

Read these yourself, line by line. Everything else you can skim.

1. `safety/*` and `execution/*`: one order per decision, correct direction and units, no automatic order retries, hard limits cannot be bypassed.
2. `decision/validator.py`: every check present; invalid output always results in SKIP.
3. `adapters/playwright_adapter.py` `place_trade`: right asset, right direction, right stake, read-back verification.
4. Tests for no look-ahead (indicators, trend, backtest).
5. `config.yaml`: stake range, cap, thresholds, `mode`, and that `live` needs `LIVE_CONFIRM`.
6. API keys: only in environment variables.
7. Backtest candle handling: closed candles only; fees and payout included.

---

## 14. Open items for later

- Pattern images: convert the user's candlestick screenshots with `tools/digitize_patterns.py` (provide 3 to 5 examples per pattern; no drawn lines on charts).
- Tune risk weights from backtest results.
- Optional: add a loss-based stop (`daily_loss_limit_pct`) after seeing real results.
- Optional: move from the user's PC to an always-on server (only the adapter's environment changes).
- Revisit the stake range after paper and demo results.
