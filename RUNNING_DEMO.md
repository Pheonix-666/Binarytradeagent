# Running Binomo Demo Mode Guide

This guide details the exact procedures to start, monitor, stop, keep Windows awake, and handle safety breaker trips when running the trading system in **Demo Mode** (`mode: demo`, `source: binomo_readonly`).

---

## 1. Prerequisites and Environment Verification

Before launching demo mode, verify that your environment variables and dependencies are properly configured without exposing any secret keys:

```powershell
.venv\Scripts\python.exe main.py check-env
```

Ensure that:
- `.env` contains valid credentials if LLM decision mode is selected.
- `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` are set (alerts and trade confirmation tickets require Telegram).
- `LIVE_CONFIRM` is **NOT** set to `YES` (real capital is never risked in demo mode).
- Playwright Chromium is installed:
  ```powershell
  .venv\Scripts\python.exe -m playwright install chromium
  ```

---

## 2. Starting Demo Mode

To launch the system in demo mode with the Binomo passive reader and trader:

```powershell
.venv\Scripts\python.exe main.py run --mode demo --source binomo_readonly
```

### What Happens at Startup:
1. **Startup Requirements Check**:
   - `binomo.ui` selectors are resolved and verified against the page DOM.
   - Platform stake limits are queried from `/platform/private/v2/currencies`.
   - `execution.demo_bypasses_paper_gate` is confirmed active (`true`).
   - Telegram configuration is validated.
2. **Persistent Headed Browser Window**:
   - A headed Chromium window opens using the persistent profile anchored at `state/playwright_profile`.
   - The browser navigates to `https://binomo.com/trading`.
3. **Manual Login Observation**:
   - The agent strictly adheres to read-only observation: credentials are **never** typed, stored, or logged by the software.
   - The console will display:
     `[HH:MM:SS] NOT LOGGED IN YET - log in manually in the browser window`
   - Log in manually to your account in the browser window.
   - The system detects successful login passively via WebSocket connection (`ws.binomo.com`) and topic joins.
4. **Strict Demo Mode Verification**:
   - The system verifies that `account_type == "demo"` in WebSocket `balance_changed` frames AND the DOM `demo_indicator` displays "Demo account".
   - If a real account is detected, startup aborts immediately.
5. **Operator Trade Confirmations (`confirm_first_n_trades`)**:
   - By default (`execution.confirm_first_n_trades: 5`), the first 5 trade opportunities require operator approval via Telegram (`/approve <ticket_id>`).
   - Once approved, subsequent trades execute automatically in demo mode.

---

## 3. Monitoring System Status

To check system health, current balance, open trades, and breaker status at any time from another terminal:

```powershell
.venv\Scripts\python.exe main.py status
```

You can also inspect the structured cycle logs and daily markdown summaries:
- **Cycle Logs**: `reports/daily/cycles.jsonl`
- **Daily Markdown Reports**: `reports/daily/daily_YYYY-MM-DD.md`
- **Application Logs**: `trading.log`

---

## 4. Stopping Gracefully

To stop the system gracefully without interrupting an in-flight trade or corrupting state:

### Method A: The KILL File (Recommended)
Touch or create the `KILL` file in the project root:

**PowerShell**:
```powershell
New-Item -ItemType File -Name KILL
```

**Command Prompt**:
```cmd
type nul > KILL
```

**CLI Tool**:
```powershell
.venv\Scripts\python.exe main.py kill
```

The orchestrator checks for the `KILL` file on every cycle, cleanly closes open resources, shuts down the Playwright session, and removes the `KILL` file upon exit.

### Method B: Terminal Signal
Press `Ctrl+C` in the terminal running `main.py`. The signal handler initiates a graceful shutdown sequence.

---

## 5. Keeping Windows Awake

Demo runs must not be interrupted by Windows going into sleep or standby mode. Use one of the following methods:

### Method A: PowerCFG Commands (System-wide)
Disable standby timeout while plugged into AC power:
```powershell
powercfg /change standby-timeout-ac 0
powercfg /change monitor-timeout-ac 0
```
*(To restore later: `powercfg /change standby-timeout-ac 30`)*

### Method B: PowerShell Keep-Awake Loop (Terminal-based)
Run this single-line keep-awake script in a separate PowerShell window:
```powershell
powershell -NoProfile -Command "$wscript = New-Object -ComObject Wscript.Shell; while ($true) { $wscript.SendKeys('{F15}'); Start-Sleep -Seconds 60 }"
```
This periodically simulates pressing the innocuous F15 key every 60 seconds, keeping the session active without altering user input.

---

## 6. What to Do After a Circuit Breaker Trip

If a critical condition occurs, the system's `CircuitBreaker` will trip and halt all trading operations immediately.

### Common Causes of a Breaker Trip:
1. **`AmbiguousOrder`**: A trade button was clicked, but no Phoenix `opened` confirmation event arrived within `binomo.confirm_timeout_seconds` (default: 15s). The trade is marked `PENDING` and **never retried**.
2. **`RealAccountDetected`**: A WebSocket frame or DOM element indicated a real/live account (`deal_type != "demo"`).
3. **Data Staleness / HTTP 429**: WebSocket frames stopped arriving or API returned rate limiting errors.
4. **Panel Blocked**: An unexpected modal popup or verification dialog blocked the trading controls.

### Recovery and Reset Procedure:
1. **Review Notifications**: Check Telegram alerts and `trading.log` for the exact error message and timestamp.
2. **Inspect the Browser**: Look at the headed browser window to see the actual state of the Binomo trading panel.
3. **Reconcile Open Trades**:
   - Check `state/trading_state.json` under `"open_trades"`.
   - Verify if the trade was actually opened on the broker platform.
   - If the trade completed, update state or allow reconciliation.
4. **Remove Obstructions**: If a modal popup appeared (e.g., promotional banner), dismiss it manually in the browser.
5. **Restart System**: Once the root cause is resolved and verified, restart the system using `python main.py run --mode demo --source binomo_readonly`.

