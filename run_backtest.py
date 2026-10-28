"""Root launcher for historical and synthetic backtests."""

from backtest.run_backtest import run_full_backtest

if __name__ == "__main__":
    report_path = run_full_backtest()
    print(f"Report generated at: {report_path}")

