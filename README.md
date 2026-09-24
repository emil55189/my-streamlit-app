# Regime-Based Trading App for BTC-USD

This application implements a regime-based trading strategy using Hidden Markov Models (HMM) to detect market regimes and a voting system with 8 confirmations for entry signals.

## Features

- **Dual Mode Strategy**:
  - **NORMAL Mode**: 2.5x leverage, requires 7/8 confirmations to enter
  - **AGGRESSIVE Mode**: 4x leverage, requires 5/8 confirmations to enter, includes 5% trailing stop loss
- **Regime Detection**: Uses Gaussian HMM with 7 components on returns, range, and volume volatility features
- **Risk Management**:
  - 48-hour cooldown after any exit
  - Exit immediately if regime flips to 'Bear' or 'Crash'
  - Trailing stop loss in AGGRESSIVE mode (5%)
- **Interactive Dashboard**:
  - Current signal (Long/Cash) and detected regime display
  - Price chart with regime overview (using Streamlit's built-in charts for reliability)
  - Performance metrics (Total Return, Alpha vs Buy & Hold, Win Rate, Max Drawdown)
  - Equity curve visualization
  - Recent trades table
  - Detailed backtest summary in expandable section

## Files

- `data_loader.py`: Loads BTC-USD hourly data using yfinance
- `backtester.py`: Core engine with HMM logic, confirmation system, and backtesting (includes both NORMAL and AGGRESSIVE modes)
- `app.py`: Streamlit dashboard interface
- `requirements.txt`: Python dependencies

## Installation

1. Clone this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the application:
   ```bash
   streamlit run app.py
   ```
4. Use the sidebar checkbox to toggle between NORMAL and AGGRESSIVE modes

## Dependencies

- yfinance
- pandas
- numpy
- hmmlearn
- ta (technical analysis library)
- streamlit
- scikit-learn (for StandardScaler)

## Notes

- The app loads the last 730 days of hourly BTC-USD data
- Charts use Streamlit's built-in charting capabilities for maximum compatibility and reliability
- To switch back to Plotly charts (if desired in the future), modify the charting sections in `app.py`

## Trading Logic

**Entry Conditions** (both modes):
- HMM Regime must be 'Bull'
- Required confirmations met (7/8 for NORMAL, 5/8 for AGGRESSIVE)
- Not in cooldown period (48 hours after last exit)

**Exit Conditions** (both modes):
- Regime flips to 'Bear' or 'Crash'
- (AGGRESSIVE only) 5% trailing stop loss from peak price since entry

**Position Sizing**:
- Simulated leverage applied to PnL calculations (2.5x NORMAL, 4x AGGRESSIVE)
- $10,000 starting capital

## Disclaimer

This is for educational purposes only. Past performance does not guarantee future results. Trading cryptocurrencies involves significant risk of loss.