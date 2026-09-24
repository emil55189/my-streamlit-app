import streamlit as st
import pandas as pd
from backtester import run_backtest

# Set page config
st.set_page_config(
    page_title="Regime-Based Trading App",
    page_icon="📈",
    layout="wide"
)

# Title and description
st.title("Regime-Based Trading App for BTC-USD")
st.markdown("""
This app implements a regime-based trading strategy using Hidden Markov Models (HMM) to detect market regimes and a voting system with 8 confirmations for entry signals.
It supports both NORMAL and AGGRESSIVE modes with different risk parameters.
""")

# Sidebar for mode selection
st.sidebar.header("Strategy Settings")
aggressive_mode = st.sidebar.checkbox("Enable AGGRESSIVE Mode", value=False)
mode = 'AGGRESSIVE' if aggressive_mode else 'NORMAL'
st.sidebar.write(f"Current Mode: {mode}")

# Run the backtest
with st.spinner('Loading data and running backtest...'):
    df, trades, summary = run_backtest(mode)

# Display current signal and regime
if not df.empty:
    last_row = df.iloc[-1]
    current_regime = last_row['Regime']
    current_signal = "Long" if last_row.get('Position', 0) == 1 else "Cash"
else:
    current_regime = "Unknown"
    current_signal = "Unknown"

# Top section: Current Signal and Detected Regime
col1, col2 = st.columns(2)
with col1:
    st.metric(label="Current Signal", value=current_signal)
with col2:
    st.metric(label="Detected Regime", value=current_regime)

# Main price chart using Streamlit's built-in line chart
st.subheader("BTC-USD Price Chart")
try:
    # Use Streamlit's built-in line chart for reliability
    price_data = df.set_index('Date')[['Close']].copy()
    st.line_chart(price_data)
except Exception as e:
    st.error(f"Error creating price chart: {e}")

# Performance Metrics
st.subheader("Performance Metrics")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric(label="Total Return", value=f"{summary['total_return_pct']:.2f}%")
with col2:
    st.metric(label="Alpha vs Buy & Hold", value=f"{summary['alpha_pct']:.2f}%")
with col3:
    st.metric(label="Win Rate", value=f"{summary['win_rate']*100:.2f}%")
with col4:
    st.metric(label="Max Drawdown", value=f"{summary['max_drawdown_pct']:.2f}%")

# Equity Curve using Streamlit's built-in line chart
st.subheader("Equity Curve ($10,000 Starting Capital)")
try:
    equity_data = df.set_index('Date')[['Equity']].copy()
    st.line_chart(equity_data)
except Exception as e:
    st.error(f"Error creating equity chart: {e}")

# Display recent trades
st.subheader("Recent Trades")
if trades:
    trades_df = pd.DataFrame(trades)
    # Format the dataframe for display
    trades_df['entry_time'] = pd.to_datetime(trades_df['entry_time'])
    trades_df['exit_time'] = pd.to_datetime(trades_df['exit_time'])
    trades_df['pnl'] = trades_df['pnl'].round(2)
    trades_df['equity_after'] = trades_df['equity_after'].round(2)
    trades_df['duration_hours'] = trades_df['duration_hours'].round(2)

    # Select and rename columns for display
    display_df = trades_df[['entry_time', 'exit_time', 'entry_price', 'exit_price', 'pnl', 'equity_after', 'duration_hours']].copy()
    display_df.columns = ['Entry Time', 'Exit Time', 'Entry Price', 'Exit Price', 'PnL ($)', 'Equity After ($)', 'Duration (hours)']

    # Show the last 10 trades
    st.dataframe(display_df.tail(10), use_container_width=True)
else:
    st.write("No trades executed during the backtest period.")

# Display summary statistics in an expandable section
with st.expander("View Full Backtest Summary"):
    st.json(summary)

# Footer
st.markdown("---")
st.markdown("*Data sourced from yfinance. Strategy backtested on hourly BTC-USD data for the last 730 days.*")