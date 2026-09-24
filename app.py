import streamlit as st
import pandas as pd
import plotly.graph_objects as go
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

# Run the backtest (with caching to avoid recomputation on every interaction)
@st.cache_data
def load_backtest_results(mode_param):
    return run_backtest(mode_param)

# Load data
with st.spinner('Loading data and running backtest...'):
    df, trades, summary = load_backtest_results(mode)

# Display current signal and regime (using the last row of the DataFrame)
if not df.empty:
    last_row = df.iloc[-1]
    current_regime = last_row['Regime']
    # Determine current signal: 1 if we are in a position (long), 0 if cash
    # We can infer from the Position column or from the last trade status
    # For simplicity, we'll check if the last row has a position (if we have a Position column)
    # In our backtester, we have a Position column that is 1 when in trade, 0 otherwise.
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

# Chart: Interactive Plotly candlestick chart with regime background
st.subheader("BTC-USD Price Chart with Regime Overlay")

# Create candlestick chart
fig = go.Figure()

# Add candlestick
fig.add_trace(go.Candlestick(
    x=df['Date'],
    open=df['Open'],
    high=df['High'],
    low=df['Low'],
    close=df['Close'],
    name="BTC-USD"
))

# Add background colors based on regime
# We'll add shapes for each regime period
# First, we need to identify contiguous periods of the same regime
if 'Regime' in df.columns:
    df['Regime_Change'] = df['Regime'] != df['Regime'].shift()
    df['Regime_Block'] = df['Regime_Change'].cumsum()

    # Define colors for regimes
    regime_colors = {
        'Bull': 'rgba(0, 255, 0, 0.2)',      # Green with 20% opacity
        'Bear/Crash': 'rgba(255, 0, 0, 0.2)', # Red with 20% opacity
        # Default for other regimes
    }
    default_color = 'rgba(128, 128, 128, 0.1)' # Gray for other regimes

    # Add vertical shading for each regime block
    for block_id in df['Regime_Block'].unique():
        block_df = df[df['Regime_Block'] == block_id]
        if not block_df.empty:
            regime = block_df['Regime'].iloc[0]
            color = regime_colors.get(regime, default_color)
            # Add a rectangle shape for the block
            fig.add_shape(
                type="rect",
                x0=block_df['Date'].iloc[0],
                x1=block_df['Date'].iloc[-1],
                y0=0,
                y1=1,
                yref="paper",
                fillcolor=color,
                line_width=0,
                layer="below"
            )

# Update layout
fig.update_layout(
    title="BTC-USD Hourly Price with Market Regimes",
    xaxis_title="Date",
    yaxis_title="Price (USD)",
    xaxis_rangeslider_visible=False,
    height=600
)

st.plotly_chart(fig, use_container_width=True)

# Metrics section
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

# Display equity curve
st.subheader("Equity Curve")
fig_equity = go.Figure()
fig_equity.add_trace(go.Scatter(
    x=df['Date'],
    y=df['Equity'],
    mode='lines',
    name='Equity',
    line=dict(color='blue')
))
fig_equity.update_layout(
    title="Equity Curve ($10,000 Starting Capital)",
    xaxis_title="Date",
    yaxis_title="Equity (USD)",
    height=400
)
st.plotly_chart(fig_equity, use_container_width=True)

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