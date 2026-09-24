import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from data_loader import load_btc_data
import ta  # Technical Analysis library
from sklearn.preprocessing import StandardScaler

def calculate_features(df):
    """
    Calculate the three features for HMM: Returns, Range, and Volume Volatility.

    Args:
        df (pd.DataFrame): DataFrame with OHLCV data

    Returns:
        pd.DataFrame: DataFrame with additional feature columns
    """
    # Make a copy to avoid modifying the original
    df = df.copy()

    # 1. Returns: log returns of closing price
    df['Returns'] = np.log(df['Close'] / df['Close'].shift(1))

    # 2. Range: (High - Low) / Close
    df['Range'] = (df['High'] - df['Low']) / df['Close']

    # 3. Volume Volatility: rolling standard deviation of volume over 20 periods
    df['Volume_Volatility'] = df['Volume'].rolling(window=20).std()

    # Replace infinite values with NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Drop rows with NaN values (due to shifting, rolling, and infinite values)
    df.dropna(inplace=True)

    return df

def train_hmm(df, n_components=7):
    """
    Train a Gaussian HMM on the three features.

    Args:
        df (pd.DataFrame): DataFrame with features: Returns, Range, Volume_Volatility
        n_components (int): Number of hidden states (regimes)

    Returns:
        GaussianHMM: Trained HMM model
        np.ndarray: Array of hidden states for each sample
        StandardScaler: The scaler used to standardize the features
    """
    # Select features for HMM
    features = df[['Returns', 'Range', 'Volume_Volatility']].values

    # Standardize the features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Create and train the HMM model
    model = GaussianHMM(n_components=n_components, covariance_type="full", n_iter=1000, random_state=42)
    model.fit(features_scaled)

    # Predict the hidden states
    hidden_states = model.predict(features_scaled)

    return model, hidden_states, scaler

def identify_regimes(df, hidden_states):
    """
    Identify the Bull Run and Bear/Crash regimes based on the average returns of each state.

    Args:
        df (pd.DataFrame): DataFrame with the original data (must have 'Returns' column)
        hidden_states (np.ndarray): Array of hidden states for each sample

    Returns:
        dict: Mapping from regime index to regime name ('Bull', 'Bear', 'Neutral', etc.)
        int: Index of the Bull Run regime
        int: Index of the Bear/Crash regime
    """
    # Compute average return for each state
    # We'll create a DataFrame with the hidden states and returns
    state_returns = pd.DataFrame({'state': hidden_states, 'return': df['Returns'].values})
    # Group by state and compute mean return
    avg_returns = state_returns.groupby('state')['return'].mean()

    # Identify the Bull Run state (highest positive return)
    bull_regime = avg_returns.idxmax()

    # Identify the Bear/Crash state (lowest return)
    bear_regime = avg_returns.idxmin()

    # Create a mapping of regime index to a descriptive name
    regime_names = {}
    for state in avg_returns.index:
        if state == bull_regime:
            regime_names[state] = 'Bull'
        elif state == bear_regime:
            regime_names[state] = 'Bear/Crash'
        else:
            # For other states, we can label based on return magnitude
            if avg_returns[state] > 0:
                regime_names[state] = 'Mild_Bull'
            else:
                regime_names[state] = 'Mild_Bear'

    return regime_names, bull_regime, bear_regime

def calculate_confirmations(df):
    """
    Calculate the 8 confirmation conditions.

    Args:
        df (pd.DataFrame): DataFrame with OHLCV data

    Returns:
        pd.DataFrame: DataFrame with boolean columns for each confirmation
    """
    df = df.copy()

    # 1. RSI < 90
    df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
    df['Conf_1'] = df['RSI'] < 90

    # 2. Momentum > 1% (price change over 1 period)
    df['Momentum'] = df['Close'].pct_change(periods=1)
    df['Conf_2'] = df['Momentum'] > 0.01

    # 3. Volatility < 6% (using rolling standard deviation of returns)
    df['Volatility'] = df['Returns'].rolling(window=20).std()
    df['Conf_3'] = df['Volatility'] < 0.06

    # 4. Volume > 20-period SMA of volume
    df['Volume_SMA_20'] = df['Volume'].rolling(window=20).mean()
    df['Conf_4'] = df['Volume'] > df['Volume_SMA_20']

    # 5. ADX > 25
    df['ADX'] = ta.trend.adx(df['High'], df['Low'], df['Close'], window=14)
    df['Conf_5'] = df['ADX'] > 25

    # 6. Price > EMA 50
    df['EMA_50'] = ta.trend.ema_indicator(df['Close'], window=50)
    df['Conf_6'] = df['Close'] > df['EMA_50']

    # 7. Price > EMA 200
    df['EMA_200'] = ta.trend.ema_indicator(df['Close'], window=200)
    df['Conf_7'] = df['Close'] > df['EMA_200']

    # 8. MACD > Signal Line
    macd = ta.trend.MACD(df['Close'])
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    df['Conf_8'] = df['MACD'] > df['MACD_Signal']

    # List of confirmation columns
    confirm_cols = [f'Conf_{i}' for i in range(1, 9)]

    # Count how many confirmations are met
    df['Confirmations_Count'] = df[confirm_cols].sum(axis=1)

    # Condition: at least 7 out of 8 confirmations
    df['Confirmations_Met'] = df['Confirmations_Count'] >= 7

    return df

def backtest_strategy(df, initial_capital=10000, leverage=2.5, cooldown_hours=48, confirmation_threshold=7, trailing_stop_pct=0.0):
    """
    Backtest the regime-based trading strategy.

    Args:
        df (pd.DataFrame): DataFrame with data, regimes, and confirmations
        initial_capital (float): Starting capital in USD
        leverage (float): Leverage multiplier for PnL
        cooldown_hours (int): Cooldown period in hours after exit
        confirmation_threshold (int): Minimum number of confirmations required (out of 8)
        trailing_stop_pct (float): Trailing stop percentage (0.0 to disable, e.g., 0.05 for 5%)

    Returns:
        pd.DataFrame: DataFrame with backtest results (including equity curve)
        list: List of trade dictionaries
    """
    # Make a copy to avoid modifying the original
    df = df.copy()

    # Initialize columns for backtesting
    df['Position'] = 0  # 1 for long, 0 for cash
    df['Entry_Price'] = np.nan
    df['Exit_Price'] = np.nan
    df['PnL'] = 0.0
    df['Equity'] = float(initial_capital)

    # Track state
    current_position = 0
    entry_price = 0
    equity = initial_capital
    last_exit_time = None  # Timestamp of last exit for cooldown
    highest_since_entry = 0  # Track highest price for trailing stop

    # List to store trades
    trades = []

    # Iterate over the DataFrame
    for i in range(len(df)):
        row = df.iloc[i]
        timestamp = row['Date']
        price = row['Close']
        regime = row['Regime']  # We'll add this column later
        confirmations_met = row['Confirmations_Met']
        confirmations_count = row['Confirmations_Count']  # For aggressive mode logic

        # Check cooldown: if we are in cooldown, we cannot enter
        in_cooldown = False
        if last_exit_time is not None:
            cooldown_expired = (timestamp - last_exit_time).total_seconds() / 3600 >= cooldown_hours
            in_cooldown = not cooldown_expired

        # If we are currently in a position
        if current_position == 1:
            # Update highest price since entry for trailing stop
            if price > highest_since_entry:
                highest_since_entry = price

            # Check exit conditions
            exit_triggered = False
            exit_price = price

            # 1. Regime flip to Bear/Crash (original condition)
            if regime == 'Bear/Crash':
                exit_triggered = True

            # 2. Trailing stop (if enabled)
            elif trailing_stop_pct > 0 and highest_since_entry > 0:
                trailing_stop_price = highest_since_entry * (1 - trailing_stop_pct)
                if price <= trailing_stop_price:
                    exit_triggered = True
                    exit_price = price  # Exit at current price

            if exit_triggered:
                # Exit the position
                # Calculate PnL for the trade (long position)
                pnl = (exit_price - entry_price) / entry_price * leverage * equity
                equity += pnl  # Update equity

                # Record trade
                trade = {
                    'entry_time': entry_time,
                    'exit_time': timestamp,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'pnl': pnl,
                    'equity_after': equity,
                    'duration_hours': (timestamp - entry_time).total_seconds() / 3600
                }
                trades.append(trade)

                # Update state
                current_position = 0
                entry_price = 0
                highest_since_entry = 0  # Reset trailing stop tracker
                last_exit_time = timestamp  # Start cooldown from this exit

                # Mark exit in DataFrame
                df.at[df.index[i], 'Exit_Price'] = exit_price
                df.at[df.index[i], 'PnL'] = pnl
                df.at[df.index[i], 'Equity'] = equity

        # If we are not in a position, check for entry
        else:
            # Entry condition: regime is Bull AND confirmations met AND not in cooldown
            # For aggressive mode, we use confirmation_threshold instead of hardcoded 7
            if regime == 'Bull' and confirmations_met and not in_cooldown:
                # For backwards compatibility with confirmations_met (which is >=7),
                # we need to check if we meet the specific threshold
                # If confirmation_threshold <= 7, the existing confirmations_met is sufficient
                # If confirmation_threshold > 7, we need to check confirmations_count directly
                meets_confirmation_req = False
                if confirmation_threshold <= 7:
                    meets_confirmation_req = confirmations_met  # Already checks >=7
                else:
                    meets_confirmation_req = confirmations_count >= confirmation_threshold

                if meets_confirmation_req:
                    # Enter long position
                    entry_price = price
                    entry_time = timestamp
                    current_position = 1
                    highest_since_entry = price  # Initialize trailing stop tracker

                    # Mark entry in DataFrame
                    df.at[df.index[i], 'Position'] = 1
                    df.at[df.index[i], 'Entry_Price'] = entry_price

        # Update equity curve: if in position, equity changes with price
        if current_position == 1:
            # Current equity if we were to close at current price (unrealized)
            current_equity = equity + (price - entry_price) / entry_price * leverage * equity
            df.at[df.index[i], 'Equity'] = current_equity
        else:
            df.at[df.index[i], 'Equity'] = equity

    # If we are still in a position at the end, close it at the last price
    if current_position == 1:
        exit_price = df.iloc[-1]['Close']
        pnl = (exit_price - entry_price) / entry_price * leverage * equity
        equity += pnl

        trade = {
            'entry_time': entry_time,
            'exit_time': df.iloc[-1]['Date'],
            'entry_price': entry_price,
            'exit_price': exit_price,
            'pnl': pnl,
            'equity_after': equity,
            'duration_hours': (df.iloc[-1]['Date'] - entry_time).total_seconds() / 3600
        }
        trades.append(trade)

        df.at[df.index[-1], 'Exit_Price'] = exit_price
        df.at[df.index[-1], 'PnL'] = pnl
        df.at[df.index[-1], 'Equity'] = equity

    return df, trades

def run_backtest(mode='NORMAL'):
    """
    Main function to run the entire backtest.

    Args:
        mode (str): 'NORMAL' or 'AGGRESSIVE' - determines strategy parameters

    Returns:
        pd.DataFrame: DataFrame with backtest results
        list: List of trade dictionaries
        dict: Summary statistics
    """
    # Set parameters based on mode
    if mode == 'AGGRESSIVE':
        leverage = 4.0
        confirmation_threshold = 5  # 5 out of 8 confirmations
        trailing_stop_pct = 0.05   # 5% trailing stop
    else:  # NORMAL mode
        leverage = 2.5
        confirmation_threshold = 7  # 7 out of 8 confirmations (original)
        trailing_stop_pct = 0.0     # No trailing stop

    # Step 1: Load data
    print("Loading BTC-USD data...")
    df = load_btc_data()

    # Step 2: Calculate features for HMM
    print("Calculating features...")
    df = calculate_features(df)

    # Step 3: Train HMM
    print("Training HMM with 7 components...")
    model, hidden_states, scaler = train_hmm(df, n_components=7)

    # Step 4: Identify regimes
    print("Identifying regimes...")
    regime_names, bull_regime, bear_regime = identify_regimes(df, hidden_states)
    df['Regime_Code'] = hidden_states
    df['Regime'] = df['Regime_Code'].map(regime_names)

    # Step 5: Calculate confirmations
    print("Calculating confirmations...")
    df = calculate_confirmations(df)

    # Step 6: Run backtest
    print("Running backtest...")
    df, trades = backtest_strategy(
        df,
        leverage=leverage,
        confirmation_threshold=confirmation_threshold,
        trailing_stop_pct=trailing_stop_pct
    )

    # Step 7: Calculate summary statistics
    total_trades = len(trades)
    winning_trades = sum(1 for t in trades if t['pnl'] > 0)
    win_rate = winning_trades / total_trades if total_trades > 0 else 0
    total_return = (df['Equity'].iloc[-1] - 10000) / 10000 * 100  # Percentage

    # Buy and hold return for comparison
    buy_hold_return = (df['Close'].iloc[-1] - df['Close'].iloc[0]) / df['Close'].iloc[0] * 100
    alpha = total_return - buy_hold_return

    # Max drawdown
    df['Peak'] = df['Equity'].cummax()
    df['Drawdown'] = (df['Equity'] - df['Peak']) / df['Peak'] * 100
    max_drawdown = df['Drawdown'].min()

    summary = {
        'total_trades': total_trades,
        'winning_trades': winning_trades,
        'win_rate': win_rate,
        'total_return_pct': total_return,
        'buy_hold_return_pct': buy_hold_return,
        'alpha_pct': alpha,
        'max_drawdown_pct': max_drawdown,
        'final_equity': df['Equity'].iloc[-1]
    }

    return df, trades, summary

if __name__ == "__main__":
    # Run the backtest when script is executed
    df, trades, summary = run_backtest()

    print("\n=== Backtest Summary ===")
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"{key}: {value:.2f}")
        else:
            print(f"{key}: {value}")

    print(f"\nNumber of trades: {len(trades)}")
    if trades:
        print("First trade:", trades[0])
        print("Last trade:", trades[-1])