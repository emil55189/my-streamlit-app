import yfinance as yf
import pandas as pd

def load_btc_data(period="730d", interval="1h"):
    """
    Load BTC-USD hourly data for the specified period.

    Args:
        period (str): Period for data download (default: "730d" for 730 days)
        interval (str): Data interval (default: "1h" for hourly)

    Returns:
        pd.DataFrame: DataFrame with OHLCV data
    """
    ticker = "BTC-USD"
    data = yf.download(ticker, period=period, interval=interval)

    # Check if data is empty
    if data.empty:
        raise ValueError(f"No data downloaded for {ticker} with period={period}, interval={interval}")

    # Reset index to have Date as a column
    data.reset_index(inplace=True)

    # Rename 'Datetime' to 'Date' if present (yfinance sometimes uses Datetime)
    if 'Datetime' in data.columns:
        data.rename(columns={'Datetime': 'Date'}, inplace=True)

    # Flatten column index if it's MultiIndex (yfinance sometimes returns multi-level columns)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    # Ensure we have the required columns
    required_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    if not all(col in data.columns for col in required_columns):
        raise ValueError(f"Missing required columns. Available columns: {data.columns.tolist()}")

    return data

if __name__ == "__main__":
    # Test the data loader
    df = load_btc_data()
    print(f"Loaded {len(df)} rows of BTC-USD data")
    print(df.head())
    print(df.tail())