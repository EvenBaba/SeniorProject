import ccxt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

limit = 1000
THRESHOLD_Z = 2.0  # Z-Score threshold for anomaly detection

def fetch_cryptocurrency_data(symbol='BTC/USDT', timeframe='1d', limit=1000):
    """
    Fetch OHLCV data from Binance via ccxt.

    Parameters
    ----------
    symbol    : trading pair, e.g. 'BTC/USDT', 'ETH/USDT', 'SOL/USDT'
    timeframe : candle interval, e.g. '1d', '4h', '1h'
    limit     : number of candles to fetch (default 1000)

    Any asset available on Binance can be used.
    Check available symbols at: https://www.binance.com/en/markets
    """
    exchange = ccxt.binance()

    print(f"Fetching {symbol} ({timeframe}, last {limit} candles)...")

    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
    except Exception as e:
        print(f"Fetching error: {e}")
        ohlcv = None

    if ohlcv:
        headers = ['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume']
        df = pd.DataFrame(ohlcv, columns=headers)
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='ms')
        df.set_index('Timestamp', inplace=True)

        print(f"--- Pandas DataFrame for {symbol} ---")
        print(df)
        print("---------------------------------------------------")
    else:
        print("Fetching failed.")

    return df if ohlcv else None


def fetch_financial_data(symbol='AAPL', timeframe='1d', limit=1000):
    """
    Fetch OHLCV data for stocks, forex, indices, and commodities via yfinance.

    Parameters
    ----------
    symbol    : Yahoo Finance ticker, e.g. 'AAPL', 'EURUSD=X', 'GC=F', '^GSPC'
    timeframe : interval string -- '1d', '1wk', '1h'
    limit     : number of candles to fetch (approximate)

    Supported asset classes:
      Stocks      : AAPL, NVDA, TSLA, MSFT, AMZN, META
      ETFs        : SPY, QQQ
      Indices     : ^GSPC (S&P 500), ^IXIC (Nasdaq), ^DJI (Dow Jones)
      Forex       : EURUSD=X, GBPUSD=X, JPY=X, TRY=X
      Commodities : GC=F (Gold), SI=F (Silver), CL=F (Oil), NG=F (Gas)
    """
    try:
        import yfinance as yf
    except ImportError:
        print("[!] yfinance not installed. Run: pip install yfinance")
        return None

    # Map limit (number of candles) to a period yfinance understands
    if timeframe == '1d':
        period = f"{min(limit, 9999)}d"
    elif timeframe == '1wk':
        period = f"{min(limit * 7, 9999)}d"
    elif timeframe == '1h':
        # yfinance only supports hourly data for last 730 days
        period = "730d"
    else:
        period = f"{limit}d"

    print(f"Fetching {symbol} ({timeframe}, ~{limit} candles) via yfinance...")

    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=timeframe, auto_adjust=True)
    except Exception as e:
        print(f"yfinance fetch error: {e}")
        return None

    if df is None or df.empty:
        print(f"[!] No data returned for {symbol}. Check the ticker symbol.")
        return None

    # Keep only OHLCV columns and standardize
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()

    # Remove timezone info from index to match ccxt format
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.index.name = 'Timestamp'

    # Trim to requested limit
    if len(df) > limit:
        df = df.iloc[-limit:]

    print(f"--- Pandas DataFrame for {symbol} ---")
    print(df)
    print("---------------------------------------------------")

    return df


def fetch_data(symbol, timeframe='1d', limit=1000, market_type='crypto'):
    """
    Unified data fetcher. Routes to ccxt or yfinance based on market_type.

    Parameters
    ----------
    symbol      : asset symbol (Binance pair for crypto, Yahoo ticker for finance)
    timeframe   : candle interval
    limit       : number of candles
    market_type : 'crypto' -> ccxt/Binance, 'finance' -> yfinance
    """
    if market_type == 'finance':
        return fetch_financial_data(symbol=symbol, timeframe=timeframe, limit=limit)
    else:
        return fetch_cryptocurrency_data(symbol=symbol, timeframe=timeframe, limit=limit)




def data_preprocessing_and_feature_engineering(
    df, 
    train_ratio=0.9,
    z_threshold=THRESHOLD_Z,
    ewma_span=30,
    ewma_k=2.0,
    create_labels=True # Whether to create anomaly labels - required for supervised learning
):
    """
    UNIFIED VERSION - NO DUPLICATION
    
    Single function that:
    1. Creates all features
    2. Applies train-only statistics (no data leakage)
    3. Creates anomaly labels (optional, for supervised learning)
    
    Parameters:
    -----------
    df : DataFrame
        Input dataframe
    train_ratio : float
        Ratio for train/test split (default: 0.9)
    z_threshold : float
        Z-score threshold for anomaly detection (default: 2.0)
    ewma_span : int
        EWMA span parameter (default: 30)
    ewma_k : float
        EWMA threshold multiplier (default: 2.0)
    create_labels : bool
        If True, creates Anomaly_ZScore, Anomaly_EWMA, Anomaly_Statistical
        Set to False if you only need features (for unsupervised)
        
    Returns:
    --------
    df : DataFrame
        DataFrame with all features and labels (if create_labels=True)
    split_idx : int
        Index where train/test split occurs
    """

    print("\n" + "="*80)
    print("DATA PREPROCESSING AND FEATURE ENGINEERING")
    print("="*80)

    # ========================================================================
    # STEP 1: BASIC FEATURES (No leakage risk)
    # ========================================================================
    print("\n[STEP 1] Creating basic features...")
    
    df['Return'] = df['Close'].pct_change()
    df['Log_Return'] = (df['Close'] / df['Close'].shift(1)).apply(
        lambda x: np.log(x) if x > 0 else np.nan
    )

    # ========================================================================
    # STEP 2: TRAIN/TEST SPLIT FOR GLOBAL STATISTICS
    # ========================================================================
    n = len(df)
    split_idx = int(n * train_ratio)
    
    print(f"\n[STEP 2] Train/Test Split")
    print(f"  Total samples: {n}")
    print(f"  Train samples: {split_idx} ({train_ratio*100:.0f}%)")
    print(f"  Test samples:  {n-split_idx} ({(1-train_ratio)*100:.0f}%)")
    
    df_train = df.iloc[:split_idx]

    # ========================================================================
    # STEP 3: GLOBAL STATISTICS (TRAIN ONLY - NO LEAKAGE!)
    # ========================================================================
    print(f"\n[STEP 3] Computing global statistics (TRAIN ONLY)")
    
    mean_log_return = df_train['Log_Return'].mean()
    std_log_return = df_train['Log_Return'].std()
    
    print(f"  Log Return Mean (train): {mean_log_return:.6f}")
    print(f"  Log Return Std (train):  {std_log_return:.6f}")

    # Apply to entire dataframe (using train statistics)
    df['Price_LogReturn_Z_Score'] = (
        df['Log_Return'] - mean_log_return
    ) / std_log_return

    # ========================================================================
    # STEP 4: ROLLING WINDOW FEATURES (No leakage - only looks back)
    # ========================================================================
    print(f"\n[STEP 4] Creating rolling window features...")
    
    # Rolling Z-Scores
    for window in [7, 14, 30]:
        df[f'Log_Return_MAvg_{window}'] = df['Log_Return'].rolling(window=window).mean()
        df[f'Log_Return_Std_{window}'] = df['Log_Return'].rolling(window=window).std()
        df[f'Price_LogReturn_Z_Score_{window}'] = (
            (df['Log_Return'] - df[f'Log_Return_MAvg_{window}']) / 
            df[f'Log_Return_Std_{window}']
        )

    # Volatility
    for window in [7, 14, 30]:
        df[f'Volatility_{window}'] = df['Log_Return'].rolling(window=window).std()

    # Moving Averages
    for window in [7, 14, 30]:
        df[f'MAvg_{window}'] = df['Close'].rolling(window=window).mean()

    # Volume features
    for window in [7, 14, 30]:
        df[f'Volume_Mean_{window}'] = df['Volume'].rolling(window=window).mean()
        df[f'Volume_Std_{window}'] = df['Volume'].rolling(window=window).std()
        df[f'Volume_Z_Score_{window}'] = (
            (df['Volume'] - df[f'Volume_Mean_{window}']) / 
            df[f'Volume_Std_{window}']
        )

    # Moving Average Deviations
    for window in [7, 14, 30]:
        df[f'Price_MAvg_Deviation_{window}'] = (
            (df['Close'] - df[f'MAvg_{window}']) / df[f'MAvg_{window}']
        ).abs()

    # Standardized Residuals
    for window in [7, 14, 30]:
        df[f'Std_Residual_{window}'] = (
            (df['Close'] - df[f'MAvg_{window}']) / df[f'Volatility_{window}']
        )

    # ========================================================================
    # STEP 5: TECHNICAL INDICATORS
    # ========================================================================
    print(f"\n[STEP 5] Computing technical indicators...")
    
    df = calculate_RSI(df)
    df = calculate_MACD(df)
    df = calculate_BB_Z_Score(df)
    df = calculate_ATR(df)
    df = calculate_candle(df)

    # ========================================================================
    # STEP 6: ANOMALY LABELS (FOR SUPERVISED LEARNING)
    # ========================================================================
    if create_labels:
        print(f"\n[STEP 6] Creating anomaly labels (TRAIN-ONLY thresholds)...")
        
        # --- Z-SCORE ANOMALIES ---
        df['Anomaly_ZScore'] = (
            df['Price_LogReturn_Z_Score'].abs() >= z_threshold
        ).astype(int)
        
        # --- EWMA ANOMALIES ---
        df['EWMA_Close'] = df['Close'].ewm(span=ewma_span, adjust=False).mean()
        df['EWMA_Error'] = (df['Close'] - df['EWMA_Close']).abs()
        
        # Compute threshold from TRAIN data only
        ewma_threshold = (
            df.iloc[:split_idx]['EWMA_Error'].mean() + 
            ewma_k * df.iloc[:split_idx]['EWMA_Error'].std()
        )
        
        df['Anomaly_EWMA'] = (
            df['EWMA_Error'] > ewma_threshold
        ).astype(int)
        
        # --- COMBINED STATISTICAL ANOMALY ---
        df['Anomaly_Statistical'] = (
            (df['Anomaly_ZScore'] == 1) | (df['Anomaly_EWMA'] == 1)
        ).astype(int)
        
        print(f"\n[ANOMALY STATISTICS]")
        print(f"  Z-threshold: {z_threshold}")
        print(f"  EWMA threshold: {ewma_threshold:.4f}")
        print(f"  Z-Score anomalies:   {df['Anomaly_ZScore'].sum()} ({df['Anomaly_ZScore'].mean():.2%})")
        print(f"  EWMA anomalies:      {df['Anomaly_EWMA'].sum()} ({df['Anomaly_EWMA'].mean():.2%})")
        print(f"  Combined anomalies:  {df['Anomaly_Statistical'].sum()} ({df['Anomaly_Statistical'].mean():.2%})")
    else:
        print(f"\n[STEP 6] Skipping label creation (create_labels=False)")

    # ========================================================================
    # STEP 7: CLEANUP
    # ========================================================================
    df.dropna(inplace=True)
    
    print(f"\n[STEP 7] Cleanup complete")
    print(f"  Samples after dropna: {len(df)}")
    print("="*80 + "\n")

    return df, split_idx


def calculate_RSI(df, window_rsi=14):
    """Relative Strength Index"""
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window_rsi).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window_rsi).mean()
    RS = gain / loss
    df['RSI'] = 100 - (100 / (1 + RS))
    return df


def calculate_MACD(df):
    """Moving Average Convergence Divergence"""
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    return df


def calculate_BB_Z_Score(df, rolling_window=30):
    """Bollinger Bands and Z-Score"""
    df['BB_MA'] = df['Close'].rolling(window=rolling_window).mean()
    df['BB_Std'] = df['Close'].rolling(window=rolling_window).std()
    df['BB_Upper'] = df['BB_MA'] + (df['BB_Std'] * 2)
    df['BB_Lower'] = df['BB_MA'] - (df['BB_Std'] * 2)
    df['BB_Z_Score'] = (df['Close'] - df['BB_MA']) / df['BB_Std']
    return df


def calculate_ATR(df, rolling_window=14):
    """Average True Range"""
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    df['ATR'] = true_range.ewm(span=rolling_window, adjust=False).mean()
    return df


def calculate_candle(df):
    """Candlestick Features"""
    df['Candle_Body'] = np.abs(df['Close'] - df['Open'])
    df['Candle_Range'] = df['High'] - df['Low']
    df['Upper_Shadow'] = df['High'] - df[['Open', 'Close']].max(axis=1)
    df['Lower_Shadow'] = df[['Open', 'Close']].min(axis=1) - df['Low']
    
    EPSILON = 1e-6 
    df['Upper_Shadow_Ratio'] = df['Upper_Shadow'] / (df['Candle_Body'] + EPSILON)
    df['Lower_Shadow_Ratio'] = df['Lower_Shadow'] / (df['Candle_Body'] + EPSILON)
    df['Candle_Body_Direction'] = np.where(df['Close'] > df['Open'], 1, -1)
    
    return df


# ============================================================================
# VISUALIZATION FUNCTIONS (Optional - for exploration)
# ============================================================================

def plot_zscore_anomaly(df, tail=200):
    """Plot Z-Score anomalies"""
    if 'Anomaly_ZScore' not in df.columns:
        print("Error: Anomaly_ZScore column not found. Use create_labels=True")
        return
    
    df_plot = df.tail(tail).copy()
    anomalies = df_plot[df_plot['Anomaly_ZScore'] == 1]

    plt.figure(figsize=(15, 6))
    plt.plot(df_plot.index, df_plot['Close'], 
             label='Closing Prices', color='blue', linewidth=1.5)
    plt.scatter(anomalies.index, anomalies['Close'], 
                color='red', marker='o', s=50,
                label=f'Z-Score Anomaly (Total: {len(anomalies)})')
    
    plt.title('Anomaly Detection Using Z-Score Method')
    plt.xlabel('Date')
    plt.ylabel('Closing Price (USDT)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()


def plot_ewma_anomaly(df, tail=200):
    """Plot EWMA anomalies"""
    if 'Anomaly_EWMA' not in df.columns:
        print("Error: Anomaly_EWMA column not found. Use create_labels=True")
        return
    
    df_plot = df.tail(tail).copy()
    anomalies = df_plot[df_plot['Anomaly_EWMA'] == 1]

    plt.figure(figsize=(15, 6))
    plt.plot(df_plot.index, df_plot['Close'], 
             label='Closing Prices', color='blue', linewidth=1.5)
    plt.plot(df_plot.index, df_plot['EWMA_Close'], 
             label='EWMA Close', color='orange', linewidth=1.5)
    plt.scatter(anomalies.index, anomalies['Close'], 
                color='red', marker='o', s=50,
                label=f'EWMA Anomaly (Total: {len(anomalies)})')
    
    plt.title('Anomaly Detection Using EWMA Method')
    plt.xlabel('Date')
    plt.ylabel('Closing Price (USDT)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    """
    Example usage showing different scenarios
    """
    
    print("\n" + "="*80)
    print("EXAMPLE USAGE")
    print("="*80)
    
    # Fetch data
    df_raw = fetch_cryptocurrency_data()
    
    if df_raw is not None:
        
        # ========================================================================
        # SCENARIO 1: For SUPERVISED learning (need labels)
        # ========================================================================
        print("\n[SCENARIO 1] Preparing data for SUPERVISED learning...")
        df_supervised, split_idx = data_preprocessing_and_feature_engineering(
            df_raw.copy(),
            train_ratio=0.9,
            z_threshold=2.0,
            ewma_span=30,
            ewma_k=2.0,
            create_labels=True  # Create anomaly labels
        )
        print(f"   Ready for supervised LSTM training")
        print(f"   Features: {len(df_supervised.columns)} columns")
        print(f"   Target: Anomaly_Statistical column")
        
        # ========================================================================
        # SCENARIO 2: For UNSUPERVISED learning (no labels needed)
        # ========================================================================
        print("\n[SCENARIO 2] Preparing data for UNSUPERVISED learning...")
        df_unsupervised, split_idx = data_preprocessing_and_feature_engineering(
            df_raw.copy(),
            train_ratio=0.9,
            create_labels=False  # Skip label creation
        )
        print(f"   Ready for unsupervised LSTM training")
        print(f"   Features: {len(df_unsupervised.columns)} columns")
        print(f"   No labels (unsupervised)")
        
        # ========================================================================
        # OPTIONAL: Visualize anomalies (if labels were created)
        # ========================================================================
        if 'Anomaly_ZScore' in df_supervised.columns:
            print("\n[VISUALIZATION] Plotting anomalies...")
            # plot_zscore_anomaly(df_supervised, tail=200)
            # plot_ewma_anomaly(df_supervised, tail=200)
            print("  (Uncomment to show plots)")
        
        print("\n" + "="*80)
        print("PREPROCESSING COMPLETE - NO DATA LEAKAGE!")
        print("="*80 + "\n")