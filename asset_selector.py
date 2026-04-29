"""
ASSET SELECTOR
==============
Interactive terminal menu for selecting the market type and asset
before running the anomaly detection pipeline.

Flow
----
  Step 1: Crypto or Finance?
  Step 2: Select asset from the chosen market
  Step 3: Select timeframe
  Step 4: Select history length

Returns (symbol, timeframe, limit, market_type)
  market_type: "crypto" or "finance"
  -- passed to statistic.py to decide which fetcher to use

Install dependency: pip install inquirer
"""

import sys

# ============================================================================
# CRYPTO ASSETS  (fetched via ccxt / Binance)
# ============================================================================

CRYPTO_ASSETS = [
    # Majors
    {"name": "Bitcoin          (BTC/USDT)", "symbol": "BTC/USDT"},
    {"name": "Ethereum         (ETH/USDT)", "symbol": "ETH/USDT"},
    {"name": "Solana           (SOL/USDT)", "symbol": "SOL/USDT"},
    {"name": "BNB              (BNB/USDT)", "symbol": "BNB/USDT"},
    {"name": "XRP              (XRP/USDT)", "symbol": "XRP/USDT"},
    # Mid-cap
    {"name": "Avalanche        (AVAX/USDT)", "symbol": "AVAX/USDT"},
    {"name": "Cardano          (ADA/USDT)",  "symbol": "ADA/USDT"},
    {"name": "Dogecoin         (DOGE/USDT)", "symbol": "DOGE/USDT"},
    {"name": "Polkadot         (DOT/USDT)",  "symbol": "DOT/USDT"},
    {"name": "Chainlink        (LINK/USDT)", "symbol": "LINK/USDT"},
    # Cross pairs
    {"name": "ETH/BTC          (ETH priced in BTC)", "symbol": "ETH/BTC"},
    {"name": "SOL/BTC          (SOL priced in BTC)", "symbol": "SOL/BTC"},
    # Custom
    {"name": "Custom symbol... (type your own Binance pair)", "symbol": "CUSTOM_CRYPTO"},
]

# ============================================================================
# FINANCIAL ASSETS  (fetched via yfinance)
# ============================================================================

FINANCE_ASSETS = [
    # --- US Stocks ---
    {"name": "Apple            (AAPL)   -- US Stock",    "symbol": "AAPL",    "category": "Stock"},
    {"name": "NVIDIA           (NVDA)   -- US Stock",    "symbol": "NVDA",    "category": "Stock"},
    {"name": "Tesla            (TSLA)   -- US Stock",    "symbol": "TSLA",    "category": "Stock"},
    {"name": "Microsoft        (MSFT)   -- US Stock",    "symbol": "MSFT",    "category": "Stock"},
    {"name": "Amazon           (AMZN)   -- US Stock",    "symbol": "AMZN",    "category": "Stock"},
    {"name": "Meta             (META)   -- US Stock",    "symbol": "META",    "category": "Stock"},
    {"name": "S&P 500 ETF      (SPY)    -- US Index",    "symbol": "SPY",     "category": "Index"},
    {"name": "Nasdaq 100 ETF   (QQQ)    -- US Index",    "symbol": "QQQ",     "category": "Index"},
    # --- Indices ---
    {"name": "S&P 500          (^GSPC)  -- Index",       "symbol": "^GSPC",   "category": "Index"},
    {"name": "Nasdaq           (^IXIC)  -- Index",       "symbol": "^IXIC",   "category": "Index"},
    {"name": "Dow Jones        (^DJI)   -- Index",       "symbol": "^DJI",    "category": "Index"},
    # --- Forex ---
    {"name": "EUR/USD          (EURUSD) -- Forex",       "symbol": "EURUSD=X", "category": "Forex"},
    {"name": "GBP/USD          (GBPUSD) -- Forex",       "symbol": "GBPUSD=X", "category": "Forex"},
    {"name": "USD/JPY          (USDJPY) -- Forex",       "symbol": "JPY=X",    "category": "Forex"},
    {"name": "USD/TRY          (USDTRY) -- Forex",       "symbol": "TRY=X",    "category": "Forex"},
    # --- Commodities ---
    {"name": "Gold             (GC=F)   -- Commodity",   "symbol": "GC=F",    "category": "Commodity"},
    {"name": "Silver           (SI=F)   -- Commodity",   "symbol": "SI=F",    "category": "Commodity"},
    {"name": "Crude Oil        (CL=F)   -- Commodity",   "symbol": "CL=F",    "category": "Commodity"},
    {"name": "Natural Gas      (NG=F)   -- Commodity",   "symbol": "NG=F",    "category": "Commodity"},
    # --- Custom ---
    {"name": "Custom symbol... (type your own Yahoo Finance ticker)", "symbol": "CUSTOM_FINANCE", "category": "Custom"},
]

# ============================================================================
# TIMEFRAMES
# Note: yfinance uses different interval strings than ccxt
# ============================================================================

CRYPTO_TIMEFRAMES = [
    {"name": "1 day    (1d) -- recommended, most data",  "value": "1d"},
    {"name": "4 hours  (4h) -- more granular",           "value": "4h"},
    {"name": "1 hour   (1h) -- high frequency",          "value": "1h"},
]

FINANCE_TIMEFRAMES = [
    {"name": "1 day    (1d) -- recommended, most data",  "value": "1d"},
    {"name": "1 week   (1wk) -- long term trends",       "value": "1wk"},
    {"name": "1 hour   (1h) -- intraday (last 730 days only)", "value": "1h"},
]

LIMITS = [
    {"name": "1000 candles -- maximum history",  "value": 1000},
    {"name": "500  candles -- last ~1.5 years",  "value": 500},
    {"name": "250  candles -- last ~8 months",   "value": 250},
]

MARKET_TYPES = [
    {"name": "Cryptocurrency  -- Bitcoin, Ethereum, Solana, BNB...", "value": "crypto"},
    {"name": "Finance         -- Stocks, Forex, Indices, Commodities...", "value": "finance"},
]


# ============================================================================
# PUBLIC ENTRY POINT
# ============================================================================

def select_asset() -> tuple:
    """
    Interactive asset selection menu.

    Returns
    -------
    (symbol, timeframe, limit, market_type) : str, str, int, str
      market_type is "crypto" or "finance"
    """
    print("\n" + "="*60)
    print("  ANOMALY DETECTION PIPELINE -- ASSET SELECTION")
    print("="*60)

    try:
        import inquirer
        return _inquirer_menu()
    except ImportError:
        print("\n  [!] 'inquirer' not installed.")
        print("      Install with: pip install inquirer")
        print("      Falling back to numbered menu...\n")
        return _text_menu()
    except Exception:
        print("\n  [!] Interactive menu unavailable in this terminal.")
        print("      Falling back to numbered menu...\n")
        return _text_menu()


# ============================================================================
# INQUIRER MENU (arrow-key)
# ============================================================================

def _inquirer_menu() -> tuple:
    import inquirer

    # Step 1: Market type
    q_market = [
        inquirer.List(
            "market",
            message="Select market type",
            choices=[m["name"] for m in MARKET_TYPES],
            carousel=True,
        )
    ]
    ans_market = inquirer.prompt(q_market)
    if ans_market is None:
        sys.exit(0)
    market_type = next(m["value"] for m in MARKET_TYPES if m["name"] == ans_market["market"])

    # Step 2: Asset
    assets = CRYPTO_ASSETS if market_type == "crypto" else FINANCE_ASSETS
    timeframes = CRYPTO_TIMEFRAMES if market_type == "crypto" else FINANCE_TIMEFRAMES

    q_asset = [
        inquirer.List(
            "asset",
            message=f"Select {'cryptocurrency' if market_type == 'crypto' else 'financial asset'}",
            choices=[a["name"] for a in assets],
            carousel=True,
        )
    ]
    ans_asset = inquirer.prompt(q_asset)
    if ans_asset is None:
        sys.exit(0)

    selected = next(a for a in assets if a["name"] == ans_asset["asset"])
    symbol = selected["symbol"]

    # Custom symbol entry
    if symbol == "CUSTOM_CRYPTO":
        q_custom = [inquirer.Text("sym", message="Enter Binance pair (e.g. PEPE/USDT)")]
        ans = inquirer.prompt(q_custom)
        if ans is None:
            sys.exit(0)
        symbol = ans["sym"].strip().upper()
        if "/" not in symbol:
            symbol += "/USDT"

    elif symbol == "CUSTOM_FINANCE":
        q_custom = [inquirer.Text("sym", message="Enter Yahoo Finance ticker (e.g. GOOGL, BABA, BTC-USD)")]
        ans = inquirer.prompt(q_custom)
        if ans is None:
            sys.exit(0)
        symbol = ans["sym"].strip().upper()

    # Step 3: Timeframe
    q_tf = [
        inquirer.List(
            "timeframe",
            message="Select timeframe",
            choices=[t["name"] for t in timeframes],
            carousel=True,
        )
    ]
    ans_tf = inquirer.prompt(q_tf)
    if ans_tf is None:
        sys.exit(0)
    timeframe = next(t["value"] for t in timeframes if t["name"] == ans_tf["timeframe"])

    # Step 4: History length
    q_lim = [
        inquirer.List(
            "limit",
            message="Select history length",
            choices=[l["name"] for l in LIMITS],
            carousel=True,
        )
    ]
    ans_lim = inquirer.prompt(q_lim)
    if ans_lim is None:
        sys.exit(0)
    limit = next(l["value"] for l in LIMITS if l["name"] == ans_lim["limit"])

    _print_selection(symbol, timeframe, limit, market_type)
    return symbol, timeframe, limit, market_type


# ============================================================================
# TEXT MENU (numbered fallback)
# ============================================================================

def _pick(options, prompt="Select"):
    print(f"\n  {prompt}:")
    for i, o in enumerate(options, 1):
        name = o["name"] if isinstance(o, dict) else o
        print(f"    {i:>2}. {name}")
    while True:
        try:
            choice = int(input("\n  Enter number: ").strip())
            if 1 <= choice <= len(options):
                return choice - 1
        except ValueError:
            pass
        print(f"  Please enter a number between 1 and {len(options)}")


def _text_menu() -> tuple:
    # Step 1: Market type
    idx = _pick(MARKET_TYPES, "Select market type")
    market_type = MARKET_TYPES[idx]["value"]

    assets    = CRYPTO_ASSETS    if market_type == "crypto" else FINANCE_ASSETS
    timeframes = CRYPTO_TIMEFRAMES if market_type == "crypto" else FINANCE_TIMEFRAMES
    label = "cryptocurrency" if market_type == "crypto" else "financial asset"

    # Step 2: Asset
    idx = _pick(assets, f"Select {label}")
    symbol = assets[idx]["symbol"]

    if symbol == "CUSTOM_CRYPTO":
        symbol = input("  Enter Binance pair (e.g. PEPE/USDT): ").strip().upper()
        if "/" not in symbol:
            symbol += "/USDT"
    elif symbol == "CUSTOM_FINANCE":
        symbol = input("  Enter Yahoo Finance ticker (e.g. GOOGL, BABA): ").strip().upper()

    # Step 3: Timeframe
    idx = _pick(timeframes, "Select timeframe")
    timeframe = timeframes[idx]["value"]

    # Step 4: History length
    idx = _pick(LIMITS, "Select history length")
    limit = LIMITS[idx]["value"]

    _print_selection(symbol, timeframe, limit, market_type)
    return symbol, timeframe, limit, market_type


# ============================================================================
# PRINT CONFIRMATION
# ============================================================================

def _print_selection(symbol, timeframe, limit, market_type):
    print("\n" + "="*60)
    print(f"  Market    : {market_type.upper()}")
    print(f"  Asset     : {symbol}")
    print(f"  Timeframe : {timeframe}")
    print(f"  History   : {limit} candles")
    print("="*60 + "\n")


# ============================================================================
# STANDALONE TEST
# ============================================================================

if __name__ == "__main__":
    symbol, timeframe, limit, market_type = select_asset()
    print(f"Selected: {symbol} / {timeframe} / {limit} / {market_type}")