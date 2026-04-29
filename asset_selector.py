"""
ASSET SELECTOR
==============
Interactive terminal menu for selecting the trading asset and timeframe
before running the anomaly detection pipeline.

Uses the 'inquirer' library for arrow-key selection menus.
Install: pip install inquirer

Usage
-----
    from asset_selector import select_asset
    symbol, timeframe, limit = select_asset()
    # Then pass symbol/timeframe/limit to CONFIG or directly to pipeline
"""

import sys

# ============================================================================
# ASSET CATALOGUE
# Extend this list to add more assets.
# Any symbol available on Binance spot market works.
# Check https://www.binance.com/en/markets for full list.
# ============================================================================

ASSETS = [
    # --- Crypto majors ---
    {"name": "Bitcoin          (BTC/USDT)", "symbol": "BTC/USDT",  "category": "Major"},
    {"name": "Ethereum         (ETH/USDT)", "symbol": "ETH/USDT",  "category": "Major"},
    {"name": "Solana           (SOL/USDT)", "symbol": "SOL/USDT",  "category": "Major"},
    {"name": "BNB              (BNB/USDT)", "symbol": "BNB/USDT",  "category": "Major"},
    {"name": "XRP              (XRP/USDT)", "symbol": "XRP/USDT",  "category": "Major"},
    # --- Crypto mid-cap ---
    {"name": "Avalanche        (AVAX/USDT)","symbol": "AVAX/USDT", "category": "Mid-cap"},
    {"name": "Cardano          (ADA/USDT)", "symbol": "ADA/USDT",  "category": "Mid-cap"},
    {"name": "Dogecoin         (DOGE/USDT)","symbol": "DOGE/USDT", "category": "Mid-cap"},
    {"name": "Polkadot         (DOT/USDT)", "symbol": "DOT/USDT",  "category": "Mid-cap"},
    {"name": "Chainlink        (LINK/USDT)","symbol": "LINK/USDT", "category": "Mid-cap"},
    # --- Cross pairs (crypto vs crypto) ---
    {"name": "ETH/BTC          (ETH priced in BTC)", "symbol": "ETH/BTC", "category": "Cross"},
    {"name": "SOL/BTC          (SOL priced in BTC)", "symbol": "SOL/BTC", "category": "Cross"},
    # --- Custom entry ---
    {"name": "Custom symbol... (type your own)", "symbol": "CUSTOM", "category": "Custom"},
]

TIMEFRAMES = [
    {"name": "1 day    (1d) -- recommended, most data",  "value": "1d"},
    {"name": "4 hours  (4h) -- more granular",           "value": "4h"},
    {"name": "1 hour   (1h) -- high frequency",          "value": "1h"},
]

LIMITS = [
    {"name": "1000 candles -- maximum history",  "value": 1000},
    {"name": "500  candles -- last ~1.5 years",  "value": 500},
    {"name": "250  candles -- last ~8 months",   "value": 250},
]


# ============================================================================
# SELECTOR
# ============================================================================

def select_asset() -> tuple:
    """
    Show interactive arrow-key menus and return the selected
    (symbol, timeframe, limit) tuple.

    Falls back to a simple numbered text menu if inquirer
    is not installed or the terminal does not support it.

    Returns
    -------
    (symbol, timeframe, limit) : str, str, int
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
        # inquirer can fail on some Windows terminals
        print("\n  [!] Interactive menu unavailable in this terminal.")
        print("      Falling back to numbered menu...\n")
        return _text_menu()


def _inquirer_menu() -> tuple:
    """Arrow-key selection using inquirer."""
    import inquirer

    # Step 1: Asset
    asset_choices = [a["name"] for a in ASSETS]
    q1 = [
        inquirer.List(
            "asset",
            message="Select asset to analyse",
            choices=asset_choices,
            carousel=True,
        )
    ]
    ans1 = inquirer.prompt(q1)
    if ans1 is None:
        print("Selection cancelled.")
        sys.exit(0)

    selected_asset = next(a for a in ASSETS if a["name"] == ans1["asset"])
    symbol = selected_asset["symbol"]

    # Handle custom symbol entry
    if symbol == "CUSTOM":
        q_custom = [
            inquirer.Text(
                "custom_symbol",
                message="Enter Binance symbol (e.g. PEPE/USDT, MATIC/USDT)",
            )
        ]
        ans_custom = inquirer.prompt(q_custom)
        if ans_custom is None:
            sys.exit(0)
        symbol = ans_custom["custom_symbol"].strip().upper()
        if "/" not in symbol:
            symbol = symbol + "/USDT"

    # Step 2: Timeframe
    tf_choices = [t["name"] for t in TIMEFRAMES]
    q2 = [
        inquirer.List(
            "timeframe",
            message="Select timeframe",
            choices=tf_choices,
            carousel=True,
        )
    ]
    ans2 = inquirer.prompt(q2)
    if ans2 is None:
        sys.exit(0)
    timeframe = next(t["value"] for t in TIMEFRAMES if t["name"] == ans2["timeframe"])

    # Step 3: History length
    lim_choices = [l["name"] for l in LIMITS]
    q3 = [
        inquirer.List(
            "limit",
            message="Select history length",
            choices=lim_choices,
            carousel=True,
        )
    ]
    ans3 = inquirer.prompt(q3)
    if ans3 is None:
        sys.exit(0)
    limit = next(l["value"] for l in LIMITS if l["name"] == ans3["limit"])

    _print_selection(symbol, timeframe, limit)
    return symbol, timeframe, limit


def _text_menu() -> tuple:
    """Simple numbered fallback menu for terminals without inquirer support."""

    # Asset
    print("  Select asset:")
    for i, a in enumerate(ASSETS, 1):
        print(f"    {i:>2}. {a['name']}")
    while True:
        try:
            choice = int(input("\n  Enter number: ").strip())
            if 1 <= choice <= len(ASSETS):
                break
            print(f"  Please enter a number between 1 and {len(ASSETS)}")
        except ValueError:
            print("  Invalid input -- please enter a number")

    selected_asset = ASSETS[choice - 1]
    symbol = selected_asset["symbol"]

    if symbol == "CUSTOM":
        symbol = input("  Enter Binance symbol (e.g. PEPE/USDT): ").strip().upper()
        if "/" not in symbol:
            symbol = symbol + "/USDT"

    # Timeframe
    print("\n  Select timeframe:")
    for i, t in enumerate(TIMEFRAMES, 1):
        print(f"    {i}. {t['name']}")
    while True:
        try:
            choice = int(input("\n  Enter number: ").strip())
            if 1 <= choice <= len(TIMEFRAMES):
                break
        except ValueError:
            pass
        print(f"  Please enter 1-{len(TIMEFRAMES)}")
    timeframe = TIMEFRAMES[choice - 1]["value"]

    # Limit
    print("\n  Select history length:")
    for i, l in enumerate(LIMITS, 1):
        print(f"    {i}. {l['name']}")
    while True:
        try:
            choice = int(input("\n  Enter number: ").strip())
            if 1 <= choice <= len(LIMITS):
                break
        except ValueError:
            pass
        print(f"  Please enter 1-{len(LIMITS)}")
    limit = LIMITS[choice - 1]["value"]

    _print_selection(symbol, timeframe, limit)
    return symbol, timeframe, limit


def _print_selection(symbol, timeframe, limit):
    print("\n" + "="*60)
    print(f"  Asset     : {symbol}")
    print(f"  Timeframe : {timeframe}")
    print(f"  History   : {limit} candles")
    print("="*60 + "\n")


# ============================================================================
# STANDALONE TEST
# ============================================================================

if __name__ == "__main__":
    symbol, timeframe, limit = select_asset()
    print(f"Selected: {symbol} / {timeframe} / {limit} candles")
