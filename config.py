CONFIG = {
    # ----------------------------------------------------------------
    # ASSET CONFIGURATION
    # Change symbol to run the full pipeline on any Binance asset.
    # Examples:
    #   Crypto  : 'BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT'
    #   Stables : 'ETH/BTC' (ETH priced in BTC)
    # timeframe: '1d' (daily), '4h' (4-hour), '1h' (hourly)
    # limit    : number of candles to fetch (max ~1000 on Binance free)
    # ----------------------------------------------------------------
    'symbol':    'BTC/USDT',
    'timeframe': '1d',
    'limit':     1000,

    # ----------------------------------------------------------------
    # MODEL HYPERPARAMETERS
    # ----------------------------------------------------------------
    'train_ratio': 0.9,
    'lookback': 100,
    'time_steps_ae': 7,
    'z_threshold': 2.0,
    'ewma_span': 30,
    'ewma_k': 2.0,
    'lstm_epochs': 100,
    'lstm_batch_size': 16,
    'unsup_k': 0.90,
    'sup_threshold': 0.2,
    'ae_epochs': 100,
    'ae_batch_size': 32,
    'ae_threshold_quantile': 0.995,
    'ocsvm_nu': 0.005,
    'output_dir': 'results',

    # ----------------------------------------------------------------
    # PHASE 2A: DUAL-STREAM LSTM
    # ----------------------------------------------------------------
    'dual_epochs': 100,
    'dual_batch_size': 16,

    # Anomaly classification threshold -- selected automatically based on market type.
    # Crypto assets (BTC, ETH) are highly volatile so 0.2 works well.
    # Finance assets (stocks, forex) move much less so need a lower threshold.
    'crypto_dual_threshold':  0.2,   # used when market_type = 'crypto'
    'finance_dual_threshold': 0.15,   # used when market_type = 'finance'
    'dual_threshold':         0.2,   # default fallback (overwritten at runtime)

    # ----------------------------------------------------------------
    # PHASE 2B: CAUSAL ANALYSIS (Surprise Factor method)
    # sf_z_threshold: SF Z-score cutoff for external_shock verdict.
    #   Raise to be stricter (fewer externals), lower for more.
    # ----------------------------------------------------------------
    'sf_z_threshold': 1.0,
}