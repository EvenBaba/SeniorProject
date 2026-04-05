CONFIG = {
    'train_ratio': 0.9,
    'lookback': 100,
    'time_steps_ae': 7,
    'z_threshold': 2.0,
    'ewma_span': 30,
    'ewma_k': 2.0,
    'lstm_epochs': 100,
    'lstm_batch_size': 16,
    'unsup_k': 0.90,  # Quantile-based (top %10)
    'sup_threshold': 0.3,
    'ae_epochs': 100,
    'ae_batch_size': 32,
    'ae_threshold_quantile': 0.995,
    'ocsvm_nu': 0.005,
    'output_dir': 'results',

    # --- Phase 2A: Dual-Stream LSTM ---
    'dual_epochs': 100,          # max training epochs
    'dual_batch_size': 16,       # batch size
    'dual_threshold': 0.3,       # anomaly classification threshold

    # --- Phase 2B: CryptoBERT Sentiment (GDELT — free, no API key) ---
    'news_sleep_sec': 0.5,
    'cryptobert_batch_size': 32,   # article titles per CryptoBERT inference batch
}