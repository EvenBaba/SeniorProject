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
    'output_dir': 'results'
}