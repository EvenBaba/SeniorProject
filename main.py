"""
MAIN.PY - COMPLETE ANOMALY DETECTION PIPELINE
Supports any Binance trading pair -- change 'symbol' in config.py to switch assets.
 
Phase 2B: Surprise Factor causal analysis (no API, no hardcoded events)
"""
import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
import os
 
from statistic import (
    fetch_data,
    data_preprocessing_and_feature_engineering
)
 
from lstm_unsupervised import (
    unsupervised_lstm_dataset,
    train_unsupervised_lstm,
    test_unsupervised_lstm,
    compute_threshold
)
 
from lstm_supervised import (
    supervised_lstm_dataset,
    train_supervised_lstm,
    test_supervised_lstm
)
 
from lstm_AE import (
    lstm_autoencoder_dataset,
    train_autoencoder_hybrid,
    test_hybrid_model
)
 
from lstm_dual import run_dual_lstm_pipeline
from causal_analysis import run_causal_analysis_pipeline
 
from evaluation import (
    evaluate_model,
    compare_models,
    plot_multiple_roc_curves,
    create_evaluation_summary
)
 
from config import CONFIG
from asset_selector import select_asset
 
os.makedirs(CONFIG['output_dir'], exist_ok=True)
 
 
# ============================================================================
# DATA PREPARATION
# ============================================================================
 
def prepare_data(symbol=None, timeframe=None, limit=None, market_type=None):
    """
    Fetch and preprocess data.
    If symbol/timeframe/limit are not provided, reads from CONFIG.
    Pass them in from select_asset() to override CONFIG at runtime.
    """
    symbol    = symbol    or CONFIG['symbol']
    timeframe = timeframe or CONFIG['timeframe']
    limit     = limit     or CONFIG['limit']
 
    # Update CONFIG so downstream modules see the selected asset
    CONFIG['symbol']      = symbol
    CONFIG['timeframe']   = timeframe
    CONFIG['limit']       = limit
    CONFIG['market_type'] = market_type or 'crypto'

    # Use asset-specific output folder so runs never overwrite each other
    asset_folder = symbol.replace('/', '_').replace('^', '')
    CONFIG['output_dir'] = f"results/{CONFIG['market_type']}_{asset_folder}"
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
 
    print("\n" + "="*80)
    print("DATA PREPARATION")
    print("="*80)
    print(f"  Market   : {CONFIG['market_type'].upper()}")
    print(f"  Asset    : {symbol}")
    print(f"  Timeframe: {timeframe}")
 
    print("\n[1/2] Fetching data...")
    df_raw = fetch_data(
        symbol=symbol,
        timeframe=timeframe,
        limit=limit,
        market_type=market_type
    )
 
    if df_raw is None:
        print("Data fetch failed.")
        exit()
 
    print("\n[2/2] Preprocessing and feature engineering...")
    df, split_idx = data_preprocessing_and_feature_engineering(
        df_raw,
        train_ratio=CONFIG['train_ratio'],
        z_threshold=CONFIG['z_threshold'],
        ewma_span=CONFIG['ewma_span'],
        ewma_k=CONFIG['ewma_k'],
        create_labels=True
    )
 
    print(f"\n Data preparation complete")
    print(f" Total samples: {len(df)}")
    print(f" Train samples: {split_idx}")
    print(f" Test samples:  {len(df) - split_idx}")
    print(f" Features: {len(df.columns)} columns")
 
    return df, split_idx
 
 
# ============================================================================
# BASELINE: STATISTICAL METHODS
# Z-score and EWMA are used ONLY to create anomaly labels in preprocessing.
# They are not evaluated as standalone detectors -- that would be circular
# since the ML models are trained on the same labels.
# ============================================================================
 
def print_label_statistics(df, split_idx):
    """
    Print a summary of the anomaly labels created by Z-score and EWMA.
    Informational only -- not used for model comparison.
    """
    print("\n" + "="*80)
    print("ANOMALY LABEL STATISTICS (Z-score + EWMA -- label creation only)")
    print("="*80)
 
    df_train = df.iloc[:split_idx]
    df_test  = df.iloc[split_idx:]
 
    print(f"\n  Train set: {len(df_train)} days")
    print(f"    Anomaly rate: {df_train['Anomaly_Statistical'].mean():.2%}")
    print(f"  Test set:  {len(df_test)} days")
    print(f"    Anomaly rate: {df_test['Anomaly_Statistical'].mean():.2%}")
    print("\n  Note: Z-score and EWMA thresholds are fit on train set only.")
    print("        These labels are used to train and evaluate ML models.")
    print("="*80)
 
 
# ============================================================================
# MODEL 1: UNSUPERVISED LSTM
# ============================================================================
 
def run_unsupervised_lstm(df):
    print("\n" + "="*80)
    print("MODEL 2: UNSUPERVISED LSTM")
    print("="*80)
 
    print("\n[1/4] Creating dataset...")
    X, y, idx_y, x_scaler, y_scaler = unsupervised_lstm_dataset(
        df, lookback=CONFIG['lookback'], train_ratio=CONFIG['train_ratio']
    )
 
    n = len(X)
    split = int(n * CONFIG['train_ratio'])
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    idx_train, idx_test = idx_y[:split], idx_y[split:]
 
    print("\n[2/4] Training model...")
    lstm_model, history = train_unsupervised_lstm(
        X_train, y_train,
        use_early_stopping=True,
        epochs=CONFIG['lstm_epochs'],
        batch_size=CONFIG['lstm_batch_size']
    )
 
    print("\n[3/4] Computing threshold...")
    threshold = compute_threshold(
        lstm_model, X_train, y_train, y_scaler, k=CONFIG['unsup_k']
    )
 
    print("\n[4/4] Testing...")
    out = test_unsupervised_lstm(
        lstm_model, X_test, y_test, idx_test, y_scaler, threshold
    )
 
    df_test_aligned = df.loc[out.index]
    result = {
        'y_true': df_test_aligned['Anomaly_Statistical'].values,
        'y_pred': out['Anomaly_LSTM'].values,
        'y_prob': out['Forecast_Error'].values
    }
 
    print(f"\nUnsupervised LSTM complete")
    return result, out
 
 
# ============================================================================
# MODEL 2: SUPERVISED LSTM
# ============================================================================
 
def run_supervised_lstm(df):
    print("\n" + "="*80)
    print("MODEL 2: SUPERVISED LSTM")
    print("="*80)
 
    print("\n[1/4] Creating dataset...")
    X, y, idx_y, scaler = supervised_lstm_dataset(
        df, lookback=CONFIG['lookback'], train_ratio=CONFIG['train_ratio']
    )
 
    n = len(X)
    split = int(n * CONFIG['train_ratio'])
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    idx_train, idx_test = idx_y[:split], idx_y[split:]
 
    print("\n[2/4] Training model...")
    lstm_model, history = train_supervised_lstm(
        X_train, y_train,
        use_early_stopping=True,
        epochs=CONFIG['lstm_epochs'],
        batch_size=CONFIG['lstm_batch_size']
    )
 
    print("\n[3/4] Testing...")
    out = test_supervised_lstm(
        lstm_model, X_test, y_test, idx_test, threshold=CONFIG['sup_threshold']
    )
 
    result = {
        'y_true': out['Anomaly_True'].values,
        'y_pred': out['Anomaly_Pred'].values,
        'y_prob': out['Anomaly_Prob'].values
    }
 
    print(f"\nSupervised LSTM complete")
    return result, out
 
 
# ============================================================================
# MODEL 3: DUAL-STREAM LSTM (Phase 2A)
# ============================================================================
 
def run_dual_lstm(df):
    """Dual-Stream LSTM: anomaly classifier + price regressor (Surprise Factor)"""
    print("\n" + "="*80)
    print("MODEL 3: DUAL-STREAM LSTM  (Classifier + Price Regressor)")
    print("="*80)

    # Use finance-specific threshold for low-volatility assets (stocks, forex)
    # Finance assets move 1-8% on anomaly days vs 10-40% for crypto,
    # so a lower threshold is needed to detect anomalies
    if CONFIG.get('market_type') == 'finance':
        CONFIG['dual_threshold'] = CONFIG.get('finance_dual_threshold', 0.1)
        print(f"  Using finance threshold: {CONFIG['dual_threshold']} (lower for low-volatility assets)")
    else:
        CONFIG['dual_threshold'] = CONFIG.get('crypto_dual_threshold', 0.2)
        print(f"  Using crypto threshold : {CONFIG['dual_threshold']}")

    eval_result, results_df = run_dual_lstm_pipeline(df, CONFIG)
    results_df.to_csv(f"{CONFIG['output_dir']}/dual_lstm_results.csv")
    print(f"  Dual LSTM results saved -> {CONFIG['output_dir']}/dual_lstm_results.csv")
    return eval_result, results_df
 
 
# ============================================================================
# MODEL 4: LSTM AUTOENCODER + OCSVM
# ============================================================================
 
def run_autoencoder_hybrid(df):
    print("\n" + "="*80)
    print("MODEL 4: LSTM AUTOENCODER + OCSVM")
    print("="*80)
 
    print("\n[1/3] Creating dataset...")
    X, y_labels, scaler, idx = lstm_autoencoder_dataset(
        df, time_steps=CONFIG['time_steps_ae'], train_ratio=CONFIG['train_ratio']
    )
 
    print("\n[2/3] Training hybrid model...")
    autoencoder, encoder, ocsvm, results = train_autoencoder_hybrid(
        X, y_labels,
        epochs=CONFIG['ae_epochs'],
        batch_size=CONFIG['ae_batch_size'],
        patience=10,
        ocsvm_nu=CONFIG['ocsvm_nu']
    )
 
    print("\n[3/3] Testing hybrid model...")
    n = len(X)
    split = int(n * CONFIG['train_ratio'])
    results_df = test_hybrid_model(
        autoencoder, encoder, ocsvm,
        X[split:], y_labels.iloc[split:], idx[split:],
        ae_threshold_quantile=0.95
    )
 
    result = {
        'y_true': results_df['Anomaly_True'].values,
        'y_pred': results_df['Anomaly_Hybrid'].values,
        'y_prob': results_df['Reconstruction_Error'].values
    }
 
    print(f"\nAutoencoder + OCSVM complete")
    return result, results_df
 
 
# ============================================================================
# EVALUATION AND COMPARISON
# ============================================================================
 
def evaluate_all_models(all_results):
    print("\n" + "="*80)
    print("MODEL EVALUATION AND COMPARISON")
    print("="*80)
 
    print("\n[1/4] Evaluating individual models...")
    for model_name, result in all_results.items():
        print(f"\n{'='*80}")
        print(f"Evaluating: {model_name}")
        print(f"{'='*80}")
        evaluate_model(
            y_true=result['y_true'],
            y_pred=result['y_pred'],
            y_prob=result.get('y_prob'),
            model_name=model_name,
            plot_curves=True,
            save_dir=CONFIG['output_dir']
        )
 
    print("\n[2/4] Comparing all models...")
    comparison_df = compare_models(
        all_results,
        save_path=f"{CONFIG['output_dir']}/model_comparison.png"
    )
    comparison_df.to_csv(f"{CONFIG['output_dir']}/comparison_metrics.csv")
    print(f"Comparison table saved")
 
    print("\n[3/4] Creating combined ROC curves...")
    models_with_prob = {
        name: data for name, data in all_results.items()
        if data.get('y_prob') is not None
    }
    if models_with_prob:
        plot_multiple_roc_curves(
            models_with_prob,
            save_path=f"{CONFIG['output_dir']}/combined_roc_curves.png"
        )
 
    print("\n[4/4] Creating summary report...")
    create_evaluation_summary(
        comparison_df,
        save_path=f"{CONFIG['output_dir']}/evaluation_summary.txt"
    )
 
    print(f"\nEvaluation complete. All results saved to: {CONFIG['output_dir']}/")
    return comparison_df
 
 
# ============================================================================
# MAIN EXECUTION
# ============================================================================
 
def main():
    print("\n" + "="*80)
    print("ANOMALY DETECTION PIPELINE - MAIN EXECUTION")
    print("="*80)
 
    print("\n[CONFIGURATION]")
    for key, value in CONFIG.items():
        print(f"  {key:20s}: {value}")
 
    df, split_idx = prepare_data()
    all_results = {}
 
    print("\n" + "="*80)
    print("RUNNING ALL MODELS")
    print("="*80)
 
    print_label_statistics(df, split_idx)
 
    unsup_result, unsup_out = run_unsupervised_lstm(df)
    all_results['Unsupervised LSTM'] = unsup_result
 
    sup_result, sup_out = run_supervised_lstm(df)
    all_results['Supervised LSTM'] = sup_result
 
    dual_result, dual_out = run_dual_lstm(df)
    all_results['Dual-Stream LSTM'] = dual_result
 
    ae_result, ae_results_df = run_autoencoder_hybrid(df)
    all_results['LSTM Autoencoder + OCSVM'] = ae_result
 
    comparison_df = evaluate_all_models(all_results)
 
    print("\n" + "="*80)
    print("SAVING INDIVIDUAL RESULTS")
    print("="*80)
 
    unsup_out.to_csv(f"{CONFIG['output_dir']}/unsupervised_lstm_results.csv")
    sup_out.to_csv(f"{CONFIG['output_dir']}/supervised_lstm_results.csv")
    ae_results_df.to_csv(f"{CONFIG['output_dir']}/autoencoder_ocsvm_results.csv")
    print("All results saved.")
 
    # Phase 2B: causal analysis on the dual-LSTM results
    print("\n" + "="*80)
    print("PHASE 2B: CAUSAL ANALYSIS")
    print("="*80)
    run_causal_analysis_pipeline(
        dual_results_df=dual_out,
        config=CONFIG,
        sf_threshold=CONFIG.get('sf_z_threshold', 1.0),
        anomaly_col="Anomaly_Pred"
    )
 
    print("\n" + "="*80)
    print("PIPELINE COMPLETE!")
    print("="*80)
 
    if 'F1-Score' in comparison_df.columns:
        best_model = comparison_df['F1-Score'].idxmax()
        best_f1 = comparison_df.loc[best_model, 'F1-Score']
        print(f"\n  Best model: {best_model}  (F1={best_f1:.4f})")
 
    return comparison_df, all_results
 
 
# ============================================================================
# QUICK RUN FUNCTIONS
# ============================================================================
 
# quick_run_statistical() removed -- Z-score and EWMA are label creators,
# not evaluated models. See print_label_statistics() for label summary.
 
 
def quick_run_unsupervised():
    df, split_idx = prepare_data()
    result, out = run_unsupervised_lstm(df)
    evaluate_model(result['y_true'], result['y_pred'], result['y_prob'],
                   model_name='Unsupervised LSTM', plot_curves=True,
                   save_dir=CONFIG['output_dir'])
    return result, out
 
 
def quick_run_supervised():
    df, split_idx = prepare_data()
    result, out = run_supervised_lstm(df)
    evaluate_model(result['y_true'], result['y_pred'], result['y_prob'],
                   model_name='Supervised LSTM', plot_curves=True,
                   save_dir=CONFIG['output_dir'])
    return result, out
 
 
def quick_run_dual(symbol=None, timeframe=None, limit=None, market_type=None):
    """Quick run: Dual-Stream LSTM + immediate causal analysis."""
    df, split_idx = prepare_data(symbol=symbol, timeframe=timeframe, limit=limit, market_type=market_type)
    result, out = run_dual_lstm(df)
    evaluate_model(result['y_true'], result['y_pred'], result['y_prob'],
                   model_name='Dual-Stream LSTM', plot_curves=True,
                   save_dir=CONFIG['output_dir'])
 
    print("\n[SURPRISE FACTOR SUMMARY]")
    print(out[['Close_True', 'Close_Pred', 'Surprise_Factor', 'Surprise_Factor_Z']].describe())
 
    print("\n[PHASE 2B -- CAUSAL ANALYSIS]")
    # Use finance-specific threshold for stocks/forex (lower volatility assets)
    threshold = (CONFIG.get('finance_dual_threshold', 0.1)
                 if CONFIG.get('market_type') == 'finance'
                 else CONFIG.get('dual_threshold', 0.2))
    run_causal_analysis_pipeline(
        dual_results_df=out,
        config=CONFIG,
        sf_threshold=CONFIG.get('sf_z_threshold', 1.0),
        anomaly_col="Anomaly_Pred"
    )
    return result, out


def quick_run_autoencoder():
    df, split_idx = prepare_data()
    result, results_df = run_autoencoder_hybrid(df)
    evaluate_model(result['y_true'], result['y_pred'], result['y_prob'],
                   model_name='Autoencoder + OCSVM', plot_curves=True,
                   save_dir=CONFIG['output_dir'])
    return result, results_df
 
 
def run_causal_analysis():
    """
    Run Phase 2B causal analysis on saved dual-LSTM results.
    Loads dual_lstm_results.csv (or gt version if available).
    Runs in seconds -- no retraining needed.
    """
    print("\n" + "="*80)
    print("PHASE 2B: CAUSAL ANALYSIS (Surprise Factor method)")
    print("="*80)
 
    gt_csv  = f"{CONFIG['output_dir']}/dual_lstm_gt_results.csv"
    std_csv = f"{CONFIG['output_dir']}/dual_lstm_results.csv"
 
    if os.path.exists(gt_csv):
        csv_path = gt_csv
    elif os.path.exists(std_csv):
        csv_path = std_csv
    else:
        print("  [!] No saved dual-LSTM results found.")
        print("  Run quick_run_dual() first.")
        return None
 
    print(f"  Loading: {csv_path}")
    dual_results_df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
 
    return run_causal_analysis_pipeline(
        dual_results_df=dual_results_df,
        config=CONFIG,
        sf_threshold=CONFIG.get('sf_z_threshold', 1.0),
        anomaly_col="Anomaly_Pred"
    )
 
 
# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":

    # Interactive asset selection -- use arrow keys to pick asset/timeframe/history
    symbol, timeframe, limit, market_type = select_asset()

    # Option 1: Full pipeline (all models + causal analysis)
    # comparison_df, all_results = main()
    # This runs everything in sequence - unsupervised LSTM, supervised LSTM, autoencoder+OCSVM,
    # dual-LSTM, causal analysis, and at the end produces a combined comparison table and combined ROC curve
    # with all models on one chart. This is the one to use for your final presentation.


    # Option 2: Quick run specific model (all respect the selected asset)
    # result, out = quick_run_unsupervised()
    # Trains and tests the unsupervised LSTM only. Saves its own ROC curve, PR curve, confusion matrix.

    # result, out = quick_run_supervised()
    # Trains and tests the supervised LSTM only. Same outputs.
    
    # result, results_df = quick_run_autoencoder()
    # Trains and tests the LSTM Autoencoder + OCSVM only. Same outputs.



    # Option 3: Causal analysis only (loads saved results, runs in seconds)
    # causal_results = run_causal_analysis()
    # Causal analysis only - loads the already-saved dual_lstm_results.csv
    # and reruns the Surprise Factor analysis instantly. No retraining.
    # Useful if you just want to tweak the SF_Z threshold without waiting for training.

    # Default: dual-LSTM + causal analysis on selected asset
    result, out = quick_run_dual(symbol=symbol, timeframe=timeframe, limit=limit, market_type=market_type)
    # Runs the dual-stream LSTM + causal analysis. Saves ROC curve, PR curve, confusion matrix. This is Phase 2A + 2B.