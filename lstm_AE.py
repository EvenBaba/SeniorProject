"""
LSTM Autoencoder + One-Class SVM Hybrid Model
Fixed version without data leakage
"""

from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.layers import Input, Dropout, Dense, LSTM, TimeDistributed, RepeatVector # type: ignore
from tensorflow.keras import regularizers # type: ignore
from tensorflow.keras.callbacks import EarlyStopping # type: ignore
from sklearn.svm import OneClassSVM
from sklearn.metrics import classification_report, roc_auc_score, f1_score
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from config import CONFIG

# --- Global Parameters ---
TIME_STEPS = 7      # Time steps for LSTM input
LATENT_DIM = 32     # Latent vector dimension


def lstm_autoencoder_dataset(df, time_steps=TIME_STEPS, train_ratio=0.9):
    """
    FIXED VERSION - NO DATA LEAKAGE
    
    Prepare dataset for LSTM Autoencoder:
    1. Select features
    2. Split train/test BEFORE scaling
    3. Fit scaler ONLY on train data
    4. Transform both train and test
    
    Parameters:
    -----------
    df : DataFrame
        Input dataframe with features and labels
    time_steps : int
        Number of time steps for LSTM sequences (default: 7)
    train_ratio : float
        Train/test split ratio (default: 0.9)
        
    Returns:
    --------
    X : array
        3D tensor (samples, time_steps, features)
    y_labels : DataFrame
        Labels (Anomaly_ZScore, Anomaly_EWMA)
    scaler : StandardScaler
        Fitted scaler (on train data only)
    idx : list
        Datetime indices
    """
    
    print("\n" + "="*80)
    print("LSTM AUTOENCODER DATASET PREPARATION")
    print("="*80)
    
    # ========================================================================
    # STEP 1: SELECT FEATURES
    # ========================================================================
    print("\n[STEP 1] Selecting features...")
    
    feature_columns = [
        # Z-Scores
        'Price_LogReturn_Z_Score', 'Price_LogReturn_Z_Score_7', 
        'Price_LogReturn_Z_Score_14', 'Price_LogReturn_Z_Score_30', 
        'Volume_Z_Score_7', 'Volume_Z_Score_14', 'Volume_Z_Score_30', 
        'BB_Z_Score', 
        
        # Volatility and Returns
        'Log_Return', 'Volatility_7', 'Volatility_14', 'Volatility_30',
        
        # Deviations and Residuals
        'Price_MAvg_Deviation_7', 'Price_MAvg_Deviation_14', 'Price_MAvg_Deviation_30',
        'Std_Residual_7', 'Std_Residual_14', 'Std_Residual_30',
        
        # Technical Indicators
        'RSI', 'MACD_Hist', 'ATR', 'Upper_Shadow_Ratio', 
        'Lower_Shadow_Ratio', 'Candle_Body_Direction'
    ]
    
    df_ml = df[feature_columns].dropna()
    data_ml = df_ml.values
    
    print(f"  Selected features: {len(feature_columns)}")
    print(f"  Samples after dropna: {len(df_ml)}")
    
    # ========================================================================
    # STEP 2: TRAIN/TEST SPLIT (BEFORE SCALING!)
    # ========================================================================
    print("\n[STEP 2] Train/Test split (BEFORE scaling)...")
    
    n_samples = len(data_ml)
    split_idx = int(n_samples * train_ratio)
    
    data_train = data_ml[:split_idx]
    data_test = data_ml[split_idx:]
    
    print(f"  Total samples: {n_samples}")
    print(f"  Train samples: {len(data_train)} ({train_ratio*100:.0f}%)")
    print(f"  Test samples:  {len(data_test)} ({(1-train_ratio)*100:.0f}%)")
    
    # ========================================================================
    # STEP 3: SCALING (FIT ONLY ON TRAIN!)
    # ========================================================================
    print("\n[STEP 3] Scaling features (TRAIN ONLY fit)...")
    
    scaler = StandardScaler()
    
    # FIT ONLY ON TRAIN DATA
    scaler.fit(data_train)
    
    # TRANSFORM BOTH
    scaled_data = scaler.transform(data_ml)
    
    print(f"  Scaler fitted on: {len(data_train)} train samples")
    print(f"  Feature mean (train): {data_train.mean():.4f}")
    print(f"  Feature std (train):  {data_train.std():.4f}")
    
    # ========================================================================
    # STEP 4: CREATE 3D SEQUENCES
    # ========================================================================
    print("\n[STEP 4] Creating 3D sequences...")
    
    X = []
    idx = []
    
    for i in range(time_steps - 1, len(scaled_data)):
        X.append(scaled_data[i - time_steps + 1:i + 1])
        idx.append(df_ml.index[i])
    
    X = np.array(X)
    
    print(f"  Input shape: {X.shape}")
    print(f"  (samples, time_steps, features)")
    
    # ========================================================================
    # STEP 5: PREPARE LABELS (ALIGNED WITH X)
    # ========================================================================
    print("\n[STEP 5] Preparing labels...")
    
    # Labels start from time_steps-1 to align with X
    start_idx = time_steps - 1
    y_labels = df[['Anomaly_ZScore', 'Anomaly_EWMA']].loc[df_ml.index].iloc[start_idx:].copy()
    
    # Verify alignment
    if X.shape[0] != y_labels.shape[0]:
        raise ValueError(f"ERROR: X ({X.shape[0]}) and Y ({y_labels.shape[0]}) size mismatch!")
    
    print(f"  Label shape: {y_labels.shape}")
    print(f"  Anomaly rate (ZScore): {y_labels['Anomaly_ZScore'].mean():.2%}")
    print(f"  Anomaly rate (EWMA):   {y_labels['Anomaly_EWMA'].mean():.2%}")
    
    print("\n" + "="*80)
    print("DATASET PREPARATION COMPLETE - NO DATA LEAKAGE")
    print("="*80 + "\n")
    
    return X, y_labels, scaler, idx


def build_lstm_autoencoder(input_shape):
    """
    Build LSTM Autoencoder architecture
    
    Parameters:
    -----------
    input_shape : tuple
        (time_steps, n_features)
        
    Returns:
    --------
    autoencoder : Model
        Full autoencoder model
    encoder : Model
        Encoder part only (for extracting latent vectors)
    """
    
    n_features = input_shape[1]
    
    print("\n[BUILDING LSTM AUTOENCODER]")
    print(f"  Input shape: {input_shape}")
    print(f"  Latent dimension: {LATENT_DIM}")
    
    # ========================================================================
    # ENCODER
    # ========================================================================
    inputs = Input(shape=input_shape, name='encoder_input')
    
    # Layer 1: LSTM (64 units)
    x = LSTM(64, activation='relu', return_sequences=True, 
             kernel_regularizer=regularizers.l2(0.00),
             name='encoder_lstm_1')(inputs)
    
    # Layer 2: LSTM (32 units) - Latent space
    latent = LSTM(LATENT_DIM, activation='relu', return_sequences=False,
                  name='latent_space')(x)
    
    # Encoder model (for extracting latent vectors)
    encoder = Model(inputs=inputs, outputs=latent, name='encoder')
    
    # ========================================================================
    # DECODER
    # ========================================================================
    # Repeat latent vector for all time steps
    x = RepeatVector(input_shape[0], name='repeat_vector')(latent)
    
    # Layer 3: LSTM (32 units)
    x = LSTM(LATENT_DIM, activation='relu', return_sequences=True,
             name='decoder_lstm_1')(x)
    
    # Layer 4: LSTM (64 units)
    x = LSTM(64, activation='relu', return_sequences=True,
             name='decoder_lstm_2')(x)
    
    # Layer 5: Output - reconstruct original features
    outputs = TimeDistributed(Dense(n_features, activation='linear'),
                             name='reconstruction')(x)
    
    # ========================================================================
    # FULL AUTOENCODER
    # ========================================================================
    autoencoder = Model(inputs=inputs, outputs=outputs, name='autoencoder')
    autoencoder.compile(optimizer='adam', loss='mae')
    
    print("\n [MODEL ARCHITECTURE]")
    print(" Encoder: Input -> LSTM(64) -> LSTM(32) -> Latent")
    print(" Decoder: Latent -> Repeat -> LSTM(32) -> LSTM(64) -> Output")
    print(f"  Total parameters: {autoencoder.count_params():,}")
    
    return autoencoder, encoder


def train_autoencoder_hybrid(X, y_labels, 
                            epochs=100, 
                            batch_size=32,
                            patience=10,
                            ocsvm_nu=0.05):
    """
    Train LSTM Autoencoder + One-Class SVM hybrid model
    
    IMPORTANT: Autoencoder is trained ONLY on normal data (unsupervised)
    
    Parameters:
    -----------
    X : array
        3D input tensor (samples, time_steps, features)
    y_labels : DataFrame
        Labels with Anomaly_ZScore column
    epochs : int
        Max training epochs (default: 100)
    batch_size : int
        Batch size (default: 32)
    patience : int
        Early stopping patience (default: 10)
    ocsvm_nu : float
        OneClassSVM nu parameter (expected anomaly fraction, default: 0.05)
        
    Returns:
    --------
    autoencoder : Model
        Trained autoencoder
    encoder : Model
        Trained encoder
    ocsvm : OneClassSVM
        Trained One-Class SVM
    results : dict
        Training results and metrics
    """
    
    print("\n" + "="*80)
    print("LSTM AUTOENCODER + ONE-CLASS SVM TRAINING")
    print("="*80)
    
    # ========================================================================
    # STEP 1: TRAIN/TEST SPLIT
    # ========================================================================
    print("\n[STEP 1] Splitting data...")
    
    n = len(X)
    split = int(n * 0.9)
    
    X_train = X[:split]
    X_test = X[split:]
    
    y_train_labels = y_labels.iloc[:split]
    y_test_labels = y_labels.iloc[split:]
    
    print(f"  Train samples: {len(X_train)}")
    print(f"  Test samples:  {len(X_test)}")
    
    # ========================================================================
    # STEP 2: FILTER ONLY NORMAL DATA FOR TRAINING
    # ========================================================================
    print("\n[STEP 2] Filtering normal data for autoencoder training...")
    
    # Train autoencoder ONLY on normal samples
    normal_mask = y_train_labels['Anomaly_ZScore'] == 0
    X_train_normal = X_train[normal_mask.values]
    
    print(f"  Total train samples: {len(X_train)}")
    print(f"  Normal samples:      {len(X_train_normal)} ({len(X_train_normal)/len(X_train):.1%})")
    print(f"  Anomaly samples:     {len(X_train) - len(X_train_normal)}")
    print(f"  Autoencoder will train ONLY on normal data")
    
    # ========================================================================
    # STEP 3: BUILD AND TRAIN AUTOENCODER
    # ========================================================================
    print("\n[STEP 3] Building and training autoencoder...")
    
    input_shape = (X.shape[1], X.shape[2])
    autoencoder, encoder = build_lstm_autoencoder(input_shape)
    
    # Early stopping
    early_stop = EarlyStopping(
        monitor='loss',
        patience=patience,
        restore_best_weights=True,
        verbose=1
    )
    
    print("\n[TRAINING AUTOENCODER ON NORMAL DATA ONLY]")
    history = autoencoder.fit(
        X_train_normal,
        X_train_normal,  # Autoencoder reconstructs input
        epochs=epochs,
        batch_size=batch_size,
        shuffle=False,   # Keep time series order
        callbacks=[early_stop],
        verbose=1
    )
    
    final_loss = history.history['loss'][-1]
    print(f"\n Autoencoder training complete")
    print(f" Final loss: {final_loss:.6f}")
    
    # ========================================================================
    # STEP 4: EXTRACT LATENT VECTORS
    # ========================================================================
    print("\n[STEP 4] Extracting latent vectors...")
    
    # Extract latent representations
    X_train_latent = encoder.predict(X_train_normal, verbose=0)
    X_test_latent = encoder.predict(X_test, verbose=0)
    
    print(f"  Train latent shape: {X_train_latent.shape}")
    print(f"  Test latent shape:  {X_test_latent.shape}")
    
    # ========================================================================
    # STEP 5: TRAIN ONE-CLASS SVM
    # ========================================================================
    print("\n[STEP 5] Training One-Class SVM on latent vectors...")
    
    ocsvm = OneClassSVM(
        kernel='rbf',
        gamma='auto',
        nu=ocsvm_nu  # Expected anomaly fraction
    )
    
    ocsvm.fit(X_train_latent)
    
    print(f" One-Class SVM training complete")
    print(f" Nu parameter: {ocsvm_nu} (expected anomaly rate)")
    print(f" Kernel: RBF")
    
    # ========================================================================
    # STEP 6: EVALUATE ON TEST SET
    # ========================================================================
    print("\n[STEP 6] Evaluating on test set...")
    
    # Get OCSVM predictions
    decision_scores = ocsvm.decision_function(X_test_latent)
    y_pred_ocsvm = (decision_scores < 0).astype(int)  # 1 = anomaly
    
    # Get true labels
    y_test = y_test_labels['Anomaly_ZScore'].values
    
    # Calculate metrics
    f1 = f1_score(y_test, y_pred_ocsvm, zero_division=0)
    
    try:
        roc_auc = roc_auc_score(y_test, -decision_scores)  # Negative for anomaly direction
    except:
        roc_auc = None
        print("  Warning: Could not calculate ROC-AUC")
    
    print(f"\n[TEST SET METRICS]")
    print(f"  F1-Score:  {f1:.4f}")
    if roc_auc:
        print(f"  ROC-AUC:   {roc_auc:.4f}")
    print(f"  Predicted anomalies: {y_pred_ocsvm.sum()}/{len(y_pred_ocsvm)} ({y_pred_ocsvm.mean():.1%})")
    print(f"  True anomalies:      {y_test.sum()}/{len(y_test)} ({y_test.mean():.1%})")
    
    # ========================================================================
    # RESULTS SUMMARY
    # ========================================================================
    results = {
        'autoencoder': autoencoder,
        'encoder': encoder,
        'ocsvm': ocsvm,
        'history': history,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'decision_scores': decision_scores,
        'y_pred': y_pred_ocsvm,
        'y_test': y_test
    }
    
    print("\n" + "="*80)
    print("HYBRID MODEL TRAINING COMPLETE")
    print("="*80 + "\n")
    
    return autoencoder, encoder, ocsvm, results


def calculate_reconstruction_error(autoencoder, X, idx, 
                                   threshold_quantile=0.95):
    """
    Calculate reconstruction error for anomaly detection
    
    Parameters:
    -----------
    autoencoder : Model
        Trained autoencoder
    X : array
        3D input tensor
    idx : list
        Datetime indices
    threshold_quantile : float
        Quantile for threshold (default: 0.98)
        
    Returns:
    --------
    error_df : DataFrame
        DataFrame with reconstruction errors and anomaly flags
    """
    
    print("\n" + "="*80)
    print("CALCULATING RECONSTRUCTION ERROR")
    print("="*80)
    
    # ========================================================================
    # STEP 1: RECONSTRUCT INPUT
    # ========================================================================
    print("\n[STEP 1] Reconstructing input...")
    
    X_pred = autoencoder.predict(X, verbose=0)
    
    print(f"  Original shape: {X.shape}")
    print(f"  Reconstructed shape: {X_pred.shape}")
    
    # ========================================================================
    # STEP 2: CALCULATE ERROR PER SAMPLE
    # ========================================================================
    print("\n[STEP 2] Calculating reconstruction error...")
    
    # Mean absolute error across time steps and features
    reconstruction_error = np.mean(np.abs(X_pred - X), axis=(1, 2))
    
    print(f"  Min error:  {reconstruction_error.min():.6f}")
    print(f"  Max error:  {reconstruction_error.max():.6f}")
    print(f"  Mean error: {reconstruction_error.mean():.6f}")
    print(f"  Std error:  {reconstruction_error.std():.6f}")
    
    # ========================================================================
    # STEP 3: DETERMINE THRESHOLD
    # ========================================================================
    print("\n[STEP 3] Determining anomaly threshold...")
    
    threshold = np.quantile(reconstruction_error, threshold_quantile)
    
    print(f"  Threshold (Q{int(threshold_quantile*100)}): {threshold:.6f}")
    
    # ========================================================================
    # STEP 4: CREATE RESULTS DATAFRAME
    # ========================================================================
    print("\n[STEP 4] Creating results dataframe...")
    
    error_df = pd.DataFrame(index=idx)
    error_df['Reconstruction_Error'] = reconstruction_error
    error_df['Threshold'] = threshold
    error_df['Anomaly_AE'] = (reconstruction_error > threshold).astype(int)
    
    n_anomalies = error_df['Anomaly_AE'].sum()
    
    print(f"  Total samples: {len(error_df)}")
    print(f"  Anomalies detected: {n_anomalies} ({n_anomalies/len(error_df):.2%})")
    
    print("\n" + "="*80)
    print("RECONSTRUCTION ERROR CALCULATION COMPLETE")
    print("="*80 + "\n")
    
    return error_df


def test_hybrid_model(autoencoder, encoder, ocsvm, 
                     X_test, y_test_labels, idx_test,
                     ae_threshold_quantile=CONFIG['ae_threshold_quantile']):
    """
    Test the hybrid model and return predictions
    
    Parameters:
    -----------
    autoencoder : Model
        Trained autoencoder
    encoder : Model
        Trained encoder
    ocsvm : OneClassSVM
        Trained One-Class SVM
    X_test : array
        Test data
    y_test_labels : DataFrame
        True labels
    idx_test : list
        Datetime indices
    ae_threshold_quantile : float
        Threshold for reconstruction error
        
    Returns:
    --------
    results_df : DataFrame
        Complete results with predictions from both methods
    """
    
    print("\n" + "="*80)
    print("TESTING HYBRID MODEL")
    print("="*80)
    
    # ========================================================================
    # METHOD 1: RECONSTRUCTION ERROR
    # ========================================================================
    print("\n[METHOD 1] Autoencoder Reconstruction Error...")
    
    error_df = calculate_reconstruction_error(
        autoencoder, X_test, idx_test,
        threshold_quantile=ae_threshold_quantile
    )
    
    # ========================================================================
    # METHOD 2: ONE-CLASS SVM ON LATENT SPACE
    # ========================================================================
    print("\n[METHOD 2] One-Class SVM on Latent Space...")
    
    X_test_latent = encoder.predict(X_test, verbose=0)
    decision_scores = ocsvm.decision_function(X_test_latent)
    y_pred_ocsvm = (decision_scores < 0).astype(int)
    
    print(f"  OCSVM anomalies: {y_pred_ocsvm.sum()} ({y_pred_ocsvm.mean():.2%})")
    
    # ========================================================================
    # COMBINE RESULTS
    # ========================================================================
    print("\n[COMBINING RESULTS]...")
    
    results_df = error_df.copy()
    results_df['OCSVM_Score'] = decision_scores
    results_df['Anomaly_OCSVM'] = y_pred_ocsvm
    results_df['Anomaly_Hybrid'] = (
        (results_df['Anomaly_AE'] == 1) | 
        (results_df['Anomaly_OCSVM'] == 1)
    ).astype(int)
    
    # Add true labels
    results_df['Anomaly_True'] = y_test_labels['Anomaly_ZScore'].values
    
    print(f"  AE anomalies:     {results_df['Anomaly_AE'].sum()}")
    print(f"  OCSVM anomalies:  {results_df['Anomaly_OCSVM'].sum()}")
    print(f"  Hybrid anomalies: {results_df['Anomaly_Hybrid'].sum()}")
    print(f"  True anomalies:   {results_df['Anomaly_True'].sum()}")
    
    print("\n" + "="*80)
    print("TESTING COMPLETE")
    print("="*80 + "\n")
    
    return results_df


# ============================================================================
# MAIN EXECUTION EXAMPLE
# ============================================================================

if __name__ == "__main__":
    """
    Example usage
    """
    
    print("\n" + "="*80)
    print("LSTM AUTOENCODER + OCSVM - EXAMPLE USAGE")
    print("="*80)
    
    # This is just a placeholder - in real usage, import your data
    # from statistic_unified import fetch_cryptocurrency_data, data_preprocessing_and_feature_engineering
    
    # # 1. Prepare data
    # df_raw = fetch_cryptocurrency_data()
    # df, split_idx = data_preprocessing_and_feature_engineering(
    #     df_raw,
    #     train_ratio=0.9,
    #     create_labels=True
    # )
    
    # # 2. Create autoencoder dataset
    # X, y_labels, scaler, idx = lstm_autoencoder_dataset(
    #     df,
    #     time_steps=7,
    #     train_ratio=0.9
    # )
    
    # # 3. Train hybrid model
    # autoencoder, encoder, ocsvm, results = train_autoencoder_hybrid(
    #     X, y_labels,
    #     epochs=100,
    #     batch_size=32,
    #     patience=10,
    #     ocsvm_nu=0.05
    # )
    
    # # 4. Test on test set
    # n = len(X)
    # split = int(n * 0.9)
    # X_test = X[split:]
    # y_test_labels = y_labels.iloc[split:]
    # idx_test = idx[split:]
    
    # results_df = test_hybrid_model(
    #     autoencoder, encoder, ocsvm,
    #     X_test, y_test_labels, idx_test,
    #     ae_threshold_quantile=0.95
    # )
    
    # print(results_df.head())
    
    print("\n To use this module:")
    print(" 1. Import your preprocessed data")
    print(" 2. Call lstm_autoencoder_dataset()")
    print(" 3. Call train_autoencoder_hybrid()")
    print(" 4. Call test_hybrid_model()")
    print("\n See code comments for detailed example.")
