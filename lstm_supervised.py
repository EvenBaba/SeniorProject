from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout # type: ignore
from tensorflow.keras.callbacks import EarlyStopping # type: ignore

import numpy as np
import pandas as pd


def supervised_lstm_dataset(df, lookback=100, train_ratio=0.9):
    """
    FIXED VERSION - NO DATA LEAKAGE
    
    Supervised LSTM dataset preparation with proper scaling
    to prevent data leakage.
    
    Parameters:
    -----------
    df : DataFrame
        Input dataframe with features and Anomaly_Statistical label
    lookback : int
        Number of time steps to look back
    train_ratio : float
        Ratio of data to use for training (for scaler fitting)
        
    Returns:
    --------
    X : array
        3D array of features (samples, timesteps, features)
    y : array
        1D array of labels (0/1)
    idx : list
        List of datetime indices
    scaler : StandardScaler
        Fitted scaler (fit only on training data)
    """
    
    labels = df["Anomaly_Statistical"].values  # 0/1

    FEATURE_COLS = [
        c for c in df.columns 
        if not c.startswith("Anomaly") and c != "Anomaly_Statistical"
    ]
    features = df[FEATURE_COLS].values
    
    print(f"\n[SUPERVISED LSTM DATASET]")
    print(f"  Total samples: {len(df)}")
    print(f"  Features: {len(FEATURE_COLS)}")
    print(f"  Lookback: {lookback}")

    # ---------- TIME-SERIES SPLIT (BEFORE SCALING!) ----------
    n_samples = len(df)
    split_idx = int(n_samples * train_ratio)
    
    features_train = features[:split_idx]
    features_test = features[split_idx:]
    
    print(f"\n[DATA SPLIT]")
    print(f"  Train samples: {len(features_train)} ({train_ratio*100:.0f}%)")
    print(f"  Test samples:  {len(features_test)} ({(1-train_ratio)*100:.0f}%)")
    
    # ---------- Scaling - FIT ONLY ON TRAIN DATA! ----------
    scaler = StandardScaler()
    
    # FIT ONLY ON TRAIN DATA
    scaler.fit(features_train)
    
    # TRANSFORM BOTH TRAIN AND TEST
    features_scaled = scaler.transform(features)
    
    print(f"\n[SCALING]")
    print(f"  Fitted on: {len(features_train)} train samples only")
    print(f"  Feature mean (train): {features_train.mean():.4f}")
    print(f"  Feature std (train):  {features_train.std():.4f}")

    # ---------- Create sequences ----------
    X, y, idx = [], [], []
    
    for i in range(lookback, len(df)):
        X.append(features_scaled[i-lookback:i])
        y.append(labels[i])
        idx.append(df.index[i])

    X = np.array(X)
    y = np.array(y)
    
    print(f"\n[SEQUENCES CREATED]")
    print(f"  X shape: {X.shape}")
    print(f"  y shape: {y.shape}")
    print(f"  Sequences: {len(X)}")
    print(f"  Anomaly rate: {y.mean():.2%}")
    
    return X, y, idx, scaler


def build_supervised_lstm(input_shape):
    """
    Build supervised LSTM model for binary classification
    """
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model


def train_supervised_lstm(X, y, use_early_stopping=True, epochs=100, batch_size=16):
    """
    Supervised LSTM training (time-series aware).
    """
    n = X.shape[0]
    split = int(n * 0.9)

    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]
    
    print(f"\n[MODEL TRAINING]")
    print(f"  Train sequences: {len(X_train)}")
    print(f"  Val sequences:   {len(X_val)}")
    print(f"  Train anomaly rate: {y_train.mean():.2%}")
    print(f"  Val anomaly rate:   {y_val.mean():.2%}")

    input_shape = (X.shape[1], X.shape[2])
    model = build_supervised_lstm(input_shape)

    callbacks = []
    if use_early_stopping:
        callbacks.append(
            EarlyStopping(
                monitor="val_loss",
                patience=8,
                restore_best_weights=True
            )
        )

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        shuffle=False,   # time-series
        callbacks=callbacks,
        verbose=1
    )

    return model, history


def test_supervised_lstm(
    model,
    X_test,
    y_test,
    idx_test,
    threshold=0.5
):
    """
    Supervised LSTM test & anomaly detection
    """
    y_prob = model.predict(X_test, verbose=0).ravel()
    y_pred = (y_prob >= threshold).astype(int)

    out = pd.DataFrame(index=idx_test)
    out["Anomaly_True"] = y_test
    out["Anomaly_Prob"] = y_prob
    out["Anomaly_Pred"] = y_pred
    
    print(f"\n[TEST RESULTS]")
    print(f"  Test samples:     {len(out)}")
    print(f"  True anomalies:   {y_test.sum()}")
    print(f"  Predicted (threshold={threshold}): {y_pred.sum()}")
    print(f"  Prediction rate:  {y_pred.mean():.2%}")

    return out
