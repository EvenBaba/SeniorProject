# lstm_forecast.py
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout # type: ignore
from tensorflow.keras.callbacks import EarlyStopping # type: ignore


def train_test_split_lstm(X, y, idx_y, train_ratio=0.9):
    n = len(X)
    split = int(n * train_ratio)

    return (
        X[:split], y[:split], idx_y[:split],   # train
        X[split:], y[split:], idx_y[split:]    # test
    )


def unsupervised_lstm_dataset(df, lookback=100, train_ratio=0.9):
    """
    Unsupervised LSTM (forecast) dataset - DATA LEAKAGE FIXED VERSION
    
    X : son lookback günün tüm feature'lari
    y : bugünün Close değeri (scaled)
    
    CRITICAL: Scaling ONLY on training data to prevent data leakage!
    """
    # ---------- Feature seçimi (supervised ile AYNI) ----------
    FEATURE_COLS = [
        c for c in df.columns
        if not c.startswith("Anomaly") and c != "Anomaly_Statistical"
    ]

    if "Close" not in FEATURE_COLS:
        raise ValueError("'Close' must be included in FEATURE_COLS")

    data = df[FEATURE_COLS].dropna()

    if len(data) <= lookback:
        raise ValueError(f"Not enough data. Need > lookback ({lookback}) rows.")

    # ---------- TIME-SERIES SPLIT (BEFORE SCALING!) ----------
    n_samples = len(data)
    split_idx = int(n_samples * train_ratio)
    
    data_train = data.iloc[:split_idx]
    data_test = data.iloc[split_idx:]
    
    print(f"\n[DATA SPLIT]")
    print(f"  Total samples: {n_samples}")
    print(f"  Train samples: {len(data_train)} ({train_ratio*100:.0f}%)")
    print(f"  Test samples:  {len(data_test)} ({(1-train_ratio)*100:.0f}%)")
    
    # ---------- Scaling - FIT ONLY ON TRAIN DATA! ----------
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    
    # FIT ONLY ON TRAIN DATA
    x_scaler.fit(data_train.values)
    
    close_idx = FEATURE_COLS.index("Close")
    y_scaler.fit(data_train.iloc[:, close_idx].values.reshape(-1, 1))
    
    # TRANSFORM BOTH TRAIN AND TEST
    data_scaled = x_scaler.transform(data.values)
    close_scaled = y_scaler.transform(
        data.iloc[:, close_idx].values.reshape(-1, 1)
    ).ravel()
    
    print(f"\n[SCALING]")
    print(f"  Fitted on: {len(data_train)} train samples only")
    print(f"  Feature mean (train): {data_train.mean().mean():.4f}")
    print(f"  Feature std (train):  {data_train.std().mean():.4f}")

    # ---------- Create sequences ----------
    X, y, idx = [], [], []

    for i in range(lookback, len(data)):
        X.append(data_scaled[i - lookback:i])
        y.append(close_scaled[i])
        idx.append(data.index[i])

    X = np.array(X)
    y = np.array(y)
    
    print(f"\n[SEQUENCES CREATED]")
    print(f"  X shape: {X.shape}")
    print(f"  y shape: {y.shape}")
    print(f"  Sequences: {len(X)}")
    
    return X, y, idx, x_scaler, y_scaler


def build_unsupervised_lstm(input_shape):
    """
    input_shape = (lookback, n_features)
    """
    model = Sequential([
        LSTM(64, return_sequences=True, activation="tanh", input_shape=input_shape),
        Dropout(0.2),
        LSTM(32, return_sequences=False, activation="tanh"),
        Dropout(0.2),
        Dense(1, activation="linear")
    ])
    model.compile(optimizer="adam", loss="mae")
    return model


def train_unsupervised_lstm(X, y, use_early_stopping=True, epochs=100, batch_size=16):
    """
    Train unsupervised LSTM model for time-series forecasting
    """
    n = X.shape[0]
    split = int(n * 0.9)
    X_train, y_train = X[:split], y[:split]
    X_val, y_val = X[split:], y[split:]

    print(f"\n[MODEL TRAINING]")
    print(f"  Train sequences: {len(X_train)}")
    print(f"  Val sequences:   {len(X_val)}")
    
    model = build_unsupervised_lstm((X.shape[1], X.shape[2]))

    callbacks = []
    if use_early_stopping:
        callbacks.append(
            EarlyStopping(
                monitor="val_loss",
                patience=10,
                restore_best_weights=True
            )
        )

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        shuffle=False,
        callbacks=callbacks,
        verbose=1
    )

    return model, history


def compute_threshold(model, X_train, y_train, y_scaler, k=3.0):
    """
    Compute anomaly threshold based on training data errors
    
    FIXED VERSION: Uses quantile-based approach
    k parameter now represents the quantile (e.g., k=0.95 means Q95)
    """
    y_pred = model.predict(X_train, verbose=0).ravel()

    y_true = y_scaler.inverse_transform(y_train.reshape(-1, 1)).ravel()
    y_pred = y_scaler.inverse_transform(y_pred.reshape(-1, 1)).ravel()

    err = np.abs(y_true - y_pred)
    
    # FIXED: Use quantile-based threshold
    # If k >= 1, treat it as old-style multiplier (backward compatibility)
    # If k < 1, treat it as quantile (recommended)
    if k >= 1.0:
        # Old style: mean + k*std (usually produces high threshold)
        threshold = err.mean() + k * err.std()
        method = f"mean + {k:.1f}*std"
    else:
        # New style: quantile-based (recommended)
        threshold = np.quantile(err, k)
        method = f"Q{int(k*100)}"
    
    print(f"\n[THRESHOLD COMPUTATION]")
    print(f"  Train error mean: {err.mean():.4f}")
    print(f"  Train error std:  {err.std():.4f}")
    print(f"  Train error min:  {err.min():.4f}")
    print(f"  Train error max:  {err.max():.4f}")
    print(f"  Method:           {method}")
    print(f"  k parameter:      {k}")
    print(f"  Threshold:        {threshold:.4f}")
    print(f"  Anomaly rate:     {(err > threshold).mean():.2%} (on train)")
    
    return threshold



def test_unsupervised_lstm(
    model,
    X_test,
    y_test,
    idx_test,
    y_scaler,
    threshold
):
    """
    Unsupervised LSTM test & anomaly detection
    """
    y_pred = model.predict(X_test, verbose=0).ravel()

    y_true = y_scaler.inverse_transform(y_test.reshape(-1, 1)).ravel()
    y_pred = y_scaler.inverse_transform(y_pred.reshape(-1, 1)).ravel()

    err = np.abs(y_true - y_pred)

    out = pd.DataFrame(index=idx_test)
    out["Close_True"] = y_true
    out["Close_Pred"] = y_pred
    out["Forecast_Error"] = err
    out["Threshold"] = threshold
    out["Anomaly_LSTM"] = (err > threshold).astype(int)
    
    print(f"\n[TEST RESULTS]")
    print(f"  Test samples:     {len(out)}")
    print(f"  Anomalies found:  {out['Anomaly_LSTM'].sum()}")
    print(f"  Anomaly rate:     {out['Anomaly_LSTM'].mean():.2%}")

    return out
