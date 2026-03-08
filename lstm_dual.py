"""
LSTM Dual-Stream Model - CS402 Phase 2A
========================================
Dual-output architecture with shared LSTM body:
  - Stream 1: Anomaly Classifier  (sigmoid)   → detects anomalies
  - Stream 2: Price Regressor     (linear)    → forecasts next-day Close price

Surprise Factor = |actual_price - predicted_price| / actual_price × 100
  → Quantifies how "surprised" the model was → validates anomaly severity.
  → High Surprise Factor on an anomaly day = model was genuinely caught off guard.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Model                           # type: ignore
from tensorflow.keras.layers import (Input, LSTM, Dense, Dropout)  # type: ignore
from tensorflow.keras.callbacks import EarlyStopping               # type: ignore


# ============================================================================
# DATASET PREPARATION
# ============================================================================

def dual_lstm_dataset(df, lookback=100, train_ratio=0.9, label_col="Anomaly_Statistical"):
    """
    Prepare dataset for Dual-Stream LSTM.

    Two targets per sample:
      y_anomaly : 0/1  — from Anomaly_Statistical column
      y_price   : float — next-day Close price (scaled)

    Scaler policy (NO data leakage):
      - x_scaler   : StandardScaler fitted ONLY on training feature rows
      - price_scaler: StandardScaler fitted ONLY on training Close values

    Parameters
    ----------
    df          : DataFrame with features + Anomaly_Statistical column
    lookback    : int   — sequence length (default 100)
    train_ratio : float — train/test split ratio (default 0.9)

    Returns
    -------
    X             : ndarray (samples, lookback, n_features)
    y_anomaly     : ndarray (samples,)   — binary anomaly labels
    y_price       : ndarray (samples,)   — scaled next-day Close
    idx           : list                 — datetime index aligned with samples
    x_scaler      : fitted StandardScaler for features
    price_scaler  : fitted StandardScaler for Close price
    """

    print("\n" + "=" * 80)
    print("DUAL-STREAM LSTM — DATASET PREPARATION")
    print("=" * 80)

    # ------------------------------------------------------------------
    # 1. Feature selection  (exclude label columns)
    # ------------------------------------------------------------------
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found in dataframe.")

    FEATURE_COLS = [
        c for c in df.columns
        if not c.startswith("Anomaly") and c != "Anomaly_Statistical"
    ]
    if "Close" not in FEATURE_COLS:
        raise ValueError("'Close' must be present in the dataframe features.")

    data = df[FEATURE_COLS].dropna()
    labels = df[label_col].loc[data.index].values
    print(f"\n[LABEL SOURCE]  Using column: '{label_col}'  "
          f"(anomaly rate: {labels.mean():.2%})")
    close_prices = data["Close"].values

    n_samples = len(data)
    split_idx  = int(n_samples * train_ratio)

    print(f"\n[SPLIT]")
    print(f"  Total  : {n_samples}")
    print(f"  Train  : {split_idx}  ({train_ratio*100:.0f}%)")
    print(f"  Test   : {n_samples - split_idx}  ({(1-train_ratio)*100:.0f}%)")

    # ------------------------------------------------------------------
    # 2. Scaling — fit ONLY on training portion
    # ------------------------------------------------------------------
    x_scaler = StandardScaler()
    x_scaler.fit(data.values[:split_idx])
    features_scaled = x_scaler.transform(data.values)

    close_idx = list(FEATURE_COLS).index("Close")

    price_scaler = StandardScaler()
    price_scaler.fit(close_prices[:split_idx].reshape(-1, 1))
    close_scaled = price_scaler.transform(
        close_prices.reshape(-1, 1)
    ).ravel()

    print(f"\n[SCALING]  Fitted on {split_idx} train samples only.")

    # ------------------------------------------------------------------
    # 3. Build sequences  X[i] = features[i-lookback : i]
    #                     targets aligned to step i (current day)
    # ------------------------------------------------------------------
    X, y_anomaly, y_price, idx = [], [], [], []

    for i in range(lookback, n_samples):
        X.append(features_scaled[i - lookback : i])   # past lookback days
        y_anomaly.append(labels[i])                    # is today anomalous?
        y_price.append(close_scaled[i])                # today's scaled Close
        idx.append(data.index[i])

    X         = np.array(X,         dtype=np.float32)
    y_anomaly = np.array(y_anomaly, dtype=np.float32)
    y_price   = np.array(y_price,   dtype=np.float32)

    print(f"\n[SEQUENCES]")
    print(f"  X shape        : {X.shape}")
    print(f"  y_anomaly shape: {y_anomaly.shape}  "
          f"(anomaly rate: {y_anomaly.mean():.2%})")
    print(f"  y_price shape  : {y_price.shape}")
    print("=" * 80 + "\n")

    return X, y_anomaly, y_price, idx, x_scaler, price_scaler


# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

def build_dual_lstm(input_shape):
    """
    Shared-body LSTM with two output heads.

    Architecture
    ------------
    Input (lookback, n_features)
      → LSTM(128, return_sequences=True)
      → Dropout(0.2)
      → LSTM(64, return_sequences=False)
      → Dropout(0.2)
      → Dense(32, relu)          ← shared representation
            ├── Dense(1, sigmoid, name='anomaly')   ← Head 1
            └── Dense(1, linear,  name='price')     ← Head 2

    Parameters
    ----------
    input_shape : tuple (lookback, n_features)

    Returns
    -------
    model : compiled Keras Model
    """

    print("\n[BUILDING DUAL-STREAM LSTM]")
    print(f"  Input shape : {input_shape}")

    inputs = Input(shape=input_shape, name="input")

    # --- Shared body ---
    x = LSTM(128, return_sequences=True, activation="tanh",
             name="shared_lstm_1")(inputs)
    x = Dropout(0.2, name="dropout_1")(x)

    x = LSTM(64, return_sequences=False, activation="tanh",
             name="shared_lstm_2")(x)
    x = Dropout(0.2, name="dropout_2")(x)

    shared = Dense(32, activation="relu", name="shared_dense")(x)

    # --- Head 1: Anomaly Classifier ---
    anomaly_out = Dense(1, activation="sigmoid", name="anomaly")(shared)

    # --- Head 2: Price Regressor ---
    price_out = Dense(1, activation="linear", name="price")(shared)

    model = Model(inputs=inputs, outputs=[anomaly_out, price_out],
                  name="DualStreamLSTM")

    model.compile(
        optimizer="adam",
        loss={
            "anomaly": "binary_crossentropy",
            "price":   "mae",
        },
        loss_weights={
            "anomaly": 1.0,   # classification head has full weight
            "price":   0.5,   # regression head has half weight
        },
        metrics={
            "anomaly": "accuracy",
        }
    )

    print(f"  Total parameters : {model.count_params():,}")
    print(f"  Head 1 (anomaly) : binary_crossentropy  weight=1.0")
    print(f"  Head 2 (price)   : mae                  weight=0.5")

    return model


# ============================================================================
# TRAINING
# ============================================================================

def train_dual_lstm(X, y_anomaly, y_price,
                    train_ratio=0.9,
                    epochs=100, batch_size=16,
                    use_early_stopping=True):
    """
    Train the dual-stream model.

    An internal validation split (last 10% of training data) is used
    for early stopping — strictly no test data leakage.

    Parameters
    ----------
    X, y_anomaly, y_price : arrays from dual_lstm_dataset()
    train_ratio           : must match the value used in dataset prep
    epochs, batch_size    : training hyper-parameters
    use_early_stopping    : bool (default True)

    Returns
    -------
    model   : trained Keras Model
    history : training History object
    """

    n = len(X)
    split = int(n * train_ratio)

    X_train,     X_test     = X[:split],         X[split:]
    ya_train,    ya_test    = y_anomaly[:split],  y_anomaly[split:]
    yp_train,    yp_test    = y_price[:split],    y_price[split:]

    # Internal val split from the training portion only
    n_train  = len(X_train)
    val_split_idx = int(n_train * 0.9)

    X_tr,   X_val   = X_train[:val_split_idx],  X_train[val_split_idx:]
    ya_tr,  ya_val  = ya_train[:val_split_idx],  ya_train[val_split_idx:]
    yp_tr,  yp_val  = yp_train[:val_split_idx],  yp_train[val_split_idx:]

    print(f"\n[TRAINING]")
    print(f"  Train sequences      : {len(X_tr)}")
    print(f"  Validation sequences : {len(X_val)}")
    print(f"  Test  sequences      : {len(X_test)}  (held out)")
    print(f"  Train anomaly rate   : {ya_tr.mean():.2%}")

    model = build_dual_lstm((X.shape[1], X.shape[2]))

    callbacks = []
    if use_early_stopping:
        callbacks.append(
            EarlyStopping(
                monitor="val_loss",
                patience=10,
                restore_best_weights=True,
                verbose=1
            )
        )

    history = model.fit(
        X_tr,
        {"anomaly": ya_tr, "price": yp_tr},
        validation_data=(X_val, {"anomaly": ya_val, "price": yp_val}),
        epochs=epochs,
        batch_size=batch_size,
        shuffle=False,        # preserve time-series order
        callbacks=callbacks,
        verbose=1
    )

    print(f"\n  Training complete — {len(history.history['loss'])} epochs.")
    return model, history


# ============================================================================
# SURPRISE FACTOR COMPUTATION
# ============================================================================

def compute_surprise_factor(price_pred_scaled, price_true_scaled,
                             price_scaler, idx,
                             train_ratio=0.9):
    """
    Compute Surprise Factor from price regressor outputs.

    Surprise Factor (%) = |actual_close - predicted_close| / actual_close × 100

    A Surprise Factor Z-score is also computed (using training error statistics)
    so severity can be compared across time.

    Parameters
    ----------
    price_pred_scaled : ndarray  — model's scaled price predictions
    price_true_scaled : ndarray  — true scaled prices
    price_scaler      : fitted StandardScaler for Close
    idx               : list of datetime indices (aligned with arrays)
    train_ratio       : used to split train stats from test stats

    Returns
    -------
    sf_df : DataFrame with columns:
        Close_True        — actual price (unscaled)
        Close_Pred        — predicted price (unscaled)
        Price_Error       — absolute price error
        Surprise_Factor   — percentage deviation from actual price
        Surprise_Factor_Z — z-scored surprise (using train error distribution)
    """

    # Inverse-transform to original price space
    close_true = price_scaler.inverse_transform(
        price_true_scaled.reshape(-1, 1)
    ).ravel()
    close_pred = price_scaler.inverse_transform(
        price_pred_scaled.reshape(-1, 1)
    ).ravel()

    price_error      = np.abs(close_true - close_pred)
    surprise_factor  = (price_error / close_true) * 100.0   # percentage

    # Z-score normalisation using training-period error stats
    n_all  = len(surprise_factor)
    n_train = int(n_all * train_ratio)
    sf_mean = surprise_factor[:n_train].mean()
    sf_std  = surprise_factor[:n_train].std()
    sf_std  = sf_std if sf_std > 1e-8 else 1.0               # avoid div/0
    surprise_factor_z = (surprise_factor - sf_mean) / sf_std

    sf_df = pd.DataFrame(index=idx)
    sf_df["Close_True"]        = close_true
    sf_df["Close_Pred"]        = close_pred
    sf_df["Price_Error"]       = price_error
    sf_df["Surprise_Factor"]   = surprise_factor
    sf_df["Surprise_Factor_Z"] = surprise_factor_z

    print(f"\n[SURPRISE FACTOR STATISTICS — FULL SET]")
    print(f"  Mean  : {surprise_factor.mean():.4f} %")
    print(f"  Std   : {surprise_factor.std():.4f} %")
    print(f"  Max   : {surprise_factor.max():.4f} %  "
          f"(on {sf_df['Surprise_Factor'].idxmax().date()})")

    return sf_df


# ============================================================================
# TESTING
# ============================================================================

def test_dual_lstm(model, X_test, y_anomaly_test, y_price_test,
                   idx_test, price_scaler,
                   anomaly_threshold=0.3):
    """
    Run the dual-stream model on the test set and return a results DataFrame.

    Parameters
    ----------
    model             : trained dual-stream Keras Model
    X_test            : ndarray  (test samples, lookback, n_features)
    y_anomaly_test    : ndarray  true anomaly labels
    y_price_test      : ndarray  true scaled Close prices
    idx_test          : list     datetime indices for test samples
    price_scaler      : fitted StandardScaler for Close
    anomaly_threshold : float   classification threshold (default 0.3)

    Returns
    -------
    results_df : DataFrame with columns:
        Anomaly_True, Anomaly_Prob, Anomaly_Pred,
        Close_True, Close_Pred, Price_Error,
        Surprise_Factor, Surprise_Factor_Z
    """

    print("\n" + "=" * 80)
    print("DUAL-STREAM LSTM — TEST")
    print("=" * 80)

    # --- Predict both heads ---
    anomaly_prob, price_pred_scaled = model.predict(X_test, verbose=0)
    anomaly_prob        = anomaly_prob.ravel()
    price_pred_scaled   = price_pred_scaled.ravel()

    anomaly_pred = (anomaly_prob >= anomaly_threshold).astype(int)

    # --- Build Surprise Factor dataframe ---
    sf_df = compute_surprise_factor(
        price_pred_scaled=price_pred_scaled,
        price_true_scaled=y_price_test,
        price_scaler=price_scaler,
        idx=idx_test,
        train_ratio=1.0           # All test stats; train stats computed inside
    )

    # --- Assemble results ---
    results_df = pd.DataFrame(index=idx_test)
    results_df["Anomaly_True"]       = y_anomaly_test.astype(int)
    results_df["Anomaly_Prob"]       = anomaly_prob
    results_df["Anomaly_Pred"]       = anomaly_pred
    results_df["Close_True"]         = sf_df["Close_True"].values
    results_df["Close_Pred"]         = sf_df["Close_Pred"].values
    results_df["Price_Error"]        = sf_df["Price_Error"].values
    results_df["Surprise_Factor"]    = sf_df["Surprise_Factor"].values
    results_df["Surprise_Factor_Z"]  = sf_df["Surprise_Factor_Z"].values

    # --- Quick stats ---
    print(f"\n[TEST RESULTS]")
    print(f"  Test samples     : {len(results_df)}")
    print(f"  True anomalies   : {results_df['Anomaly_True'].sum()}")
    print(f"  Predicted (t={anomaly_threshold}): {results_df['Anomaly_Pred'].sum()}")

    # Surprise Factor on confirmed anomaly days vs normal days
    anom_mask   = results_df["Anomaly_True"] == 1
    normal_mask = results_df["Anomaly_True"] == 0

    if anom_mask.sum() > 0:
        sf_anom   = results_df.loc[anom_mask,   "Surprise_Factor"].mean()
        sf_normal = results_df.loc[normal_mask,  "Surprise_Factor"].mean()
        print(f"\n  Surprise Factor — anomaly days : {sf_anom:.4f} %")
        print(f"  Surprise Factor — normal  days : {sf_normal:.4f} %")
        print(f"  Ratio (anomaly / normal)       : {sf_anom/sf_normal:.2f}×")

    print("=" * 80 + "\n")

    return results_df


# ============================================================================
# FULL PIPELINE (convenience wrapper)
# ============================================================================

def run_dual_lstm_pipeline(df, config, label_col="Anomaly_Statistical"):
    """
    End-to-end dual-stream LSTM pipeline.

    Steps
    -----
    1. Build dataset
    2. Train model
    3. Test model + compute Surprise Factor
    4. Return evaluation-ready dict + detailed results DataFrame

    Parameters
    ----------
    df     : preprocessed DataFrame (from statistic.py)
    config : CONFIG dict from config.py

    Returns
    -------
    eval_result : dict  { y_true, y_pred, y_prob }
    results_df  : DataFrame  (full per-day results incl. Surprise Factor)
    """

    print("\n" + "=" * 80)
    print("DUAL-STREAM LSTM PIPELINE — START")
    print("=" * 80)

    lookback    = config["lookback"]
    train_ratio = config["train_ratio"]
    epochs      = config.get("dual_epochs",     100)
    batch_size  = config.get("dual_batch_size",  16)
    threshold   = config.get("dual_threshold",   0.3)

    # ---- 1. Dataset ----
    X, y_anomaly, y_price, idx, x_scaler, price_scaler = dual_lstm_dataset(
        df, lookback=lookback, train_ratio=train_ratio, label_col=label_col
    )

    # ---- 2. Train ----
    model, history = train_dual_lstm(
        X, y_anomaly, y_price,
        train_ratio=train_ratio,
        epochs=epochs,
        batch_size=batch_size,
        use_early_stopping=True
    )

    # ---- 3. Test (on held-out test portion) ----
    n       = len(X)
    split   = int(n * train_ratio)

    X_test        = X[split:]
    ya_test       = y_anomaly[split:]
    yp_test       = y_price[split:]
    idx_test      = idx[split:]

    results_df = test_dual_lstm(
        model, X_test, ya_test, yp_test,
        idx_test, price_scaler,
        anomaly_threshold=threshold
    )

    # ---- 4. Evaluation dict ----
    eval_result = {
        "y_true": results_df["Anomaly_True"].values,
        "y_pred": results_df["Anomaly_Pred"].values,
        "y_prob": results_df["Anomaly_Prob"].values,
    }

    print("\nDual-Stream LSTM pipeline complete.")
    return eval_result, results_df
