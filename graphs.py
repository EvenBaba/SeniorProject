import matplotlib.pyplot as plt

limit = 1000
THRESHOLD_Z = 2.0  # Z-Score threshold for anomaly detection

def plot_EWMA_graph(df, tail=limit):
    if tail > limit:
        tail = limit
    df = df.tail(tail).copy()
    # Filtering anomalies for plotting
    anomalies = df[df['Anomaly_EWMA'] == 1]
    num_of_anomaly = df['Anomaly_EWMA'].sum()

    plt.figure(figsize=(15, 6))
    plt.plot(df.index, df['Close'], label='Closing Prices', color='blue', linewidth=1.5)
    plt.plot(df.index, df['EWMA_Close'], label='EWMA Close', color='orange', linewidth=1.5)

    # Marking detected anomaly points on the graph
    plt.scatter(anomalies.index, anomalies['Close'], 
                color='red', 
                marker='o', 
                s=50, # Marker size
                label=f'EWMA Anomaly (Total: {num_of_anomaly})')

    plt.title('Anomaly Detection Using EWMA Method')
    plt.xlabel('Date')
    plt.ylabel('Closing Price (USDT)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()


def plot_zscore_graph(df, tail=limit):
    if tail > limit:
        tail = limit
    df = df.tail(tail).copy()
    # Filtering anomalies for plotting
    anomalies = df[df['Anomaly_ZScore'] == 1]
    num_of_anomaly = df['Anomaly_ZScore'].sum()

    plt.figure(figsize=(15, 6))
    plt.plot(df.index, df['Close'], label='Closing Prices', color='blue', linewidth=1.5)

    # Marking detected anomaly points on the graph
    plt.scatter(anomalies.index, anomalies['Close'], 
                color='red', 
                marker='o', 
                s=50, # Marker size
                label=f'Z-Score Anomaly (Total: {num_of_anomaly})')

    plt.title(f'Anomaly Detection Using Global Log Return Z-Score (Threshold: {THRESHOLD_Z} std dev)')
    plt.xlabel('Date')
    plt.ylabel('Closing Price (USDT)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()


def plot_unsupervised_vs_statistical(
    out,
    df_stat,
    title="Unsupervised LSTM vs Statistical Anomaly Detection"
):
    """
    out      : Unsupervised LSTM test output (DataFrame)
    df_stat  : Statistical anomaly DataFrame (Z-score, EWMA)
    """

    plt.figure(figsize=(14, 6))

    # --- Actual price ---
    plt.plot(
        df_stat.index,
        df_stat["Close"],
        label="Close Price",
        color="black",
        alpha=0.6
    )

    # --- Unsupervised LSTM anomalies ---
    lstm_anom = out[out["Anomaly_LSTM"] == 1]
    plt.scatter(
        lstm_anom.index,
        lstm_anom["Close_True"],
        color="red",
        label="Unsupervised LSTM Anomaly",
        marker="x",
        s=80,
        zorder=3
    )

    # --- Statistical anomalies (Z-score OR EWMA) ---
    stat_anom = df_stat[
        (df_stat["Anomaly_ZScore"] == 1) |
        (df_stat["Anomaly_EWMA"] == 1)
    ]

    plt.scatter(
        stat_anom.index,
        stat_anom["Close"],
        color="blue",
        label="Statistical Anomaly",
        marker="o",
        facecolors="none",
        s=70,
        zorder=2
    )

    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

def plot_supervised_vs_statistical(
    out_sup,
    df_stat,
    title="Supervised LSTM vs Statistical Anomaly Detection",
    use_prob=False,
    prob_threshold=0.5
):
    """
    out_sup : output of test_supervised_lstm (index: date, columns: Anomaly_Prob, Anomaly_Pred)
    df_stat : df after statistic pipeline (has Close, Anomaly_ZScore, Anomaly_EWMA columns)

    If use_prob=True, selects anomalies where Anomaly_Prob >= prob_threshold.
    """

    plt.figure(figsize=(14, 6))

    # --- Actual price ---
    plt.plot(
        df_stat.index,
        df_stat["Close"],
        label="Close Price",
        color="black",
        alpha=0.6
    )

    # --- Supervised LSTM anomalies ---
    if use_prob:
        sup_anom = out_sup[out_sup["Anomaly_Prob"] >= prob_threshold]
        label_sup = f"Supervised LSTM Anomaly (prob>={prob_threshold})"
    else:
        sup_anom = out_sup[out_sup["Anomaly_Pred"] == 1]
        label_sup = "Supervised LSTM Anomaly"

    # out_sup has no Close column; pulling Close from df_stat by index
    sup_close = df_stat.reindex(sup_anom.index)["Close"]

    plt.scatter(
        sup_anom.index,
        sup_close,
        color="red",
        label=label_sup,
        marker="x",
        s=80,
        zorder=3
    )

    # --- Statistical anomalies (Z-score OR EWMA) ---
    stat_anom = df_stat[
        (df_stat["Anomaly_ZScore"] == 1) |
        (df_stat["Anomaly_EWMA"] == 1)
    ]

    plt.scatter(
        stat_anom.index,
        stat_anom["Close"],
        color="blue",
        label="Statistical Anomaly",
        marker="o",
        facecolors="none",
        s=70,
        zorder=2
    )

    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_price_unsupervised_vs_statistical(
    out,
    df_stat,
    last_n=100,
    title="Unsupervised LSTM Forecast vs Statistical Anomalies (Last 100 Days)"
):
    import matplotlib.pyplot as plt

    out_last = out.iloc[-last_n:]

    plt.figure(figsize=(14, 6))

    # --- Actual price ---
    plt.plot(
        out_last.index,
        out_last["Close_True"],
        label="Actual Close Price",
        color="black",
        linewidth=2
    )

    # --- LSTM forecast ---
    plt.plot(
        out_last.index,
        out_last["Close_Pred"],
        label="Unsupervised LSTM Forecast",
        linestyle="--",
        linewidth=2
    )

    # --- Statistical anomalies ---
    stat_last = df_stat.loc[out_last.index]
    stat_anom = stat_last[
        (stat_last["Anomaly_ZScore"] == 1) |
        (stat_last["Anomaly_EWMA"] == 1)
    ]

    plt.scatter(
        stat_anom.index,
        stat_anom["Close"],
        facecolors="none",
        edgecolors="blue",
        label="Statistical Anomaly",
        s=80,
        zorder=3
    )

    # ---  LSTM anomalies (forecast error based) ---
    lstm_anom = out_last[out_last["Anomaly_LSTM"] == 1]

    plt.scatter(
        lstm_anom.index,
        lstm_anom["Close_True"],
        color="red",
        marker="x",
        label="LSTM Anomaly",
        s=90,
        zorder=4
    )

    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_supervised_vs_statisticall(
    out_sup,
    df_stat,
    title="Supervised LSTM vs Statistical Anomaly Detection",
    use_prob=False,
    prob_threshold=0.5
):
    plt.figure(figsize=(14, 6))

    # --- LSTM test period only ---
    start_idx = out_sup.index.min()
    df_plot = df_stat.loc[start_idx:]

    # --- Actual price (SADECE SON 100 GN) ---
    plt.plot(
        df_plot.index,
        df_plot["Close"],
        label="Close Price",
        color="black",
        alpha=0.6
    )

    # --- Supervised LSTM anomalies ---
    if use_prob:
        sup_anom = out_sup[out_sup["Anomaly_Prob"] >= prob_threshold]
        label_sup = f"Supervised LSTM Anomaly (prob>={prob_threshold})"
    else:
        sup_anom = out_sup[out_sup["Anomaly_Pred"] == 1]
        label_sup = "Supervised LSTM Anomaly"

    sup_close = df_plot.reindex(sup_anom.index)["Close"]

    plt.scatter(
        sup_anom.index,
        sup_close,
        color="red",
        label=label_sup,
        marker="x",
        s=80,
        zorder=3
    )

    # --- Statistical anomalies (same time range) ---
    stat_anom = df_plot[
        (df_plot["Anomaly_ZScore"] == 1) |
        (df_plot["Anomaly_EWMA"] == 1)
    ]

    plt.scatter(
        stat_anom.index,
        stat_anom["Close"],
        color="blue",
        label="Statistical Anomaly",
        marker="o",
        facecolors="none",
        s=70,
        zorder=2
    )

    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()