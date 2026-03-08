"""
GROUND TRUTH LABELS — Known Real-World BTC Anomaly Events
===========================================================
Instead of relying on statistical methods (Z-Score / EWMA) to define
what counts as an anomaly, this module uses documented, real-world
market events as the ground truth.

Each event is a date where BTC experienced a significant, news-driven
price movement that is publicly verifiable.

Event types:
  crash    — sharp price drop driven by external event
  pump     — sharp price rise driven by external event
  volatile — extreme two-way volatility (e.g. halvings, regulatory decisions)
"""

import pandas as pd
import numpy as np


# ============================================================================
# KNOWN BTC ANOMALY EVENTS
# ============================================================================

KNOWN_BTC_EVENTS = {
    # ---- 2023 ----
    "2023-06-05": {
        "description": "SEC sues Binance, charges with operating unregistered exchange",
        "type": "crash"
    },
    "2023-06-06": {
        "description": "SEC sues Coinbase — continued sell-off",
        "type": "crash"
    },
    "2023-08-17": {
        "description": "BTC flash crash -10% from ~$29k to ~$26k (liquidation cascade)",
        "type": "crash"
    },
    "2023-10-23": {
        "description": "BlackRock Bitcoin ETF ticker appears on DTCC — speculation pump",
        "type": "pump"
    },

    # ---- 2024 ----
    "2024-01-10": {
        "description": "SEC approves Bitcoin Spot ETFs (BlackRock, Fidelity, etc.)",
        "type": "pump"
    },
    "2024-01-11": {
        "description": "First trading day of Bitcoin Spot ETFs — high volatility",
        "type": "volatile"
    },
    "2024-02-28": {
        "description": "BTC reclaims $60k for first time since 2021 bull run",
        "type": "pump"
    },
    "2024-03-05": {
        "description": "BTC breaks 2021 all-time high (~$69k) — new ATH",
        "type": "pump"
    },
    "2024-03-14": {
        "description": "BTC reaches new ATH ~$73,750",
        "type": "pump"
    },
    "2024-04-19": {
        "description": "Pre-halving volatility — BTC drops ~$60k",
        "type": "volatile"
    },
    "2024-04-20": {
        "description": "Bitcoin 4th Halving — block reward drops from 6.25 to 3.125 BTC",
        "type": "volatile"
    },
    "2024-06-07": {
        "description": "BTC drops ~10% on US jobs data / macro sell-off",
        "type": "crash"
    },
    "2024-08-05": {
        "description": "Global market crash — Japan carry trade unwind, BTC drops ~15%",
        "type": "crash"
    },
    "2024-08-06": {
        "description": "Continued sell-off from Japan carry trade unwind",
        "type": "crash"
    },
    "2024-09-18": {
        "description": "US Fed cuts interest rates 50bps — risk assets including BTC pump",
        "type": "pump"
    },
    "2024-11-05": {
        "description": "US Election Day — Trump wins presidency, BTC begins major rally",
        "type": "pump"
    },
    "2024-11-06": {
        "description": "BTC surges past $75k on Trump election win",
        "type": "pump"
    },
    "2024-11-13": {
        "description": "BTC crosses $90k for the first time",
        "type": "pump"
    },
    "2024-12-05": {
        "description": "BTC crosses $100k for the first time in history",
        "type": "pump"
    },
    "2024-12-18": {
        "description": "Fed signals fewer rate cuts in 2025 — BTC flash crash from ~$108k",
        "type": "crash"
    },

    # ---- 2025 (up to knowledge cutoff Aug 2025) ----
    "2025-01-20": {
        "description": "Trump inauguration — BTC volatile, initial pump then correction",
        "type": "volatile"
    },
    "2025-02-03": {
        "description": "BTC flash crash ~$90k → ~$78k on tariff war escalation fears",
        "type": "crash"
    },
    "2025-03-04": {
        "description": "Trump announces US Strategic Bitcoin Reserve — BTC pump",
        "type": "pump"
    },
}


# ============================================================================
# LABEL CREATION
# ============================================================================

def create_ground_truth_labels(df, window_days=1, verbose=True):
    """
    Create binary anomaly labels from known real-world BTC events.

    For each known event date, marks that day AND the surrounding
    window_days as anomalies (price impact often spans multiple days).

    Parameters
    ----------
    df          : DataFrame with DatetimeIndex (from statistic.py)
    window_days : int  — days around each event to also mark as anomaly
                         (e.g. 1 means the event day ± 1 day)
    verbose     : bool — print summary

    Returns
    -------
    df : DataFrame with new column 'Anomaly_GroundTruth' (0/1)
    event_report : DataFrame — which known events were found in the data
    """

    df = df.copy()
    df["Anomaly_GroundTruth"] = 0

    df_index_dates = df.index.normalize()   # strip time, keep date only

    found_events   = []
    missing_events = []

    for date_str, info in KNOWN_BTC_EVENTS.items():
        event_date = pd.Timestamp(date_str)

        # Mark event day ± window_days
        for offset in range(-window_days, window_days + 1):
            target_date = event_date + pd.Timedelta(days=offset)
            mask = df_index_dates == target_date
            if mask.any():
                df.loc[mask, "Anomaly_GroundTruth"] = 1

        # Check if the core event date is in data
        core_mask = df_index_dates == event_date
        if core_mask.any():
            row = df[core_mask].iloc[0]
            found_events.append({
                "Date":        date_str,
                "Type":        info["type"],
                "Description": info["description"],
                "Close":       row["Close"],
                "Log_Return":  row.get("Log_Return", float("nan")),
                "Z_Score":     row.get("Price_LogReturn_Z_Score", float("nan")),
                "Stat_Label":  row.get("Anomaly_Statistical", -1),
                "GT_Label":    1,
            })
        else:
            missing_events.append(date_str)

    event_report = pd.DataFrame(found_events)

    if verbose:
        print("\n" + "=" * 80)
        print("GROUND TRUTH LABELS — KNOWN BTC EVENTS")
        print("=" * 80)

        print(f"\n  Total known events defined : {len(KNOWN_BTC_EVENTS)}")
        print(f"  Events found in data       : {len(found_events)}")
        print(f"  Events outside data range  : {len(missing_events)}")
        print(f"  Window applied             : ± {window_days} day(s)")
        print(f"\n  Ground truth anomaly days  : {df['Anomaly_GroundTruth'].sum()}")
        print(f"  Ground truth anomaly rate  : {df['Anomaly_GroundTruth'].mean():.2%}")

        if missing_events:
            print(f"\n  [OUTSIDE DATA RANGE] {missing_events}")

        if len(found_events) > 0:
            print(f"\n  {'Date':<12} {'Type':<10} {'Close':>10} {'Log_Ret':>9} "
                  f"{'Z_Score':>8} {'StatLbl':>8} {'Description'}")
            print("  " + "-" * 100)
            for _, row in event_report.iterrows():
                stat_lbl = int(row['Stat_Label']) if row['Stat_Label'] != -1 else "N/A"
                print(f"  {row['Date']:<12} {row['Type']:<10} "
                      f"${row['Close']:>10,.0f} "
                      f"{row['Log_Return']:>9.4f} "
                      f"{row['Z_Score']:>8.3f} "
                      f"{str(stat_lbl):>8} "
                      f"  {row['Description'][:60]}")

        # Agreement analysis between statistical and ground truth
        if "Anomaly_Statistical" in df.columns and len(found_events) > 0:
            both_flagged = (
                (df["Anomaly_GroundTruth"] == 1) &
                (df["Anomaly_Statistical"] == 1)
            ).sum()
            gt_only = (
                (df["Anomaly_GroundTruth"] == 1) &
                (df["Anomaly_Statistical"] == 0)
            ).sum()
            stat_only = (
                (df["Anomaly_GroundTruth"] == 0) &
                (df["Anomaly_Statistical"] == 1)
            ).sum()

            print(f"\n  [AGREEMENT vs STATISTICAL LABELS]")
            print(f"  Both flagged (agree)     : {both_flagged}")
            print(f"  GT only (stat missed)    : {gt_only}")
            print(f"  Stat only (GT missed)    : {stat_only}")
            if df["Anomaly_GroundTruth"].sum() > 0:
                print(f"  Overlap rate             : "
                      f"{both_flagged / df['Anomaly_GroundTruth'].sum():.1%}")

        print("=" * 80 + "\n")

    return df, event_report


# ============================================================================
# STANDALONE RUN
# ============================================================================

if __name__ == "__main__":
    from statistic import fetch_cryptocurrency_data, data_preprocessing_and_feature_engineering

    df_raw = fetch_cryptocurrency_data()
    df, split_idx = data_preprocessing_and_feature_engineering(
        df_raw, train_ratio=0.9, create_labels=True
    )

    df_gt, report = create_ground_truth_labels(df, window_days=1)

    print("\nSample of ground truth labels on event dates:")
    print(df_gt[df_gt["Anomaly_GroundTruth"] == 1][
        ["Close", "Log_Return", "Anomaly_Statistical", "Anomaly_GroundTruth"]
    ])
