"""
PHASE 2B -- Surprise Factor Causal Analysis
=============================================
No external API. No hardcoded event list. Fully generalizable to any asset.

Core idea
---------
The dual-LSTM has two heads:
  - Head 1 (classifier) : flags anomaly days
  - Head 2 (regressor)  : predicts next-day closing price

The Surprise Factor = |actual - predicted| / actual * 100

When the model flags an anomaly AND the Surprise Factor Z-score is HIGH,
it means the model could NOT anticipate the move from price history alone.
This is consistent with an EXTERNAL cause -- macro shock, regulatory event,
geopolitical news -- something outside the price patterns.

When the model flags an anomaly but the Surprise Factor Z-score is LOW,
the model already saw unusual patterns building before the move.
This is consistent with INTERNAL market mechanics -- liquidation cascade,
whale accumulation, technical breakout.

Causal verdict rule
-------------------
  SF_Z >= threshold  ->  "external_shock"
  SF_Z <  threshold  ->  "internal_mechanics"

This requires no news API and no hardcoded dates.
It works for any asset the dual-LSTM is trained on.

Output files
------------
  causal_verdicts.csv          -- per anomaly day: verdict, SF, SF_Z, price
  causal_sf_distribution.csv   -- full SF distribution for all days
  causal_monthly_pattern.csv   -- monthly external vs internal breakdown
  causal_summary.txt           -- human-readable report
"""

import os
import warnings
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

DEFAULT_SF_THRESHOLD = 1.0


# ============================================================================
# COMPONENT 1 -- CAUSAL VERDICT ENGINE
# ============================================================================

def assign_causal_verdicts(dual_results_df: pd.DataFrame,
                            sf_threshold: float = DEFAULT_SF_THRESHOLD,
                            anomaly_col: str = "Anomaly_Pred") -> pd.DataFrame:
    """
    Classify each anomaly day as external_shock or internal_mechanics
    using the Surprise Factor Z-score from the dual-LSTM price regressor.

    Parameters
    ----------
    dual_results_df : DataFrame from run_dual_lstm_pipeline()
    sf_threshold    : SF_Z cutoff for external_shock classification
    anomaly_col     : column to use as anomaly flag

    Returns
    -------
    verdicts_df : DataFrame indexed by date with columns:
        verdict, Anomaly_Prob, Surprise_Factor, Surprise_Factor_Z,
        Close_True, externality_score
    """
    print("\n" + "="*80)
    print("COMPONENT 1 -- CAUSAL VERDICT ENGINE")
    print("="*80)
    print(f"\n  SF_Z threshold : {sf_threshold}")
    print(f"  Rule           : SF_Z >= {sf_threshold} -> external_shock")
    print(f"                   SF_Z <  {sf_threshold} -> internal_mechanics")

    df = dual_results_df.copy()
    df.index = pd.to_datetime(df.index)

    # Pick best available anomaly column
    col = anomaly_col
    if col not in df.columns:
        col = "Anomaly_Pred" if "Anomaly_Pred" in df.columns else "Anomaly_True"
    if (df[col] == 1).sum() == 0:
        for fallback in ["Anomaly_True", "Anomaly_Pred"]:
            if fallback in df.columns and (df[fallback] == 1).sum() > 0:
                col = fallback
                break

    anomaly_days = df[df[col] == 1].copy()
    print(f"  Anomaly column : {col}")
    print(f"  Anomaly days   : {len(anomaly_days)}")

    if anomaly_days.empty:
        print("  [!] No anomaly days found -- returning empty verdicts.")
        return pd.DataFrame()

    sf_z = anomaly_days["Surprise_Factor_Z"].fillna(0)

    # Externality score: normalised to [0,1] -- 1 = most external
    sf_z_range = sf_z.max() - sf_z.min()
    externality = (sf_z - sf_z.min()) / max(sf_z_range, 1e-8)

    verdicts_df = pd.DataFrame(index=anomaly_days.index)
    verdicts_df["verdict"]           = np.where(sf_z >= sf_threshold,
                                                 "external_shock",
                                                 "internal_mechanics")
    verdicts_df["Anomaly_Prob"]      = anomaly_days["Anomaly_Prob"].values
    verdicts_df["Surprise_Factor"]   = anomaly_days["Surprise_Factor"].values
    verdicts_df["Surprise_Factor_Z"] = sf_z.values
    verdicts_df["Close_True"]        = anomaly_days["Close_True"].values
    verdicts_df["externality_score"] = externality.values

    ext   = (verdicts_df["verdict"] == "external_shock").sum()
    inte  = (verdicts_df["verdict"] == "internal_mechanics").sum()
    total = len(verdicts_df)

    print(f"\n  Verdict breakdown:")
    print(f"    external_shock     : {ext:>4}  ({ext/total:.1%})")
    print(f"    internal_mechanics : {inte:>4}  ({inte/total:.1%})")

    print("\n  Top 5 highest SF_Z anomalies (most likely external):")
    top5 = verdicts_df.nlargest(5, "Surprise_Factor_Z")[
        ["Surprise_Factor", "Surprise_Factor_Z", "Close_True", "verdict"]
    ]
    for date, row in top5.iterrows():
        print(f"    {str(date.date()):<12}  "
              f"SF={row['Surprise_Factor']:>7.2f}%  "
              f"SF_Z={row['Surprise_Factor_Z']:>6.3f}  "
              f"Close=${row['Close_True']:>10,.0f}  "
              f"[{row['verdict']}]")

    print("="*80)
    return verdicts_df


# ============================================================================
# COMPONENT 2 -- SURPRISE FACTOR DISTRIBUTION ANALYSIS
# ============================================================================

def analyze_sf_distribution(dual_results_df: pd.DataFrame,
                             verdicts_df: pd.DataFrame) -> dict:
    """
    Statistical analysis of Surprise Factor.
    Tests whether external anomalies have significantly higher SF
    than internal ones using Mann-Whitney U test.
    """
    print("\n" + "="*80)
    print("COMPONENT 2 -- SURPRISE FACTOR DISTRIBUTION ANALYSIS")
    print("="*80)

    stats_out = {}
    all_sf = dual_results_df["Surprise_Factor"].dropna()

    print(f"\n  All days ({len(all_sf)} total):")
    print(f"    mean={all_sf.mean():.2f}%  "
          f"median={all_sf.median():.2f}%  "
          f"std={all_sf.std():.2f}%  "
          f"max={all_sf.max():.2f}%")

    if verdicts_df.empty:
        print("  [!] No anomaly days to analyse.")
        print("="*80)
        return stats_out

    ext_sf  = verdicts_df.loc[verdicts_df["verdict"] == "external_shock",
                               "Surprise_Factor"].values
    int_sf  = verdicts_df.loc[verdicts_df["verdict"] == "internal_mechanics",
                               "Surprise_Factor"].values
    anom_sf = verdicts_df["Surprise_Factor"].values

    print(f"\n  Anomaly days ({len(anom_sf)} total):")
    print(f"    mean={anom_sf.mean():.2f}%  "
          f"median={np.median(anom_sf):.2f}%  "
          f"std={anom_sf.std():.2f}%")

    if len(ext_sf) > 0:
        print(f"\n  External shock (n={len(ext_sf)}):")
        print(f"    mean={ext_sf.mean():.2f}%  "
              f"median={np.median(ext_sf):.2f}%  "
              f"std={ext_sf.std():.2f}%")

    if len(int_sf) > 0:
        print(f"\n  Internal mechanics (n={len(int_sf)}):")
        print(f"    mean={int_sf.mean():.2f}%  "
              f"median={np.median(int_sf):.2f}%  "
              f"std={int_sf.std():.2f}%")

    if len(ext_sf) >= 3 and len(int_sf) >= 3:
        stat, p = stats.mannwhitneyu(ext_sf, int_sf, alternative="greater")
        sig = "SIGNIFICANT (p<0.05)" if p < 0.05 else "not significant"
        print(f"\n  Mann-Whitney U (external SF > internal SF):")
        print(f"    U={stat:.1f}  p={p:.4f}  [{sig}]")
        print(f"    Interpretation: "
              f"{'external shocks genuinely surprise the model more' if p < 0.05 else 'no significant difference in surprise magnitude'}")
        stats_out["mwu_stat"] = stat
        stats_out["mwu_p"]    = p

    if len(int_sf) > 0 and int_sf.mean() > 0 and len(ext_sf) > 0:
        ratio = ext_sf.mean() / int_sf.mean()
        print(f"\n  SF ratio (external / internal): {ratio:.2f}x")
        stats_out["sf_ratio"] = ratio

    stats_out.update({
        "ext_sf_mean":  float(ext_sf.mean())  if len(ext_sf)  > 0 else 0.0,
        "int_sf_mean":  float(int_sf.mean())  if len(int_sf)  > 0 else 0.0,
        "anom_sf_mean": float(anom_sf.mean()) if len(anom_sf) > 0 else 0.0,
        "all_sf_mean":  float(all_sf.mean()),
        "n_external":   int(len(ext_sf)),
        "n_internal":   int(len(int_sf)),
    })

    print("="*80)
    return stats_out


# ============================================================================
# COMPONENT 3 -- TEMPORAL PATTERN ANALYSIS
# ============================================================================

def analyze_temporal_patterns(verdicts_df: pd.DataFrame) -> pd.DataFrame:
    """
    Show how external vs internal anomalies distribute month by month.
    Useful for spotting periods of high external shock activity.
    """
    print("\n" + "="*80)
    print("COMPONENT 3 -- TEMPORAL PATTERN ANALYSIS")
    print("="*80)

    if verdicts_df.empty:
        print("  [!] No data to analyse.")
        print("="*80)
        return pd.DataFrame()

    df = verdicts_df.copy()
    df.index = pd.to_datetime(df.index)
    df["year_month"] = df.index.to_period("M")

    monthly = df.groupby(["year_month", "verdict"]).size().unstack(fill_value=0)
    for col in ["external_shock", "internal_mechanics"]:
        if col not in monthly.columns:
            monthly[col] = 0

    monthly["total"]   = monthly["external_shock"] + monthly["internal_mechanics"]
    monthly["ext_pct"] = (monthly["external_shock"] / monthly["total"] * 100).round(1)

    print(f"\n  {'Month':<12} {'External':>10} {'Internal':>10} "
          f"{'Total':>8} {'Ext%':>8}")
    print("  " + "-"*52)
    for period, row in monthly.iterrows():
        print(f"  {str(period):<12} {int(row['external_shock']):>10} "
              f"{int(row['internal_mechanics']):>10} "
              f"{int(row['total']):>8} "
              f"{row['ext_pct']:>7.1f}%")

    if len(monthly) > 0:
        peak = monthly["ext_pct"].idxmax()
        print(f"\n  Highest external shock period: {peak} "
              f"({monthly.loc[peak, 'ext_pct']:.1f}% external)")

    print("="*80)
    return monthly


# ============================================================================
# REPORT WRITER
# ============================================================================

def _write_summary_report(verdicts_df, statistics, sf_threshold, output_dir, symbol):
    n_ext  = statistics.get("n_external", 0)
    n_int  = statistics.get("n_internal", 0)
    total  = n_ext + n_int

    lines = [
        f"CAUSAL ANALYSIS REPORT -- CS402 Senior Project",
        f"Asset: {symbol}",
        "="*60,
        "",
        "METHOD",
        "-"*40,
        "  The dual-stream LSTM price regressor predicts next-day",
        "  closing price. The Surprise Factor measures how much the",
        "  actual price deviated from the model prediction.",
        f"  Anomalies with SF_Z >= {sf_threshold} -> external_shock",
        f"  Anomalies with SF_Z <  {sf_threshold} -> internal_mechanics",
        "",
        "RESULTS",
        "-"*40,
        f"  Total anomaly days   : {total}",
        f"  External shock       : {n_ext}  ({n_ext/total:.1%})" if total > 0 else "  External shock       : 0",
        f"  Internal mechanics   : {n_int}  ({n_int/total:.1%})" if total > 0 else "  Internal mechanics   : 0",
        "",
        "SURPRISE FACTOR",
        "-"*40,
        f"  All days mean SF     : {statistics.get('all_sf_mean', 0):.2f}%",
        f"  Anomaly days mean SF : {statistics.get('anom_sf_mean', 0):.2f}%",
        f"  External mean SF     : {statistics.get('ext_sf_mean', 0):.2f}%",
        f"  Internal mean SF     : {statistics.get('int_sf_mean', 0):.2f}%",
    ]

    if "sf_ratio" in statistics:
        lines.append(f"  SF ratio (ext/int)   : {statistics['sf_ratio']:.2f}x")
    if "mwu_p" in statistics:
        sig = "SIGNIFICANT" if statistics["mwu_p"] < 0.05 else "not significant"
        lines.append(f"  Mann-Whitney p       : {statistics['mwu_p']:.4f}  [{sig}]")

    lines += [
        "",
        "INTERPRETATION",
        "-"*40,
        "  High SF_Z on an anomaly day = the price regressor was",
        "  genuinely blindsided. Historical price patterns alone",
        "  could not anticipate the move. Consistent with external",
        "  events (news, macro, regulatory shocks).",
        "  Low SF_Z = unusual patterns were visible in price history",
        "  before the anomaly. Consistent with internal mechanics.",
        "",
        "NOTE",
        "-"*40,
        "  No external news API required.",
        f"  Fully generalizable -- this exact pipeline runs on any",
        f"  asset by changing the symbol in config.py.",
        "",
    ]

    path = os.path.join(output_dir, "causal_summary.txt")
    with open(path, "w") as f:
        f.write("\n".join(lines))


# ============================================================================
# FULL PIPELINE
# ============================================================================

def run_causal_analysis_pipeline(dual_results_df: pd.DataFrame,
                                  config: dict,
                                  sf_threshold: float = DEFAULT_SF_THRESHOLD,
                                  anomaly_col: str = "Anomaly_Pred") -> dict:
    """
    End-to-end Surprise Factor causal analysis. No API, no hardcoding.

    Parameters
    ----------
    dual_results_df : output DataFrame from run_dual_lstm_pipeline()
    config          : CONFIG dict
    sf_threshold    : SF_Z cutoff for external_shock (default 1.0)
    anomaly_col     : column to use as anomaly flag

    Returns
    -------
    dict with keys: verdicts_df, statistics, monthly_df
    """
    print("\n" + "="*80)
    print("CAUSAL ANALYSIS PIPELINE -- START")
    print("="*80)

    output_dir = config.get("output_dir", "results")
    symbol     = config.get("symbol", "BTC/USDT")
    os.makedirs(output_dir, exist_ok=True)

    print(f"  Asset         : {symbol}")
    print(f"  SF_Z threshold: {sf_threshold}")

    verdicts_df = assign_causal_verdicts(
        dual_results_df, sf_threshold=sf_threshold, anomaly_col=anomaly_col
    )
    if not verdicts_df.empty:
        verdicts_df.to_csv(f"{output_dir}/causal_verdicts.csv")
        print(f"\n  Saved -> {output_dir}/causal_verdicts.csv")

    statistics = analyze_sf_distribution(dual_results_df, verdicts_df)

    sf_dist = dual_results_df[["Close_True", "Surprise_Factor",
                                "Surprise_Factor_Z", "Anomaly_Prob"]].copy()
    sf_dist.to_csv(f"{output_dir}/causal_sf_distribution.csv")
    print(f"  Saved -> {output_dir}/causal_sf_distribution.csv")

    monthly_df = analyze_temporal_patterns(verdicts_df)
    if not monthly_df.empty:
        monthly_df.to_csv(f"{output_dir}/causal_monthly_pattern.csv")
        print(f"  Saved -> {output_dir}/causal_monthly_pattern.csv")

    _write_summary_report(verdicts_df, statistics, sf_threshold, output_dir, symbol)
    print(f"  Saved -> {output_dir}/causal_summary.txt")

    print("\nCausal analysis pipeline complete.")
    print("="*80)

    return {
        "verdicts_df": verdicts_df,
        "statistics":  statistics,
        "monthly_df":  monthly_df,
    }


# ============================================================================
# STANDALONE RUN
# ============================================================================

if __name__ == "__main__":
    from config import CONFIG
    import pandas as pd

    gt_csv  = f"{CONFIG['output_dir']}/dual_lstm_gt_results.csv"
    std_csv = f"{CONFIG['output_dir']}/dual_lstm_results.csv"
    csv_path = gt_csv if os.path.exists(gt_csv) else std_csv

    print(f"Loading: {csv_path}")
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)

    results = run_causal_analysis_pipeline(df, CONFIG)