"""
PHASE 2B — CryptoBERT Sentiment Analysis
=========================================
Uses the CryptoBERT model (ElKulako/cryptobert) from HuggingFace to classify
crypto news headlines as Bullish / Bearish / Neutral.

Data source: CryptoPanic API (free tier, no key required for public feed)
             Falls back to synthetic stub if API is unreachable.

Pipeline:
  1. fetch_crypto_news()      → raw headline DataFrame
  2. run_cryptobert()         → per-headline sentiment scores
  3. aggregate_daily_sentiment() → daily Bullish/Bearish/Neutral scores + net score
  4. merge_with_price_df()    → joined DataFrame for downstream analysis
  5. run_cryptobert_pipeline()→ end-to-end convenience wrapper

Output columns added to price DataFrame:
  sentiment_bullish   — fraction of headlines that day classified Bullish
  sentiment_bearish   — fraction classified Bearish
  sentiment_neutral   — fraction classified Neutral
  sentiment_net       — bullish - bearish  (range −1 … +1)
  headline_count      — number of headlines that day
"""

import os
import time
import warnings
import datetime
import requests
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ============================================================================
# MODEL LOADING
# ============================================================================

_pipeline = None   # lazy-loaded


def _load_model():
    """
    Load CryptoBERT pipeline (lazy, in-process cached).

    HuggingFace downloads the model once to ~/.cache/huggingface/hub/
    and reuses that local copy on every subsequent run — no re-download.
    The ~5-10s loading time per run is just reading weights from disk.
    """
    global _pipeline
    if _pipeline is not None:
        return _pipeline

    try:
        from huggingface_hub import try_to_load_from_cache  # type: ignore
        cached = try_to_load_from_cache("ElKulako/cryptobert", "config.json")
        is_cached = cached is not None
    except Exception:
        is_cached = False

    if is_cached:
        print("\n[CryptoBERT] Loading model from local cache…")
    else:
        print("\n[CryptoBERT] First run — downloading model (~400 MB, once only)…")

    try:
        from transformers import pipeline as hf_pipeline  # type: ignore
        _pipeline = hf_pipeline(
            "text-classification",
            model="ElKulako/cryptobert",
            tokenizer="ElKulako/cryptobert",
            truncation=True,
            max_length=512,
            device=-1,          # CPU; change to 0 for GPU
            top_k=None,         # return all labels
        )
        print("[CryptoBERT] Model ready.")
    except Exception as e:
        print(f"[CryptoBERT] WARNING: Could not load model — {e}")
        print("[CryptoBERT] Falling back to random-stub mode (for testing only).")
        _pipeline = "stub"

    return _pipeline


# ============================================================================
# DATA FETCHING — GDELT Doc 2.0 API (completely free, no API key required)
# ============================================================================
#
# GDELT monitors global news media and is fully free with no registration.
# Coverage: 2015 to present, updated every 15 minutes.
# Docs: https://blog.gdeltproject.org/gdelt-doc-2-0-api-debuts/
#
# Endpoint: https://api.gdeltproject.org/api/v2/doc/doc
# Max 250 results per request; paginate via startdatetime shifting.

GDELT_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
GDELT_QUERY = "bitcoin cryptocurrency BTC"


def _gdelt_dt(ts: pd.Timestamp) -> str:
    """Convert Timestamp to GDELT datetime format: YYYYMMDDHHMMSS"""
    return ts.strftime("%Y%m%d%H%M%S")


def fetch_crypto_news(
    start_date:   str,
    end_date:     str,
    max_per_day:  int = 10,   # articles to fetch per day-window request
    sleep_sec:    float = 1.0,
) -> pd.DataFrame:
    """
    Fetch BTC crypto news articles via GDELT Doc 2.0 API.

    Completely free — no API key, no registration required.
    Covers news from 2015 to present.

    Strategy: splits the date range into monthly windows and fires one
    GDELT request per month (250 articles max per request).

    Parameters
    ----------
    start_date  : "YYYY-MM-DD"
    end_date    : "YYYY-MM-DD"
    max_per_day : articles per monthly window (up to 250)
    sleep_sec   : seconds between requests

    Returns
    -------
    pd.DataFrame with columns: [date, title, published_at, kind, source]
    """
    start_dt = pd.Timestamp(start_date)
    end_dt   = pd.Timestamp(end_date)

    print(f"\n[GDELT] Fetching BTC news {start_date} → {end_date} (free, no key) …")

    # Build monthly windows to stay within GDELT's 250-result cap
    windows = []
    cur = start_dt
    while cur < end_dt:
        nxt = min(cur + pd.DateOffset(months=1), end_dt)
        windows.append((cur, nxt))
        cur = nxt

    records = []
    for i, (win_start, win_end) in enumerate(windows):
        params = {
            "query":         GDELT_QUERY,
            "mode":          "artlist",
            "maxrecords":    250,
            "startdatetime": _gdelt_dt(win_start),
            "enddatetime":   _gdelt_dt(win_end),
            "format":        "json",
            "sort":          "datedesc",
        }

        try:
            resp = requests.get(GDELT_URL, params=params, timeout=20)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"[GDELT] Request failed for {win_start.date()} window: {e}")
            time.sleep(sleep_sec)
            continue

        articles = data.get("articles", [])
        for art in articles:
            raw_date = art.get("seendate", "")  # format: YYYYMMDDTHHMMSSZ
            try:
                pub = pd.Timestamp(raw_date).tz_localize(None)
            except Exception:
                continue

            title = art.get("title", "").strip()
            if not title:
                continue

            records.append({
                "date":         pub.normalize(),
                "title":        title,
                "published_at": pub,
                "kind":         "news",
                "source":       art.get("domain", ""),
            })

        print(f"[GDELT]   {win_start.date()} → {win_end.date()} : {len(articles)} articles")
        time.sleep(sleep_sec)

    if records:
        df = pd.DataFrame(records).drop_duplicates(subset=["title"])
        print(f"[GDELT] Total unique articles: {len(df)} across {len(windows)} window(s).")
        return df

    print("[GDELT] No articles fetched — returning empty DataFrame.")
    return pd.DataFrame(columns=["date", "title", "published_at", "kind", "source"])


# ============================================================================
# SENTIMENT INFERENCE
# ============================================================================

LABEL_MAP = {
    # CryptoBERT label names  → canonical
    "Bullish":  "bullish",
    "Bearish":  "bearish",
    "Neutral":  "neutral",
    # Some HF versions use these
    "LABEL_0":  "bullish",
    "LABEL_1":  "bearish",
    "LABEL_2":  "neutral",
}


def _stub_sentiment(titles):
    """Random stub when real model unavailable (for testing)."""
    rng   = np.random.default_rng(42)
    labels = rng.choice(["bullish", "bearish", "neutral"], size=len(titles),
                        p=[0.4, 0.3, 0.3])
    scores = rng.uniform(0.5, 0.99, size=len(titles))
    return [{"label": l, "score": float(s)} for l, s in zip(labels, scores)]


def run_cryptobert(titles: list[str], batch_size: int = 32) -> list[dict]:
    """
    Run CryptoBERT on a list of headline strings.

    Returns list of dicts: [{"label": "bullish"|"bearish"|"neutral", "score": float}, …]
    """
    model = _load_model()

    if model == "stub":
        return _stub_sentiment(titles)

    results = []
    total = len(titles)
    print(f"[CryptoBERT] Classifying {total} headlines (batch_size={batch_size}) …")

    for start in range(0, total, batch_size):
        batch = titles[start: start + batch_size]
        preds = model(batch)           # list of lists (top_k=None → all labels)

        for pred_set in preds:
            # pred_set = [{"label": X, "score": Y}, …] — all labels
            best = max(pred_set, key=lambda x: x["score"])
            canonical = LABEL_MAP.get(best["label"], best["label"].lower())
            results.append({"label": canonical, "score": best["score"]})

        if (start + batch_size) % 200 == 0 or (start + batch_size) >= total:
            print(f"  … {min(start + batch_size, total)}/{total}")

    return results


# ============================================================================
# DAILY AGGREGATION
# ============================================================================

def aggregate_daily_sentiment(news_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate per-headline sentiment to daily scores.

    Input columns required: [date, sentiment_label, sentiment_score]

    Returns DataFrame indexed by date with columns:
      sentiment_bullish, sentiment_bearish, sentiment_neutral,
      sentiment_net, headline_count
    """
    if news_df.empty:
        return pd.DataFrame()

    # Fraction per sentiment label per day
    daily = (
        news_df.groupby(["date", "sentiment_label"])
               .size()
               .unstack(fill_value=0)
    )

    for col in ["bullish", "bearish", "neutral"]:
        if col not in daily.columns:
            daily[col] = 0

    daily["total"] = daily[["bullish", "bearish", "neutral"]].sum(axis=1)

    agg = pd.DataFrame(index=daily.index)
    agg["sentiment_bullish"] = daily["bullish"] / daily["total"]
    agg["sentiment_bearish"] = daily["bearish"] / daily["total"]
    agg["sentiment_neutral"]  = daily["neutral"] / daily["total"]
    agg["sentiment_net"]      = agg["sentiment_bullish"] - agg["sentiment_bearish"]
    agg["headline_count"]     = daily["total"].astype(int)

    agg.index = pd.to_datetime(agg.index)
    return agg


# ============================================================================
# MERGE WITH PRICE DATAFRAME
# ============================================================================

def merge_with_price_df(price_df: pd.DataFrame, sentiment_df: pd.DataFrame) -> pd.DataFrame:
    """
    Left-join daily sentiment onto price DataFrame.

    Days with no headlines get NaN sentiment (then filled to 0 / 0.33).
    """
    merged = price_df.copy()
    merged.index = pd.to_datetime(merged.index).normalize()

    sent = sentiment_df.copy()
    sent.index = pd.to_datetime(sent.index).normalize()

    merged = merged.join(sent, how="left")

    # Fill missing days
    for col in ["sentiment_bullish", "sentiment_bearish", "sentiment_neutral"]:
        merged[col] = merged[col].fillna(1/3)   # equal probability = no signal
    merged["sentiment_net"]    = merged["sentiment_net"].fillna(0.0)
    merged["headline_count"]   = merged["headline_count"].fillna(0).astype(int)

    return merged


# ============================================================================
# ANALYSIS & PRINTING
# ============================================================================

def print_sentiment_summary(merged_df: pd.DataFrame, anomaly_col: str = "Anomaly_Statistical"):
    """Print sentiment statistics, broken down by anomaly vs normal days."""

    print("\n" + "=" * 80)
    print("CRYPTOBERT SENTIMENT ANALYSIS — SUMMARY")
    print("=" * 80)

    req = ["sentiment_net", "sentiment_bullish", "sentiment_bearish", "headline_count"]
    if not all(c in merged_df.columns for c in req):
        print("  [!] Sentiment columns not found in DataFrame.")
        return

    total_headlines = merged_df["headline_count"].sum()
    days_with_news  = (merged_df["headline_count"] > 0).sum()

    print(f"\n  Total headlines processed : {total_headlines}")
    print(f"  Days with news coverage   : {days_with_news} / {len(merged_df)}")
    print(f"\n  Overall sentiment_net (mean) : {merged_df['sentiment_net'].mean():+.4f}")
    print(f"  Overall bullish fraction     : {merged_df['sentiment_bullish'].mean():.2%}")
    print(f"  Overall bearish fraction     : {merged_df['sentiment_bearish'].mean():.2%}")

    if anomaly_col in merged_df.columns:
        anom  = merged_df[merged_df[anomaly_col] == 1]
        norm  = merged_df[merged_df[anomaly_col] == 0]

        print(f"\n  {'':30s}  {'Anomaly Days':>14}  {'Normal Days':>12}")
        print("  " + "-" * 62)

        for label, subset in [("Anomaly", anom), ("Normal", norm)]:
            pass   # print below

        def fmt(s):
            return f"{s['sentiment_net'].mean():+.4f}"

        print(f"  {'sentiment_net (mean)':<30s}  {fmt(anom):>14}  {fmt(norm):>12}")
        print(f"  {'bullish fraction (mean)':<30s}  {anom['sentiment_bullish'].mean():>14.2%}  "
              f"{norm['sentiment_bullish'].mean():>12.2%}")
        print(f"  {'bearish fraction (mean)':<30s}  {anom['sentiment_bearish'].mean():>14.2%}  "
              f"{norm['sentiment_bearish'].mean():>12.2%}")
        print(f"  {'headline_count (mean)':<30s}  {anom['headline_count'].mean():>14.1f}  "
              f"{norm['headline_count'].mean():>12.1f}")

    print("=" * 80 + "\n")


# ============================================================================
# END-TO-END PIPELINE
# ============================================================================

def run_cryptobert_pipeline(
    price_df:    pd.DataFrame,
    config:      dict,
    anomaly_col: str = "Anomaly_Statistical",
) -> pd.DataFrame:
    """
    End-to-end CryptoBERT sentiment pipeline.

    Steps
    -----
    1. Determine date range from price_df index
    2. Fetch BTC news via GDELT (free, no API key)
    3. Run CryptoBERT inference on article titles
    4. Aggregate to daily scores
    5. Merge with price_df
    6. Print summary

    Parameters
    ----------
    price_df    : DataFrame with DatetimeIndex (from statistic.py)
    config      : CONFIG dict (for output_dir, gdelt_sleep_sec)
    anomaly_col : column for anomaly/normal breakdown in summary

    Returns
    -------
    merged_df : price_df enriched with sentiment columns
    """

    # 1. Date range
    idx       = pd.to_datetime(price_df.index).normalize()
    start_str = idx.min().strftime("%Y-%m-%d")
    end_str   = idx.max().strftime("%Y-%m-%d")

    # 2. Fetch news from GDELT (no key needed)
    news_df = fetch_crypto_news(
        start_date=start_str,
        end_date=end_str,
        sleep_sec=config.get("gdelt_sleep_sec", 1.0),
    )

    if news_df.empty:
        print("[CryptoBERT] No news data — sentiment columns filled with neutral defaults.")
        merged = price_df.copy()
        merged["sentiment_bullish"] = 1/3
        merged["sentiment_bearish"] = 1/3
        merged["sentiment_neutral"]  = 1/3
        merged["sentiment_net"]      = 0.0
        merged["headline_count"]     = 0
        return merged

    # 3. Inference
    titles = news_df["title"].tolist()
    preds  = run_cryptobert(titles, batch_size=config.get("cryptobert_batch_size", 32))

    news_df["sentiment_label"] = [p["label"] for p in preds]
    news_df["sentiment_score"] = [p["score"] for p in preds]

    # Save raw headlines + labels
    output_dir = config.get("output_dir", "results")
    os.makedirs(output_dir, exist_ok=True)
    news_df.to_csv(f"{output_dir}/cryptobert_headlines.csv", index=False)
    print(f"[CryptoBERT] Headlines + labels saved → {output_dir}/cryptobert_headlines.csv")

    # 4. Aggregate
    daily_sent = aggregate_daily_sentiment(news_df)

    # Save daily scores
    daily_sent.to_csv(f"{output_dir}/cryptobert_daily_sentiment.csv")
    print(f"[CryptoBERT] Daily sentiment saved → {output_dir}/cryptobert_daily_sentiment.csv")

    # 5. Merge
    merged = merge_with_price_df(price_df, daily_sent)

    # 6. Summary
    print_sentiment_summary(merged, anomaly_col=anomaly_col)

    return merged


# ============================================================================
# STANDALONE RUN
# ============================================================================

if __name__ == "__main__":
    from statistic import fetch_cryptocurrency_data, data_preprocessing_and_feature_engineering
    from config import CONFIG

    df_raw = fetch_cryptocurrency_data()
    df, split_idx = data_preprocessing_and_feature_engineering(
        df_raw, train_ratio=CONFIG["train_ratio"], create_labels=True
    )

    merged = run_cryptobert_pipeline(df, CONFIG)

    print("\nSample of merged DataFrame (last 10 rows):")
    cols = ["Close", "sentiment_net", "sentiment_bullish", "sentiment_bearish",
            "headline_count", "Anomaly_Statistical"]
    print(merged[cols].tail(10))
