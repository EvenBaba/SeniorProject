"""
PHASE 2B — CryptoBERT Sentiment Analysis
=========================================
Uses the CryptoBERT model (ElKulako/cryptobert) from HuggingFace to classify
crypto news headlines as Bullish / Bearish / Neutral.

Data source (priority order):
  1. free-crypto-news.vercel.app — free, no API key, archive + live BTC feed
                                   archive updated every 6h, live feed real-time
  2. CoinDesk RSS               — free, no key, recent articles only (fallback)

Pipeline:
  1. fetch_crypto_news()         → raw headline DataFrame
  2. _fetch_from_cryptocurrencycv() → primary source
  3. _fetch_from_coindesk_rss()     → fallback source
  4. run_cryptobert()            → per-headline sentiment scores
  5. aggregate_daily_sentiment() → daily Bullish/Bearish/Neutral scores + net score
  6. merge_with_price_df()       → joined DataFrame for downstream analysis
  7. run_cryptobert_pipeline()   → end-to-end convenience wrapper

Output columns added to price DataFrame:
  sentiment_bullish   — fraction of headlines that day classified Bullish
  sentiment_bearish   — fraction classified Bearish
  sentiment_neutral   — fraction classified Neutral
  sentiment_net       — bullish - bearish  (range -1 … +1)
  headline_count      — number of headlines that day
"""

import os
import time
import warnings
import requests
import xml.etree.ElementTree as ET
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
# DATA FETCHING — PRIMARY: free-crypto-news (Vercel)
# ============================================================================
#
# Open-source, completely free, no API key required.
# Base URL: https://free-crypto-news.vercel.app
# GitHub:   https://github.com/nirholas/free-crypto-news
#
# Two endpoints used:
#   /api/archive  — historical news, queryable by start_date / end_date
#   /api/bitcoin  — live Bitcoin-specific feed (recent articles)
#
# The archive is updated every 6 hours via GitHub Actions.
# The /api/bitcoin feed is updated in real-time.

FCN_BASE_URL    = "https://free-crypto-news.vercel.app"
FCN_ARCHIVE_URL = f"{FCN_BASE_URL}/api/archive"
FCN_BITCOIN_URL = f"{FCN_BASE_URL}/api/bitcoin"


def _fetch_from_cryptocurrencycv(
    start_date: str,
    end_date:   str,
    sleep_sec:  float = 0.5,
) -> pd.DataFrame:
    """
    Fetch BTC news from free-crypto-news.vercel.app.

    Strategy:
      1. Try /api/archive with start_date / end_date for historical data,
         paginating month by month (50 articles per window).
      2. Also call /api/bitcoin (live feed) and merge to catch recent
         articles not yet stored in the archive.

    Returns a DataFrame with columns: [date, title, published_at, kind, source]

    Parameters
    ----------
    start_date : "YYYY-MM-DD"
    end_date   : "YYYY-MM-DD"
    sleep_sec  : polite delay between requests (default 0.5s)
    """
    start_dt = pd.Timestamp(start_date)
    end_dt   = pd.Timestamp(end_date)

    print(f"\n[free-crypto-news] Fetching BTC articles {start_date} -> {end_date} …")

    records = []

    # ------------------------------------------------------------------
    # 1. Historical archive — paginate month by month
    # ------------------------------------------------------------------
    months_start = pd.date_range(start_dt, end_dt, freq="MS")
    windows = []
    for ms in months_start:
        me = min(ms + pd.DateOffset(months=1) - pd.Timedelta(days=1), end_dt)
        windows.append((ms, me))
    if len(windows) == 0:
        windows = [(start_dt, end_dt)]

    for win_start, win_end in windows:
        params = {
            "start_date": win_start.strftime("%Y-%m-%d"),
            "end_date":   win_end.strftime("%Y-%m-%d"),
            "q":          "bitcoin BTC",
            "limit":      50,
        }
        try:
            resp = requests.get(FCN_ARCHIVE_URL, params=params, timeout=20)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"[free-crypto-news]   archive {win_start.date()} -> "
                  f"{win_end.date()} failed: {e}")
            time.sleep(sleep_sec)
            continue

        articles = data.get("articles", [])
        for art in articles:
            raw_date = art.get("pubDate") or art.get("published_at", "")
            try:
                pub = pd.Timestamp(raw_date).tz_localize(None)
            except Exception:
                continue
            if pub.normalize() < start_dt or pub.normalize() > end_dt:
                continue
            title = art.get("title", "").strip()
            if not title:
                continue
            records.append({
                "date":         pub.normalize(),
                "title":        title,
                "published_at": pub,
                "kind":         "news",
                "source":       art.get("source", ""),
            })

        print(f"[free-crypto-news]   archive {win_start.date()} -> "
              f"{win_end.date()}: {len(articles)} articles")
        time.sleep(sleep_sec)

    # ------------------------------------------------------------------
    # 2. Live /api/bitcoin feed — catches articles not yet archived
    # ------------------------------------------------------------------
    try:
        resp = requests.get(FCN_BITCOIN_URL, params={"limit": 50}, timeout=20)
        resp.raise_for_status()
        data = resp.json()
        live_articles = data.get("articles", [])
        added = 0
        for art in live_articles:
            raw_date = art.get("pubDate") or art.get("published_at", "")
            try:
                pub = pd.Timestamp(raw_date).tz_localize(None)
            except Exception:
                continue
            if pub.normalize() < start_dt or pub.normalize() > end_dt:
                continue
            title = art.get("title", "").strip()
            if not title:
                continue
            records.append({
                "date":         pub.normalize(),
                "title":        title,
                "published_at": pub,
                "kind":         "news",
                "source":       art.get("source", ""),
            })
            added += 1
        print(f"[free-crypto-news]   live feed: {added} additional articles in range")
    except Exception as e:
        print(f"[free-crypto-news]   live feed failed: {e}")

    if records:
        df = pd.DataFrame(records).drop_duplicates(subset=["title"])
        print(f"[free-crypto-news] Total unique articles: {len(df)}")
        return df

    print("[free-crypto-news] No articles returned.")
    return pd.DataFrame(columns=["date", "title", "published_at", "kind", "source"])


# ============================================================================
# DATA FETCHING — FALLBACK: CoinDesk RSS
# ============================================================================
#
# Public RSS feed from CoinDesk — no API key, no registration.
# Covers recent articles only (typically last 30–60 days).
# Used as a fallback when free-crypto-news.vercel.app is unreachable.
# Feed URL: https://www.coindesk.com/arc/outboundfeeds/rss/

COINDESK_RSS_URL = "https://www.coindesk.com/arc/outboundfeeds/rss/"


def _fetch_from_coindesk_rss(
    start_date: str,
    end_date:   str,
) -> pd.DataFrame:
    """
    Fetch recent BTC news from the CoinDesk public RSS feed.

    No API key required. Returns recent articles only — typically the
    last 30–60 days. Used as a fallback when free-crypto-news.vercel.app fails.

    Parameters
    ----------
    start_date : "YYYY-MM-DD"  — articles before this date are dropped
    end_date   : "YYYY-MM-DD"  — articles after this date are dropped
    """
    start_dt = pd.Timestamp(start_date)
    end_dt   = pd.Timestamp(end_date)

    print(f"\n[CoinDesk RSS] Fetching feed (fallback, recent articles only) …")

    try:
        resp = requests.get(COINDESK_RSS_URL, timeout=20,
                            headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
    except Exception as e:
        print(f"[CoinDesk RSS] Request failed: {e}")
        return pd.DataFrame(columns=["date", "title", "published_at", "kind", "source"])

    try:
        root = ET.fromstring(resp.content)
    except ET.ParseError as e:
        print(f"[CoinDesk RSS] XML parse error: {e}")
        return pd.DataFrame(columns=["date", "title", "published_at", "kind", "source"])

    # RSS structure: <rss><channel><item>…</item></channel></rss>
    items = root.findall(".//item")
    records = []

    for item in items:
        title_el = item.find("title")
        pubdate_el = item.find("pubDate")

        if title_el is None or pubdate_el is None:
            continue

        title = (title_el.text or "").strip()
        if not title:
            continue

        try:
            pub = pd.Timestamp(pubdate_el.text).tz_localize(None)
        except Exception:
            continue

        # Filter to requested date range
        if pub.normalize() < start_dt or pub.normalize() > end_dt:
            continue

        records.append({
            "date":         pub.normalize(),
            "title":        title,
            "published_at": pub,
            "kind":         "news",
            "source":       "coindesk.com",
        })

    if records:
        df = pd.DataFrame(records).drop_duplicates(subset=["title"])
        print(f"[CoinDesk RSS] {len(df)} articles in range.")
        return df

    print("[CoinDesk RSS] No articles found in the requested date range.")
    return pd.DataFrame(columns=["date", "title", "published_at", "kind", "source"])


# ============================================================================
# PUBLIC FETCH FUNCTION (primary + fallback combined)
# ============================================================================

def fetch_crypto_news(
    start_date: str,
    end_date:   str,
    sleep_sec:  float = 0.5,
) -> pd.DataFrame:
    """
    Fetch BTC crypto news headlines for CryptoBERT inference.

    Tries free-crypto-news.vercel.app first (primary — archive + live BTC feed,
    no key required). Falls back to CoinDesk RSS if the primary is unreachable.

    Parameters
    ----------
    start_date : "YYYY-MM-DD"
    end_date   : "YYYY-MM-DD"
    sleep_sec  : delay between requests to free-crypto-news.vercel.app (default 0.5s)

    Returns
    -------
    pd.DataFrame with columns: [date, title, published_at, kind, source]
    Rows are deduplicated by title. Empty DataFrame if both sources fail.
    """
    # --- Primary: free-crypto-news.vercel.app ---
    df = _fetch_from_cryptocurrencycv(start_date, end_date, sleep_sec=sleep_sec)

    if not df.empty:
        return df

    # --- Fallback: CoinDesk RSS ---
    print("\n[fetch_crypto_news] free-crypto-news returned nothing — "
          "trying CoinDesk RSS fallback …")
    df = _fetch_from_coindesk_rss(start_date, end_date)

    if not df.empty:
        return df

    print("[fetch_crypto_news] Both sources failed. "
          "Returning empty DataFrame — sentiment will be neutral-filled.")
    return pd.DataFrame(columns=["date", "title", "published_at", "kind", "source"])


# ============================================================================
# SENTIMENT INFERENCE
# ============================================================================

LABEL_MAP = {
    # CryptoBERT label names  -> canonical
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


def run_cryptobert(titles: list, batch_size: int = 32) -> list:
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
        preds = model(batch)           # list of lists (top_k=None -> all labels)

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
    2. Fetch BTC news (free-crypto-news.vercel.app -> CoinDesk RSS fallback)
    3. Run CryptoBERT inference on article titles
    4. Aggregate to daily scores
    5. Merge with price_df
    6. Print summary

    Parameters
    ----------
    price_df    : DataFrame with DatetimeIndex (from statistic.py)
    config      : CONFIG dict (for output_dir, news_sleep_sec, cryptobert_batch_size)
    anomaly_col : column for anomaly/normal breakdown in summary

    Returns
    -------
    merged_df : price_df enriched with sentiment columns
    """

    # 1. Date range
    idx       = pd.to_datetime(price_df.index).normalize()
    start_str = idx.min().strftime("%Y-%m-%d")
    end_str   = idx.max().strftime("%Y-%m-%d")

    # 2. Fetch news
    news_df = fetch_crypto_news(
        start_date=start_str,
        end_date=end_str,
        sleep_sec=config.get("news_sleep_sec", 0.5),
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
    print(f"[CryptoBERT] Headlines + labels saved -> {output_dir}/cryptobert_headlines.csv")

    # 4. Aggregate
    daily_sent = aggregate_daily_sentiment(news_df)

    # Save daily scores
    daily_sent.to_csv(f"{output_dir}/cryptobert_daily_sentiment.csv")
    print(f"[CryptoBERT] Daily sentiment saved -> {output_dir}/cryptobert_daily_sentiment.csv")

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