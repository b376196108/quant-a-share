from __future__ import annotations

import datetime as dt
import hashlib
import os
import re
import threading
from typing import Any, Dict, List, Tuple
from urllib.parse import urlparse

import requests


GDELT_DOC_ENDPOINT = "https://api.gdeltproject.org/api/v2/doc/doc"

# Default to a small set of mainstream Chinese finance domains known to appear in GDELT.
DEFAULT_GDELT_QUERY = os.getenv(
    "NEWS_GDELT_QUERY",
    "(domain:finance.ifeng.com OR domain:fund.eastmoney.com OR domain:finance.sina.com.cn)",
)

DEFAULT_LOOKBACK_HOURS = int(os.getenv("NEWS_LOOKBACK_HOURS", "24"))
DEFAULT_LIMIT = int(os.getenv("NEWS_DEFAULT_LIMIT", "30"))
CACHE_TTL_SECONDS = int(os.getenv("NEWS_CACHE_TTL_SECONDS", "300"))
REQUEST_TIMEOUT_SECONDS = float(os.getenv("NEWS_REQUEST_TIMEOUT_SECONDS", "8"))


_cache_lock = threading.Lock()
_cache: dict[tuple[str, int, int], tuple[float, dict[str, Any]]] = {}


def _now_utc() -> dt.datetime:
    return dt.datetime.now(tz=dt.timezone.utc)


def _format_gdelt_datetime(ts: dt.datetime) -> str:
    return ts.strftime("%Y%m%d%H%M%S")


def _parse_gdelt_seendate(seendate: str | None) -> dt.datetime:
    if not seendate:
        return _now_utc()
    try:
        # example: 20251213T073000Z
        parsed = dt.datetime.strptime(seendate, "%Y%m%dT%H%M%SZ")
        return parsed.replace(tzinfo=dt.timezone.utc)
    except Exception:
        return _now_utc()


def _format_cn_time(seen_utc: dt.datetime) -> str:
    # Keep it deterministic without relying on tzdata availability.
    cn = seen_utc.astimezone(dt.timezone(dt.timedelta(hours=8)))
    return cn.strftime("%H:%M")


_BULLISH_PAT = re.compile(
    r"(利好|上涨|大涨|走强|反弹|突破|新高|增持|回购|支持|改革|降息|宽松|提振|重估|增长)",
    re.IGNORECASE,
)
_BEARISH_PAT = re.compile(
    r"(利空|下跌|大跌|走弱|回调|暴跌|亏损|预警|下调|收紧|加息|风险|净流出|承压|爆雷)",
    re.IGNORECASE,
)


def classify_sentiment(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return "Neutral"
    if _BULLISH_PAT.search(text):
        return "Bullish"
    if _BEARISH_PAT.search(text):
        return "Bearish"
    return "Neutral"


_TAG_RULES: list[tuple[re.Pattern[str], list[str]]] = [
    (re.compile(r"(央行|PBOC|流动性|降准|降息|逆回购|LPR)", re.IGNORECASE), ["宏观", "货币政策"]),
    (re.compile(r"(证监会|CSRC|注册制|IPO|再融资)", re.IGNORECASE), ["政策", "监管"]),
    (re.compile(r"(美联储|FED|加息|降息|通胀)", re.IGNORECASE), ["海外", "利率"]),
    (re.compile(r"(人民币|汇率|外汇)", re.IGNORECASE), ["外汇", "人民币"]),
    (re.compile(r"(北向资金|外资|沪深股通)", re.IGNORECASE), ["资金流向"]),
    (re.compile(r"(白酒|茅台|五粮液)", re.IGNORECASE), ["白酒"]),
    (re.compile(r"(新能源|光伏|风电|锂电|电池)", re.IGNORECASE), ["新能源"]),
    (re.compile(r"(半导体|芯片|先进制程)", re.IGNORECASE), ["半导体"]),
    (re.compile(r"(\bAI\b|人工智能|大模型|算力)", re.IGNORECASE), ["AI"]),
    (re.compile(r"(房地产|房企|地产|楼市)", re.IGNORECASE), ["房地产"]),
    (re.compile(r"(银行|券商|保险)", re.IGNORECASE), ["金融"]),
    (re.compile(r"(汽车|特斯拉|电动车|智驾)", re.IGNORECASE), ["汽车"]),
]


def extract_tags(title: str) -> list[str]:
    text = (title or "").strip()
    tags: list[str] = []
    for pat, values in _TAG_RULES:
        if pat.search(text):
            tags.extend(values)
    # de-dup while preserving order
    deduped: list[str] = []
    for t in tags:
        if t not in deduped:
            deduped.append(t)
    return deduped[:6] if deduped else ["综合"]


def gdelt_article_to_news_item(article: dict[str, Any]) -> dict[str, Any]:
    url = (article.get("url") or "").strip()
    title = (article.get("title") or "").strip()
    domain = (article.get("domain") or "").strip()
    if not domain and url:
        try:
            domain = urlparse(url).netloc
        except Exception:
            domain = ""

    seendate = _parse_gdelt_seendate(article.get("seendate"))
    time_str = _format_cn_time(seendate)

    if url:
        item_id = hashlib.sha1(url.encode("utf-8")).hexdigest()[:12]
    else:
        seed = f"{title}|{domain}|{time_str}"
        item_id = hashlib.sha1(seed.encode("utf-8")).hexdigest()[:12]

    sentiment = classify_sentiment(title)
    tags = extract_tags(title)

    # GDELT ArtList does not provide a content snippet; keep a short, stable placeholder.
    summary = title if len(title) <= 120 else f"{title[:117]}..."

    return {
        "id": item_id,
        "title": title or "(untitled)",
        "summary": summary,
        "source": domain or "GDELT",
        "time": time_str,
        "sentiment": sentiment,
        "tags": tags,
        "url": url or None,
    }


def build_trending(items: list[dict[str, Any]], limit: int = 6) -> list[str]:
    counts: dict[str, int] = {}
    for item in items:
        for tag in item.get("tags") or []:
            if not tag:
                continue
            counts[tag] = counts.get(tag, 0) + 1

    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return [f"#{name}" for name, _ in ranked[:limit]]


def _fetch_gdelt_articles(query: str, limit: int, lookback_hours: int) -> list[dict[str, Any]]:
    end = _now_utc()
    start = end - dt.timedelta(hours=max(int(lookback_hours), 1))

    params = {
        "query": query,
        "mode": "ArtList",
        "format": "json",
        "sort": "datedesc",
        "maxrecords": int(limit),
        "startdatetime": _format_gdelt_datetime(start),
        "enddatetime": _format_gdelt_datetime(end),
    }

    resp = requests.get(GDELT_DOC_ENDPOINT, params=params, timeout=REQUEST_TIMEOUT_SECONDS)
    resp.raise_for_status()
    data = resp.json()
    articles = data.get("articles") or []
    if not isinstance(articles, list):
        return []
    return articles


def get_news(
    q: str | None = None,
    limit: int | None = None,
    lookback_hours: int | None = None,
) -> Tuple[list[dict[str, Any]], list[str]]:
    """
    Returns (items, trending).

    - q: optional override query (GDELT syntax).
    - limit: max number of items.
    - lookback_hours: time window for fetching articles.
    """
    query = (q or DEFAULT_GDELT_QUERY).strip()
    limit_v = int(limit if limit is not None else DEFAULT_LIMIT)
    limit_v = max(1, min(limit_v, 100))
    lookback_v = int(lookback_hours if lookback_hours is not None else DEFAULT_LOOKBACK_HOURS)
    lookback_v = max(1, min(lookback_v, 24 * 14))

    cache_key = (query, limit_v, lookback_v)
    now_ts = dt.datetime.now().timestamp()

    with _cache_lock:
        cached = _cache.get(cache_key)
        if cached and (now_ts - cached[0] <= CACHE_TTL_SECONDS):
            data = cached[1]
            return data["items"], data["trending"]

    try:
        articles = _fetch_gdelt_articles(query=query, limit=limit_v, lookback_hours=lookback_v)
    except requests.exceptions.RequestException:
        # Upstream can rate-limit; serve stale cache if available.
        with _cache_lock:
            cached = _cache.get(cache_key)
            if cached:
                data = cached[1]
                return data["items"], data["trending"]
        raise

    seen_urls: set[str] = set()
    items: list[dict[str, Any]] = []
    for art in articles:
        if not isinstance(art, dict):
            continue
        url = (art.get("url") or "").strip()
        if url and url in seen_urls:
            continue
        if url:
            seen_urls.add(url)
        item = gdelt_article_to_news_item(art)
        items.append(item)

    trending = build_trending(items)

    with _cache_lock:
        _cache[cache_key] = (now_ts, {"items": items, "trending": trending})

    return items, trending
