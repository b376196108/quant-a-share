from __future__ import annotations

import datetime as dt

import requests
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from .news_service import get_news

router = APIRouter()


class NewsItemOut(BaseModel):
    id: str
    title: str
    summary: str
    source: str
    time: str
    sentiment: str
    tags: list[str]
    url: str | None = None


class NewsResponse(BaseModel):
    source: str
    generated_at: str
    items: list[NewsItemOut]
    trending: list[str]


@router.get("/api/news", response_model=NewsResponse)
def list_news(
    q: str | None = Query(default=None, description="GDELT query syntax override"),
    limit: int = Query(default=30, ge=1, le=100),
    lookback_hours: int = Query(default=24, ge=1, le=24 * 14),
) -> NewsResponse:
    """
    Free news aggregation endpoint (default: GDELT Doc API).
    """
    try:
        items, trending = get_news(q=q, limit=limit, lookback_hours=lookback_hours)
    except requests.exceptions.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"news upstream error: {exc}")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return NewsResponse(
        source="gdelt",
        generated_at=dt.datetime.utcnow().isoformat(),
        items=[NewsItemOut(**item) for item in items],
        trending=trending,
    )
