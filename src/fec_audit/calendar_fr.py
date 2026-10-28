"""Jours fériés français (métropole), calculés — y compris ceux liés à Pâques."""
from __future__ import annotations

from datetime import date, timedelta

import pandas as pd


def _paques(year: int) -> date:
    """Algorithme de Meeus/Jones/Butcher (calendrier grégorien)."""
    a, b, c = year % 19, year // 100, year % 100
    d, e = b // 4, b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = c // 4, c % 4
    l_ = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l_) // 451
    month = (h + l_ - 7 * m + 114) // 31
    day = ((h + l_ - 7 * m + 114) % 31) + 1
    return date(year, month, day)


def jours_feries(year: int) -> set[pd.Timestamp]:
    p = _paques(year)
    fixes = [(1, 1), (5, 1), (5, 8), (7, 14), (8, 15), (11, 1), (11, 11), (12, 25)]
    mobiles = [p + timedelta(days=1), p + timedelta(days=39), p + timedelta(days=50)]
    return {pd.Timestamp(year, m, d) for m, d in fixes} | {pd.Timestamp(x) for x in mobiles}


def is_business_day(d: pd.Timestamp, feries: set[pd.Timestamp] | None = None) -> bool:
    feries = feries if feries is not None else jours_feries(d.year)
    return d.dayofweek < 5 and pd.Timestamp(d.date()) not in feries
