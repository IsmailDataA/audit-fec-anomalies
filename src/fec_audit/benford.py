"""Loi de Benford (premier chiffre) et indicateur de conformité MAD de Nigrini."""
from __future__ import annotations

import numpy as np
import pandas as pd

BENFORD = pd.Series(np.log10(1 + 1 / np.arange(1, 10)), index=range(1, 10))

# Seuils MAD pour le premier chiffre (Nigrini, 2012)
MAD_SEUILS = [(0.006, "Conformité forte"), (0.012, "Conformité acceptable"),
              (0.015, "Conformité marginale"), (np.inf, "Non conforme")]


def first_digit(amounts: pd.Series) -> pd.Series:
    a = amounts[amounts >= 10].abs()
    return (a / 10 ** np.floor(np.log10(a))).astype(int)


def distribution(amounts: pd.Series) -> pd.Series:
    d = first_digit(amounts).value_counts(normalize=True)
    return d.reindex(range(1, 10), fill_value=0.0)


def mad(amounts: pd.Series) -> float:
    return float((distribution(amounts) - BENFORD).abs().mean())


def verdict(m: float) -> str:
    return next(lbl for s, lbl in MAD_SEUILS if m <= s)


def by_account(df: pd.DataFrame, key: str = "CompAuxNum", min_n: int = 60) -> pd.DataFrame:
    """MAD par tiers (fournisseur) disposant d'assez d'observations."""
    rows = []
    for k, g in df.groupby(key):
        if not k or len(g) < min_n:
            continue
        m = mad(g["montant"])
        rows.append({key: k, "n": len(g), "mad": round(m, 4), "verdict": verdict(m)})
    return pd.DataFrame(rows).sort_values("mad", ascending=False)
