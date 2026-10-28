"""Tests d'audit analytique au niveau de l'écriture comptable.

Chaque test renvoie l'ensemble des numéros d'écriture qu'il signale. Les poids
traduisent le niveau de risque (jugement d'auditeur) et alimentent le score.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from .calendar_fr import jours_feries
from .generate_fec import SEUIL_VALIDATION

SUPPLIER_ACCOUNT = "401"


def entries_view(fec: pd.DataFrame) -> pd.DataFrame:
    """Une ligne par écriture, avec ses attributs utiles à l'analyse."""
    g = fec.groupby("EcritureNum", sort=False)
    e = g.agg(JournalCode=("JournalCode", "first"), EcritureDate=("EcritureDate", "first"),
              ValidDate=("ValidDate", "max"), PieceRef=("PieceRef", "first"),
              EcritureLib=("EcritureLib", "first"), debit=("Debit", "sum"), credit=("Credit", "sum"),
              n_lignes=("Debit", "size"))
    sup = fec[fec["CompteNum"].str.startswith(SUPPLIER_ACCOUNT) & (fec["Credit"] > 0)]
    sup = sup.groupby("EcritureNum").agg(tiers=("CompAuxNum", "first"), montant_tiers=("Credit", "sum"))
    e = e.join(sup)
    e["montant"] = e[["debit", "credit"]].max(axis=1)
    e["tiers"] = e["tiers"].fillna("")
    return e.reset_index()


@dataclass(frozen=True)
class Control:
    code: str
    libelle: str
    poids: int
    fn: Callable[[pd.DataFrame], set]


def c_desequilibre(e):
    return set(e.loc[(e["debit"] - e["credit"]).abs() > 0.01, "EcritureNum"])


def c_doublon(e):
    a = e[(e["JournalCode"] == "ACH") & (e["tiers"] != "")].sort_values("EcritureDate")
    dup = a.duplicated(subset=["tiers", "PieceRef", "montant_tiers"], keep="first")
    return set(a.loc[dup, "EcritureNum"])


def c_jour_non_ouvre(e):
    years = e["EcritureDate"].dt.year.unique()
    feries = set().union(*[jours_feries(int(y)) for y in years])
    mask = (e["EcritureDate"].dt.dayofweek >= 5) | e["EcritureDate"].isin(feries)
    return set(e.loc[mask, "EcritureNum"])


def c_validation_tardive(e, jours: int = 60):
    return set(e.loc[(e["ValidDate"] - e["EcritureDate"]).dt.days > jours, "EcritureNum"])


def c_montant_rond_manuel(e):
    m = (e["JournalCode"] == "OD") & (e["montant"] >= 1_000) & (e["montant"] % 1_000 == 0)
    return set(e.loc[m, "EcritureNum"])


def c_fractionnement(e, seuil: float = SEUIL_VALIDATION, fenetre: int = 10):
    """≥ 2 factures d'un même fournisseur entre 85 % et 100 % du seuil de validation
    dans une fenêtre glissante de `fenetre` jours, dont le cumul dépasse le seuil."""
    a = e[(e["JournalCode"] == "ACH") & e["montant_tiers"].between(0.85 * seuil, seuil, inclusive="left")]
    flagged: set = set()
    for _, g in a.sort_values("EcritureDate").groupby("tiers"):
        dates, nums, amts = g["EcritureDate"].values, g["EcritureNum"].values, g["montant_tiers"].values
        for i in range(len(g)):
            win = (dates >= dates[i]) & (dates <= dates[i] + np.timedelta64(fenetre, "D"))
            if win.sum() >= 2 and amts[win].sum() > seuil:
                flagged.update(nums[win])
    return flagged


def c_montant_atypique(e, z: float = 3.5):
    """Z-score robuste (médiane / MAD) du log-montant au sein de chaque fournisseur."""
    a = e[(e["JournalCode"] == "ACH") & (e["tiers"] != "")].copy()
    a["lg"] = np.log(a["montant_tiers"])
    med = a.groupby("tiers")["lg"].transform("median")
    mad = a.groupby("tiers")["lg"].transform(lambda s: (s - s.median()).abs().median())
    a["z"] = 0.6745 * (a["lg"] - med) / mad.replace(0, np.nan)
    return set(a.loc[a["z"] > z, "EcritureNum"])


CONTROLS: list[Control] = [
    Control("DESEQUILIBRE", "Écriture déséquilibrée (débit ≠ crédit)", 5, c_desequilibre),
    Control("DOUBLON", "Facture fournisseur saisie deux fois", 5, c_doublon),
    Control("FRACTIONNEMENT", "Factures fractionnées sous le seuil de validation", 4, c_fractionnement),
    Control("MONTANT_ATYPIQUE", "Montant hors norme pour le fournisseur", 4, c_montant_atypique),
    Control("VALIDATION_TARDIVE", "Validée plus de 60 jours après sa date", 3, c_validation_tardive),
    Control("MONTANT_ROND_OD", "OD manuelle à montant rond (≥ 1 000 €)", 3, c_montant_rond_manuel),
    Control("JOUR_NON_OUVRE", "Passée un week-end ou un jour férié", 2, c_jour_non_ouvre),
]


def run_controls(e: pd.DataFrame) -> pd.DataFrame:
    """Matrice écriture × contrôle (booléens)."""
    out = pd.DataFrame({"EcritureNum": e["EcritureNum"]})
    for c in CONTROLS:
        out[c.code] = out["EcritureNum"].isin(c.fn(e))
    return out
