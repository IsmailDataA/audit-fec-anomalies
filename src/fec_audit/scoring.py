"""Score de risque par écriture : règles d'audit pondérées + détection non supervisée.

Les règles capturent les schémas connus ; l'Isolation Forest repère des écritures
« qui ne ressemblent pas aux autres » sans règle écrite à l'avance. Le score final
sert à constituer un échantillon de contrôle ciblé plutôt qu'aléatoire.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from .controls import CONTROLS


def features(e: pd.DataFrame) -> pd.DataFrame:
    f = pd.DataFrame(index=e.index)
    f["log_montant"] = np.log1p(e["montant"])
    f["jour_semaine"] = e["EcritureDate"].dt.dayofweek
    f["fin_de_mois"] = (e["EcritureDate"].dt.is_month_end).astype(int)
    f["delai_validation"] = (e["ValidDate"] - e["EcritureDate"]).dt.days
    f["centimes_nuls"] = ((e["montant"] * 100).round() % 100 == 0).astype(int)
    f["journal_manuel"] = e["JournalCode"].isin(["OD"]).astype(int)
    f["n_lignes"] = e["n_lignes"]
    # écart au comportement habituel du tiers
    lg = np.log1p(e["montant"])
    grp = e["tiers"].where(e["tiers"] != "", e["JournalCode"])
    f["ecart_tiers"] = (lg - lg.groupby(grp).transform("median")).abs()
    f["freq_tiers"] = grp.map(grp.value_counts()).pipe(np.log1p)
    return f.fillna(0)


def isolation_scores(e: pd.DataFrame, seed: int = 0) -> pd.Series:
    X = features(e)
    model = IsolationForest(n_estimators=300, contamination="auto", random_state=seed)
    model.fit(X)
    s = -model.score_samples(X)  # plus haut = plus anormal
    return pd.Series((s - s.min()) / (s.max() - s.min()), index=e.index)


def risk_score(e: pd.DataFrame, flags: pd.DataFrame, ml: pd.Series) -> pd.DataFrame:
    weights = {c.code: c.poids for c in CONTROLS}
    rules = sum(flags[c] * w for c, w in weights.items())
    out = e[["EcritureNum", "JournalCode", "EcritureDate", "tiers", "PieceRef",
             "EcritureLib", "montant"]].copy()
    out["score_regles"] = rules.values
    out["score_ml"] = ml.round(3).values
    out["score_risque"] = (out["score_regles"] + 3 * out["score_ml"]).round(2)
    out["motifs"] = flags[list(weights)].apply(
        lambda r: ", ".join(c for c in weights if r[c]), axis=1).values
    return out.sort_values("score_risque", ascending=False).reset_index(drop=True)
