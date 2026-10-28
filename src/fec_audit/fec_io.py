"""Lecture d'un FEC réel ou généré + contrôles de conformité du format (DGFiP)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .generate_fec import FEC_COLUMNS

OBLIGATOIRES = ["JournalCode", "JournalLib", "EcritureNum", "EcritureDate", "CompteNum",
                "CompteLib", "PieceRef", "PieceDate", "EcritureLib", "ValidDate"]


def read_fec(path: str | Path) -> pd.DataFrame:
    """Lit un FEC (tabulation ou barre verticale, décimales virgule ou point)."""
    path = Path(path)
    head = path.read_text(encoding="utf-8", errors="replace").splitlines()[0]
    sep = "|" if head.count("|") > head.count("\t") else "\t"
    df = pd.read_csv(path, sep=sep, dtype=str, keep_default_na=False, encoding="utf-8")
    df.columns = [c.strip() for c in df.columns]
    for c in ("Debit", "Credit"):
        df[c] = pd.to_numeric(df[c].str.replace(" ", "").str.replace(",", "."), errors="coerce").fillna(0.0)
    for c in ("EcritureDate", "PieceDate", "ValidDate"):
        df[c] = pd.to_datetime(df[c], format="%Y%m%d", errors="coerce")
    return df


def conformity_checks(df: pd.DataFrame) -> pd.DataFrame:
    """Contrôles structurels attendus par l'administration fiscale."""
    res = []

    def add(ctrl: str, ok: bool, detail: str) -> None:
        res.append({"controle": ctrl, "statut": "OK" if ok else "KO", "detail": detail})

    missing = [c for c in FEC_COLUMNS if c not in df.columns]
    add("18 colonnes réglementaires présentes", not missing,
        "Toutes présentes" if not missing else f"Manquantes : {', '.join(missing)}")
    for c in OBLIGATOIRES:
        if c in df.columns:
            n = int((df[c].astype(str).str.strip().isin(["", "NaT", "nan"]) | df[c].isna()).sum())
            add(f"Champ {c} renseigné", n == 0, f"{n} ligne(s) vide(s)")
    bal = df.groupby("EcritureNum")[["Debit", "Credit"]].sum()
    n_unbal = int(((bal["Debit"] - bal["Credit"]).abs() > 0.01).sum())
    add("Chaque écriture est équilibrée", n_unbal == 0, f"{n_unbal} écriture(s) déséquilibrée(s)")
    gap = round(float(df["Debit"].sum() - df["Credit"].sum()), 2)
    add("Balance générale équilibrée", abs(gap) < 0.01, f"Écart total : {gap:,.2f} €".replace(",", " "))
    chrono_ko = 0
    for _, g in df.drop_duplicates("EcritureNum").groupby("JournalCode"):
        g = g.sort_values("EcritureNum")
        chrono_ko += int((g["EcritureDate"].diff().dt.days < 0).sum())
    add("Numérotation chronologique par journal", chrono_ko == 0, f"{chrono_ko} rupture(s)")
    late = int((df["ValidDate"] < df["EcritureDate"]).sum())
    add("Date de validation ≥ date d'écriture", late == 0, f"{late} ligne(s) incohérente(s)")
    return pd.DataFrame(res)
