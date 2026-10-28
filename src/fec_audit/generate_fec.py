"""Génère un FEC (Fichier des Écritures Comptables) fictif mais réaliste pour une
PME sur un exercice complet, avec des anomalies injectées et tracées.

Le fichier respecte le format DGFiP (18 colonnes, séparateur tabulation,
décimales à la virgule) : il se lit exactement comme un FEC réel.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .calendar_fr import is_business_day, jours_feries

FEC_COLUMNS = [
    "JournalCode", "JournalLib", "EcritureNum", "EcritureDate", "CompteNum", "CompteLib",
    "CompAuxNum", "CompAuxLib", "PieceRef", "PieceDate", "EcritureLib", "Debit", "Credit",
    "EcritureLet", "DateLet", "ValidDate", "Montantdevise", "Idevise",
]
JOURNAUX = {"ACH": "Achats", "VTE": "Ventes", "BQ": "Banque", "OD": "Opérations diverses",
            "PAIE": "Paie"}
CHARGES = {"601000": "Achats matières premières", "606100": "Fournitures non stockables",
           "606400": "Fournitures administratives", "613200": "Locations immobilières",
           "615000": "Entretien et réparations", "622600": "Honoraires",
           "626000": "Frais postaux et télécom", "624100": "Transports sur achats"}
SEUIL_VALIDATION = 5_000.0  # au-delà, une facture fournisseur exige une double validation


@dataclass
class FecDataset:
    fec: pd.DataFrame
    truth: pd.DataFrame  # EcritureNum, anomaly_type


class _Builder:
    def __init__(self, rng: np.random.Generator):
        self.rng = rng
        self.lines: list[dict] = []
        self.counters: dict[str, int] = {j: 0 for j in JOURNAUX}

    def entry(self, journal: str, date: pd.Timestamp, piece: str, lib: str,
              lines: list[tuple[str, str, str, str, float, float]],
              valid_delay: int | None = None) -> str:
        self.counters[journal] += 1
        num = f"{journal}{self.counters[journal]:06d}"
        delay = valid_delay if valid_delay is not None else int(self.rng.integers(0, 8))
        for compte, compte_lib, aux, aux_lib, debit, credit in lines:
            self.lines.append({
                "JournalCode": journal, "JournalLib": JOURNAUX[journal], "EcritureNum": num,
                "EcritureDate": date, "CompteNum": compte, "CompteLib": compte_lib,
                "CompAuxNum": aux, "CompAuxLib": aux_lib, "PieceRef": piece, "PieceDate": date,
                "EcritureLib": lib, "Debit": round(debit, 2), "Credit": round(credit, 2),
                "EcritureLet": "", "DateLet": "", "ValidDate": date + pd.Timedelta(days=delay),
                "Montantdevise": "", "Idevise": "",
            })
        return num


def _purchase_lines(sup: dict, ttc: float) -> list:
    ht = round(ttc / 1.2, 2)
    return [(sup["charge"], CHARGES[sup["charge"]], "", "", ht, 0.0),
            ("445660", "TVA déductible sur ABS", "", "", round(ttc - ht, 2), 0.0),
            ("401000", "Fournisseurs", sup["aux"], sup["name"], 0.0, ttc)]


def generate(year: int = 2025, n_suppliers: int = 45, n_customers: int = 140,
             seed: int = 2025) -> FecDataset:
    rng = np.random.default_rng(seed)
    b = _Builder(rng)
    truth: list[tuple[str, str]] = []
    days = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")
    feries = jours_feries(year)
    workdays = [d for d in days if is_business_day(d, feries)]
    sundays = [d for d in days if d.dayofweek == 6] + sorted(feries)

    suppliers = [{
        "aux": f"F{i:04d}", "name": f"Fournisseur {i:02d}",
        "charge": rng.choice(list(CHARGES)),
        "median": float(np.exp(rng.uniform(np.log(150), np.log(2_500)))),
        "freq": rng.uniform(40, 200),
    } for i in range(1, n_suppliers + 1)]
    fabricated = suppliers[min(36, n_suppliers - 1)]  # fournisseur aux montants « inventés »
    fabricated["freq"] = 260

    def pick_day() -> pd.Timestamp:
        return workdays[int(rng.integers(0, len(workdays)))]

    # ---------------------------------------------------------------- achats + règlements
    for sup in suppliers:
        n = int(rng.poisson(sup["freq"]))
        for k in range(n):
            d = pick_day()
            if sup is fabricated:
                ttc = round(float(rng.uniform(6_000, 9_999)) / 3, 2)  # 2 000 – 3 333 €
                if rng.random() < 0.85:
                    ttc = round(float(rng.uniform(700, 999)), 2)       # chiffres 7-9 sur-représentés
            else:
                ttc = round(float(rng.lognormal(np.log(sup["median"]), 0.9)), 2)
            piece = f"FA-{sup['aux']}-{year}{k:04d}"
            num = b.entry("ACH", d, piece, f"Facture {sup['name']}", _purchase_lines(sup, ttc))
            if sup is fabricated:
                truth.append((num, "SERIE_FABRIQUEE"))
            pay = d + pd.Timedelta(days=int(rng.integers(20, 60)))
            if pay.year == year:
                while not is_business_day(pay, feries):
                    pay += pd.Timedelta(days=1)
                if pay.year == year:
                    b.entry("BQ", pay, f"VIR-{piece}", f"Règlement {sup['name']}",
                            [("401000", "Fournisseurs", sup["aux"], sup["name"], ttc, 0.0),
                             ("512000", "Banque", "", "", 0.0, ttc)])

    # ---------------------------------------------------------------- ventes + encaissements
    for c in range(1, n_customers + 1):
        aux, name = f"C{c:04d}", f"Client {c:03d}"
        med = float(np.exp(rng.uniform(np.log(300), np.log(6_000))))
        for k in range(int(rng.poisson(rng.uniform(25, 90)))):
            d = pick_day()
            ttc = round(float(rng.lognormal(np.log(med), 0.7)), 2)
            ht = round(ttc / 1.2, 2)
            piece = f"FV-{year}-{aux}-{k:03d}"
            b.entry("VTE", d, piece, f"Facture {name}",
                    [("411000", "Clients", aux, name, ttc, 0.0),
                     ("706000", "Prestations de services", "", "", 0.0, ht),
                     ("445710", "TVA collectée", "", "", 0.0, round(ttc - ht, 2))])
            enc = d + pd.Timedelta(days=int(rng.integers(15, 75)))
            while enc.year == year and not is_business_day(enc, feries):
                enc += pd.Timedelta(days=1)
            if enc.year == year:
                b.entry("BQ", enc, f"ENC-{piece}", f"Encaissement {name}",
                        [("512000", "Banque", "", "", ttc, 0.0),
                         ("411000", "Clients", aux, name, 0.0, ttc)])

    # ---------------------------------------------------------------- paie mensuelle
    for m in range(1, 13):
        d = pd.Timestamp(year, m, 1) + pd.offsets.MonthEnd(0)
        while not is_business_day(d, feries):
            d -= pd.Timedelta(days=1)
        brut = round(float(rng.normal(98_000, 4_000)), 2)
        charges = round(brut * 0.42, 2)
        net = round(brut * 0.78, 2)
        b.entry("PAIE", d, f"PAIE-{year}{m:02d}", f"Paie {d:%m/%Y}",
                [("641000", "Rémunérations du personnel", "", "", brut, 0.0),
                 ("645000", "Charges sociales", "", "", charges, 0.0),
                 ("421000", "Personnel - rémunérations dues", "", "", 0.0, net),
                 ("431000", "Sécurité sociale", "", "", 0.0, round(brut + charges - net, 2))])

    # ================================================================ anomalies injectées
    def rand_sup():
        while True:
            s = suppliers[int(rng.integers(0, n_suppliers))]
            if s is not fabricated:
                return s

    # A. Doublons de factures fournisseurs (même pièce, même montant)
    fec_tmp = pd.DataFrame(b.lines)
    ach = fec_tmp[(fec_tmp["JournalCode"] == "ACH") & (fec_tmp["CompteNum"] == "401000")
                  & (fec_tmp["CompAuxNum"] != fabricated["aux"])]
    for _, row in ach.sample(25, random_state=seed).iterrows():
        sup = next(s for s in suppliers if s["aux"] == row["CompAuxNum"])
        d = row["EcritureDate"] + pd.Timedelta(days=int(rng.integers(3, 25)))
        while not is_business_day(d, feries):
            d += pd.Timedelta(days=1)
        if d.year != year:
            continue
        num = b.entry("ACH", d, row["PieceRef"], f"Facture {sup['name']}",
                      _purchase_lines(sup, row["Credit"]))
        truth.append((num, "DOUBLON"))

    # B. Écritures passées un dimanche ou un jour férié
    for _ in range(30):
        sup = rand_sup()
        d = sundays[int(rng.integers(0, len(sundays)))]
        ttc = round(float(rng.lognormal(np.log(sup["median"]), 0.9)), 2)
        num = b.entry("ACH", d, f"FA-{sup['aux']}-X{rng.integers(1e5)}", f"Facture {sup['name']}",
                      _purchase_lines(sup, ttc))
        truth.append((num, "JOUR_NON_OUVRE"))

    # C. Validation très tardive (écriture antidatée)
    for _ in range(25):
        sup = rand_sup()
        d = workdays[int(rng.integers(0, len(workdays) - 120))]
        ttc = round(float(rng.lognormal(np.log(sup["median"]), 0.9)), 2)
        num = b.entry("ACH", d, f"FA-{sup['aux']}-L{rng.integers(1e5)}", f"Facture {sup['name']}",
                      _purchase_lines(sup, ttc), valid_delay=int(rng.integers(75, 200)))
        truth.append((num, "VALIDATION_TARDIVE"))

    # D. OD manuelles à montants ronds
    for _ in range(20):
        d = pick_day()
        amt = float(rng.choice([2_000, 5_000, 10_000, 15_000, 20_000, 25_000]))
        compte = rng.choice(["622600", "615000", "606400"])
        num = b.entry("OD", d, f"OD-{rng.integers(1e5)}", "Régularisation",
                      [(compte, CHARGES[compte], "", "", amt, 0.0),
                       ("512000", "Banque", "", "", 0.0, amt)])
        truth.append((num, "MONTANT_ROND_OD"))

    # E. Fractionnement de factures sous le seuil de validation
    for _ in range(10):
        sup = rand_sup()
        d0 = workdays[int(rng.integers(0, len(workdays) - 10))]
        for j in range(3):
            d = d0 + pd.Timedelta(days=j * 2)
            while not is_business_day(d, feries):
                d += pd.Timedelta(days=1)
            ttc = round(float(rng.uniform(0.9, 0.995) * SEUIL_VALIDATION), 2)
            num = b.entry("ACH", d, f"FA-{sup['aux']}-S{rng.integers(1e5)}",
                          f"Facture {sup['name']}", _purchase_lines(sup, ttc))
            truth.append((num, "FRACTIONNEMENT"))

    # F. Montants très atypiques pour le fournisseur
    for _ in range(25):
        sup = rand_sup()
        d = pick_day()
        ttc = round(sup["median"] * float(rng.uniform(25, 60)), 2)
        num = b.entry("ACH", d, f"FA-{sup['aux']}-A{rng.integers(1e5)}", f"Facture {sup['name']}",
                      _purchase_lines(sup, ttc))
        truth.append((num, "MONTANT_ATYPIQUE"))

    fec = pd.DataFrame(b.lines)

    # G. Écritures déséquilibrées (débit ≠ crédit)
    nums = fec.loc[fec["JournalCode"] == "VTE", "EcritureNum"].drop_duplicates()
    for num in nums.sample(15, random_state=seed + 3):
        idx = fec.index[(fec["EcritureNum"] == num) & (fec["Credit"] > 0)][0]
        fec.at[idx, "Credit"] = round(fec.at[idx, "Credit"] + float(rng.uniform(1, 80)), 2)
        truth.append((num, "DESEQUILIBRE"))

    # Renumérotation chronologique continue par journal (exigence du format FEC) :
    # les écritures injectées ne se distinguent donc pas par leur numéro.
    fec = fec.sort_values(["EcritureDate", "JournalCode", "EcritureNum"], kind="stable")
    first = fec.drop_duplicates("EcritureNum")[["JournalCode", "EcritureNum"]]
    first["new"] = first["JournalCode"] + (first.groupby("JournalCode").cumcount() + 1).map("{:06d}".format)
    mapping = dict(zip(first["EcritureNum"], first["new"]))
    fec["EcritureNum"] = fec["EcritureNum"].map(mapping)
    truth_df = pd.DataFrame(truth, columns=["EcritureNum", "anomaly_type"])
    truth_df["EcritureNum"] = truth_df["EcritureNum"].map(mapping)
    return FecDataset(fec=fec[FEC_COLUMNS].reset_index(drop=True), truth=truth_df)


def to_fec_file(fec: pd.DataFrame, path) -> None:
    """Écrit au format DGFiP : tabulation, dates AAAAMMJJ, décimales à la virgule."""
    out = fec.copy()
    for c in ("EcritureDate", "PieceDate", "ValidDate"):
        out[c] = pd.to_datetime(out[c]).dt.strftime("%Y%m%d")
    for c in ("Debit", "Credit"):
        out[c] = out[c].map(lambda x: f"{x:.2f}".replace(".", ","))
    out.to_csv(path, sep="\t", index=False, encoding="utf-8")
