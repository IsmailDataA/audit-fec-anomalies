"""Audit analytique d'un FEC : conformité, tests d'audit, Benford, score de risque.

Usage :
    python run_audit.py                       # génère un FEC de démonstration puis l'audite
    python run_audit.py --fec mon_fichier.txt # audite un vrai FEC (sans mesure de détection)
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from jinja2 import Environment, FileSystemLoader

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from fec_audit import benford as B  # noqa: E402
from fec_audit import charts  # noqa: E402
from fec_audit.controls import CONTROLS, entries_view, run_controls  # noqa: E402
from fec_audit.fec_io import conformity_checks, read_fec  # noqa: E402
from fec_audit.generate_fec import generate, to_fec_file  # noqa: E402
from fec_audit.scoring import isolation_scores, risk_score  # noqa: E402

REPO_URL = "https://github.com/ismaildataa/audit-fec-anomalies"
SAMPLE_SIZE = 300


def fr(x: float, d: int = 0) -> str:
    return f"{x:,.{d}f}".replace(",", " ").replace(".", ",")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fec", help="chemin d'un FEC réel")
    ap.add_argument("--seed", type=int, default=2025)
    args = ap.parse_args()
    t0 = time.perf_counter()
    out, docs = ROOT / "outputs", ROOT / "docs"
    out.mkdir(exist_ok=True)
    (docs / "img").mkdir(parents=True, exist_ok=True)

    truth = None
    if args.fec:
        fec_path = Path(args.fec)
    else:
        print("1/5  Génération d'un FEC de démonstration…")
        ds = generate(seed=args.seed)
        fec_path = ROOT / "data" / "FEC_DEMO_2025.txt"
        fec_path.parent.mkdir(exist_ok=True)
        to_fec_file(ds.fec, fec_path)
        truth = ds.truth

    print("2/5  Lecture et conformité du fichier…")
    fec = read_fec(fec_path)
    conf = conformity_checks(fec)

    print("3/5  Tests d'audit…")
    e = entries_view(fec)
    flags = run_controls(e)
    summary = pd.DataFrame([{"code": c.code, "libelle": c.libelle, "poids": c.poids,
                             "signalees": int(flags[c.code].sum())} for c in CONTROLS])

    print("4/5  Benford & score de risque…")
    ach = e[(e["JournalCode"] == "ACH") & (e["tiers"] != "")].copy()
    ach["montant"] = ach["montant_tiers"]
    mad_g = B.mad(ach["montant"])
    base = mad_g
    contrib = sorted(((k, len(g), base - B.mad(ach.loc[ach["tiers"] != k, "montant"]))
                      for k, g in ach.groupby("tiers") if len(g) >= 30), key=lambda x: -x[2])
    sup_top = contrib[0][0]
    mad_s = B.mad(ach.loc[ach["tiers"] == sup_top, "montant"])
    benford_table = pd.DataFrame(contrib[:5], columns=["tiers", "n", "contribution"])
    benford_table["mad_tiers"] = [round(B.mad(ach.loc[ach["tiers"] == k, "montant"]), 4)
                                  for k in benford_table["tiers"]]

    ml = isolation_scores(e)
    ranked = risk_score(e, flags, ml)

    print("5/5  Rapport & exports…")
    charts.benford_pair(B.distribution(ach["montant"]), B.distribution(ach.loc[ach["tiers"] == sup_top, "montant"]),
                        f"Fournisseur {sup_top}", mad_g, mad_s, docs / "img" / "benford.png")
    charts.controls_hits(summary, docs / "img" / "tests_audit.png")

    metrics = None
    if truth is not None:
        bad_all = set(truth["EcritureNum"])
        bad_entry = set(truth.loc[truth["anomaly_type"] != "SERIE_FABRIQUEE", "EcritureNum"])
        hit = ranked["EcritureNum"].isin(bad_entry).values
        ml_rank = e.assign(s=ml.values).sort_values("s", ascending=False)["EcritureNum"]
        hit_ml = ml_rank.isin(bad_entry).values
        rng = np.random.default_rng(0)
        hit_rand = rng.permutation(hit)
        curves = {"Score de risque (règles + ML)": np.cumsum(hit) / len(bad_entry),
                  "Isolation Forest seule": np.cumsum(hit_ml) / len(bad_entry),
                  "Échantillon aléatoire": np.cumsum(hit_rand) / len(bad_entry)}
        charts.capture_curve(curves, len(e), docs / "img" / "courbe_capture.png")
        per_type = []
        for c in CONTROLS:
            det = set(flags.loc[flags[c.code], "EcritureNum"])
            tr = set(truth.loc[truth["anomaly_type"] == c.code, "EcritureNum"])
            per_type.append({"libelle": c.libelle, "injectees": len(tr), "signalees": len(det),
                             "trouvees": len(det & tr),
                             "rappel": fr(100 * len(det & tr) / max(len(tr), 1), 0)})
        top = ranked.head(SAMPLE_SIZE)
        metrics = {
            "p100": fr(100 * ranked.head(100)["EcritureNum"].isin(bad_all).mean(), 0),
            "p100_ml": fr(100 * ml_rank.head(100).isin(bad_all).mean(), 0),
            "recall_sample": fr(100 * top["EcritureNum"].isin(bad_entry).sum() / len(bad_entry), 1),
            "random_sample": fr(100 * hit_rand[:SAMPLE_SIZE].sum() / len(bad_entry), 1),
            "n_injected": len(bad_entry), "per_type": per_type,
            "fab_rank": [r[0] for r in contrib].index(sup_top) + 1,
        }

    with pd.ExcelWriter(out / "echantillon_a_controler.xlsx") as xw:
        ranked.head(SAMPLE_SIZE).to_excel(xw, sheet_name="Échantillon", index=False)
        summary.to_excel(xw, sheet_name="Tests d'audit", index=False)
        conf.to_excel(xw, sheet_name="Conformité FEC", index=False)
        benford_table.to_excel(xw, sheet_name="Benford", index=False)

    duree = time.perf_counter() - t0
    top20 = ranked.head(20).copy()
    top20["EcritureDate"] = pd.to_datetime(top20["EcritureDate"]).dt.strftime("%d/%m/%Y")
    top20["montant"] = top20["montant"].map(lambda x: fr(x, 2) + " €")
    courts = {"DESEQUILIBRE": "déséquilibre", "DOUBLON": "doublon", "FRACTIONNEMENT": "fractionnement",
              "MONTANT_ATYPIQUE": "montant atypique", "VALIDATION_TARDIVE": "validation tardive",
              "MONTANT_ROND_OD": "OD montant rond", "JOUR_NON_OUVRE": "jour non ouvré"}
    top20["motifs"] = top20["motifs"].map(
        lambda m: ", ".join(courts[x] for x in m.split(", ")) if m else "profil atypique (Isolation Forest)")
    env = Environment(loader=FileSystemLoader(ROOT / "templates"))
    html = env.get_template("report.html.j2").render(
        run_date=date.today().strftime("%d/%m/%Y"), repo_url=REPO_URL,
        n_lines=fr(len(fec)), n_entries=fr(len(e)), n_journaux=fec["JournalCode"].nunique(),
        exercice=f"{fec['EcritureDate'].min():%d/%m/%Y} – {fec['EcritureDate'].max():%d/%m/%Y}",
        conf=conf.to_dict("records"), n_conf_ko=int((conf["statut"] == "KO").sum()),
        summary=summary.to_dict("records"), sample=SAMPLE_SIZE,
        pct_sample=fr(100 * SAMPLE_SIZE / len(e), 1),
        mad_g=f"{mad_g:.4f}".replace(".", ","), verdict_g=B.verdict(mad_g), sup_top=sup_top, mad_s=f"{mad_s:.4f}",
        top20=top20.to_dict("records"), m=metrics, duree=fr(duree, 0))
    (docs / "index.html").write_text(html, encoding="utf-8")
    ranked.head(SAMPLE_SIZE).to_csv(out / "echantillon_a_controler.csv", index=False)

    print(f"\nTerminé en {duree:.1f} s — {len(e)} écritures, échantillon de {SAMPLE_SIZE}")
    if metrics:
        print(f"Précision top 100 : {metrics['p100']} % (Isolation Forest seule : {metrics['p100_ml']} %)")
        print(f"Anomalies captées dans l'échantillon : {metrics['recall_sample']} % "
              f"(aléatoire : {metrics['random_sample']} %)")


if __name__ == "__main__":
    main()
