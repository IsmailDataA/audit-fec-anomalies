import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fec_audit import benford as B  # noqa: E402
from fec_audit.calendar_fr import jours_feries  # noqa: E402
from fec_audit.controls import CONTROLS, entries_view, run_controls  # noqa: E402
from fec_audit.fec_io import conformity_checks, read_fec  # noqa: E402
from fec_audit.generate_fec import generate, to_fec_file  # noqa: E402


@pytest.fixture(scope="module")
def demo(tmp_path_factory):
    ds = generate(n_suppliers=20, n_customers=40, seed=11)
    path = tmp_path_factory.mktemp("fec") / "FEC.txt"
    to_fec_file(ds.fec, path)
    fec = read_fec(path)
    e = entries_view(fec)
    return ds, fec, e, run_controls(e)


def test_feries_2025_incluent_paques_et_pentecote():
    f = jours_feries(2025)
    assert pd.Timestamp("2025-04-21") in f  # lundi de Pâques
    assert pd.Timestamp("2025-06-09") in f  # lundi de Pentecôte
    assert pd.Timestamp("2025-05-29") in f  # Ascension


def test_round_trip_format_dgfip(demo):
    ds, fec, _, _ = demo
    assert list(fec.columns)[:4] == ["JournalCode", "JournalLib", "EcritureNum", "EcritureDate"]
    assert len(fec) == len(ds.fec)
    assert fec["Debit"].sum() == pytest.approx(ds.fec["Debit"].sum(), abs=0.01)


def test_conformite_detecte_les_desequilibres(demo):
    _, fec, _, _ = demo
    conf = conformity_checks(fec).set_index("controle")
    assert conf.loc["Chaque écriture est équilibrée", "statut"] == "KO"
    assert conf.loc["18 colonnes réglementaires présentes", "statut"] == "OK"


@pytest.mark.parametrize("code", ["DESEQUILIBRE", "DOUBLON", "VALIDATION_TARDIVE",
                                  "MONTANT_ROND_OD", "JOUR_NON_OUVRE", "FRACTIONNEMENT"])
def test_chaque_anomalie_injectee_est_signalee(demo, code):
    ds, _, _, flags = demo
    injected = set(ds.truth.loc[ds.truth["anomaly_type"] == code, "EcritureNum"])
    flagged = set(flags.loc[flags[code], "EcritureNum"])
    assert injected and injected <= flagged


def test_benford_reference():
    assert B.BENFORD.sum() == pytest.approx(1.0)
    assert B.BENFORD[1] == pytest.approx(0.30103, abs=1e-5)
    assert B.verdict(0.004) == "Conformité forte"
    assert B.verdict(0.05) == "Non conforme"


def test_tous_les_controles_sont_executes(demo):
    *_, flags = demo
    assert all(c.code in flags.columns for c in CONTROLS)
