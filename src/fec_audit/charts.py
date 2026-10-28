"""Graphiques du rapport d'audit (matplotlib, palette validée daltonisme)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .benford import BENFORD  # noqa: E402

BLUE, ORANGE, AQUA, CRITICAL = "#2a78d6", "#eb6834", "#1baf7a", "#d03b3b"
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"


def _style(ax, title, subtitle=None):
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold", color=INK, pad=22 if subtitle else 10)
    if subtitle:
        ax.text(0, 1.02, subtitle, transform=ax.transAxes, fontsize=9.2, color=INK2)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor="#fcfcfb")
    plt.close(fig)


def benford_pair(global_dist: pd.Series, sup_dist: pd.Series, sup_label: str,
                 mad_g: float, mad_s: float, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for ax, dist, title, sub in [
        (axes[0], global_dist, "Tous les achats", f"MAD = {mad_g:.4f} → conforme à Benford".replace(".", ",")),
        (axes[1], sup_dist, sup_label, f"MAD = {mad_s:.4f} → non conforme".replace(".", ",")),
    ]:
        x = np.arange(1, 10)
        ax.bar(x, dist.values * 100, color=BLUE, width=0.62, label="Observé")
        ax.plot(x, BENFORD.values * 100, color=ORANGE, linewidth=2, marker="o", markersize=5,
                label="Loi de Benford")
        ax.set_xticks(x)
        ax.set_xlabel("Premier chiffre du montant", color=INK2, fontsize=9)
        _style(ax, title, sub)
    axes[0].set_ylabel("% des montants", color=INK2, fontsize=9)
    axes[0].legend(frameon=False, fontsize=9, labelcolor=INK2)
    _save(fig, path)


def capture_curve(curves: dict[str, np.ndarray], n_total: int, path: Path, zoom: int = 600) -> None:
    fig, ax = plt.subplots(figsize=(9, 3.8))
    colors = {"Score de risque (règles + ML)": BLUE, "Isolation Forest seule": ORANGE,
              "Échantillon aléatoire": MUTED}
    for name, y in curves.items():
        x = np.arange(1, len(y) + 1)
        ax.plot(x[:zoom], 100 * y[:zoom], color=colors[name], linewidth=2,
                linestyle="--" if "aléatoire" in name else "-")
        ax.text(zoom * 1.01, 100 * y[zoom - 1], name, color=INK2, fontsize=9, va="center")
    ax.set_xlim(0, zoom * 1.42)
    ax.set_ylim(0, 105)
    ax.set_xlabel(f"Nombre d'écritures contrôlées (sur {n_total:,})".replace(",", " "), color=INK2, fontsize=9)
    ax.set_ylabel("% des anomalies trouvées", color=INK2, fontsize=9)
    _style(ax, "Combien d'écritures faut-il contrôler pour trouver les anomalies ?",
           "Courbe de capture : plus elle monte vite, moins l'auditeur perd de temps")
    _save(fig, path)


def controls_hits(summary: pd.DataFrame, path: Path) -> None:
    s = summary.sort_values("signalees")
    fig, ax = plt.subplots(figsize=(9, 3.6))
    bars = ax.barh(s["libelle"], s["signalees"], color=BLUE, height=0.6)
    ax.bar_label(bars, padding=4, color=INK2, fontsize=9)
    _style(ax, "Écritures signalées par test d'audit")
    ax.grid(axis="x", color=GRID)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", colors=INK2)
    _save(fig, path)
