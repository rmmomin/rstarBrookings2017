from __future__ import annotations

import csv
import os
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
MATLAB_CSV = BASE_DIR.parents[0] / "tvar" / "FiguresModel1" / "OutMod1forCharts.csv"
PYTHON_CSV = BASE_DIR / "output" / "OutputModel1" / "Data" / "OutMod1forCharts.csv"
OUT_BASE = BASE_DIR / "output" / "BandedFigures"


def parse_date(date_str: str) -> np.datetime64:
    for fmt in ("%Y-%m-%d", "%d-%b-%Y", "%d-%m-%Y"):
        try:
            return np.datetime64(datetime.strptime(date_str, fmt))
        except ValueError:
            continue
    # Fallback: let numpy try
    return np.datetime64(date_str)


def load_outmod(csv_path: Path, adjust_day: bool = False) -> dict:
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        records = list(reader)
    dates = [parse_date(r["Date"]) for r in records]
    if adjust_day:
        dates = [d - np.timedelta64(1, "D") for d in dates]

    def col(name: str) -> np.ndarray:
        return np.array([float(r[name]) for r in records], dtype=float)

    data = {
        "Date": np.array(dates),
        "Pi": col("Pi_bar_med"),
        "Pi_q": np.column_stack(
            [col("Pi_bar_p2_5"), col("Pi_bar_p16"), col("Pi_bar_med"), col("Pi_bar_p84"), col("Pi_bar_p97_5")]
        ),
        "R": col("R_bar_med"),
        "R_q": np.column_stack(
            [col("R_bar_p2_5"), col("R_bar_p16"), col("R_bar_med"), col("R_bar_p84"), col("R_bar_p97_5")]
        ),
        "Ts": col("Ts_bar_med"),
        "Ts_q": np.column_stack(
            [col("Ts_bar_p2_5"), col("Ts_bar_p16"), col("Ts_bar_med"), col("Ts_bar_p84"), col("Ts_bar_p97_5")]
        ),
    }
    return data


def plot_with_bands(
    time: np.ndarray,
    quantiles: np.ndarray,
    title: str,
    overlays: Optional[List[dict]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    ylabel: str = "Percent, annualized",
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 4))
    t = time.astype("datetime64[ns]")

    q = quantiles
    # Shaded 95 and 68 percent intervals
    ax.fill_between(t, q[:, 0], q[:, 4], color="0.8", alpha=0.6, linewidth=0)
    ax.fill_between(t, q[:, 1], q[:, 3], color="0.6", alpha=0.4, linewidth=0)
    ax.plot(t, q[:, 2], color="tab:blue", linewidth=1.8, label="Median")

    if overlays:
        for ov in overlays:
            ax.plot(t, ov["y"], ov.get("style", "k-"), **ov.get("kwargs", {}))

    if ylim:
        ax.set_ylim(*ylim)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold", fontsize=14, loc="left")
    ax.set_xlabel("Year")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, linestyle=":", linewidth=0.5, color="0.8")

    median_series = q[:, 2]
    last_val = float(median_series[-1])
    last_date = np.datetime_as_string(t[-1], unit="D")[:7]  # YYYY-MM
    ax.annotate(
        f"{last_date}  {last_val:.2f}",
        xy=(t[-1], median_series[-1]),
        xytext=(0, 10),
        textcoords="offset points",
        fontsize=10,
        fontweight="bold",
        ha="center",
        va="bottom",
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "0.7"},
    )
    fig.text(
        0.07,
        0.01,
        "Source: Del Negro et al. (2017), Brookings BPEA replication and extension.",
        fontsize=9,
    )
    fig.tight_layout()
    return fig


def make_banded_figures(df: dict, out_dir: Path, label_prefix: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    time = df["Date"]

    q_pi = df["Pi_q"]
    q_r = df["R_q"]
    q_ts = df["Ts_q"]

    figs = [
        (
            q_pi,
            r"Trend inflation $(\pi_t^{\mathrm{trend}})$ estimated from Model 1",
            "PIbar_bands.png",
            None,
            None,
        ),
        (
            q_r,
            r"Natural real rate $(r_t^*)$ estimated from Model 1",
            "Rbar_bands.png",
            None,
            None,
        ),
        (
            q_ts,
            "Term premium estimated from Model 1 (baseline trendy VAR)",
            "TSbar_bands.png",
            None,
            None,
        ),
    ]

    for quant, title, fname, overlays, ylim in figs:
        fig = plot_with_bands(time, quant, title, overlays=overlays, ylim=ylim)
        save_path = out_dir / fname
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    py_df = load_outmod(PYTHON_CSV)
    matlab_df = load_outmod(MATLAB_CSV, adjust_day=True)

    make_banded_figures(py_df, OUT_BASE / "python", "Python")
    make_banded_figures(matlab_df, OUT_BASE / "matlab", "MATLAB")
    print(f"Wrote banded figures to {OUT_BASE}")


if __name__ == "__main__":
    main()

