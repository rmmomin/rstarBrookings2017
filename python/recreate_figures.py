from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .routines import plot_states_shaded, save_figure

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_RUN = BASE_DIR / "output" / "OutputModel1"
DATA_DIR = DEFAULT_RUN / "Data"
FIG_DIR = DEFAULT_RUN / "Figures"


def load_quantiles(df: pd.DataFrame, prefix: str) -> np.ndarray:
    cols = [
        f"{prefix}_p2_5",
        f"{prefix}_p16",
        f"{prefix}_med",
        f"{prefix}_p84",
        f"{prefix}_p97_5",
    ]
    return df[cols].to_numpy()


def median_series(df: pd.DataFrame, series: str) -> np.ndarray:
    subset = df[df["Series"] == series].sort_values("Date")
    return subset["p50"].to_numpy()


def fig_with_shaded(
    time: np.ndarray,
    quantiles: np.ndarray,
    title: str,
    ylabel: str = "Percent, annualized",
    overlays: Optional[List[dict]] = None,
    ylim: Optional[tuple[float, float]] = None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 4))
    plot_states_shaded(time, quantiles, ax=ax)
    median_series = quantiles[:, 2]
    last_value = median_series[-1]
    last_date = pd.to_datetime(time[-1]).to_period("Q")
    if overlays:
        for overlay in overlays:
            ax.plot(time, overlay["y"], overlay["style"], **overlay.get("kwargs", {}))
    if ylim:
        ax.set_ylim(*ylim)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold", fontsize=14, loc="left")
    ax.set_xlabel("Year")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, linestyle=":", linewidth=0.5, color="0.8")
    ax.annotate(
        f"{last_date}  {last_value:.2f}",
        xy=(time[-1], median_series[-1]),
        xytext=(0, 12),
        textcoords="offset points",
        fontsize=10,
        fontweight="bold",
        ha="center",
        va="bottom",
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "0.7"},
    )
    fig.text(
        0.07,
        0.01,
        "Source: Del Negro et al. (2017), Brookings BPEA replication and extension.",
        fontsize=9,
    )
    return fig


def recreate_figures(data_dir: Path = DATA_DIR, fig_dir: Path = FIG_DIR) -> None:
    data_dir = data_dir.expanduser().resolve()
    fig_dir.mkdir(parents=True, exist_ok=True)

    outmod = pd.read_csv(data_dir / "OutMod1forCharts.csv", parse_dates=["Date"])
    time = outmod["Date"].to_numpy(dtype="datetime64[ns]")
    q_pi = load_quantiles(outmod, "Pi_bar")
    q_r = load_quantiles(outmod, "R_bar")
    q_nr = load_quantiles(outmod, "NR_bar")
    q_ts = load_quantiles(outmod, "Ts_bar")

    observed = pd.read_csv(data_dir / "ObservedTrends_quantiles.csv", parse_dates=["Date"])
    observed = observed.sort_values("Date")
    pi_obs = median_series(observed, "Pi")
    epi_obs = median_series(observed, "EPi")
    bill = median_series(observed, "BILL")
    ebill = median_series(observed, "EBILL")
    tblong = median_series(observed, "TBlong")

    overlays = {
        "pi": pi_obs,
        "pi_e": epi_obs,
        "r_minus_pi_ex_post": ebill - epi_obs,
        "r_minus_pi_ex_ante": bill - epi_obs,
        "credit_spread": tblong - bill,
    }

    specs = [
        ("PIbar.png", q_pi, r"Trend inflation $(\pi_t^{\mathrm{trend}})$ estimated from Model 1", None, None),
        (
            "PIbar_obs.png",
            q_pi,
            r"Observed inflation $(\pi_t)$ vs. survey inflation expectations $(\pi_t^e)$",
            [
                {"y": overlays["pi_e"], "style": "b-", "kwargs": {"linewidth": 2.0}},
                {"y": overlays["pi"], "style": "b:", "kwargs": {"linewidth": 1.0}},
            ],
            None,
        ),
        (
            "Rbar.png",
            q_r,
            r"Natural real rate $(r_t^*)$ estimated from Model 1 (baseline trendy VAR)",
            None,
            None,
        ),
        (
            "NRbar.png",
            q_nr,
            r"Nominal neutral short rate $(i_t^* = r_t^* + \pi_t^{\mathrm{trend}})$ implied by Model 1",
            None,
            None,
        ),
        (
            "Rbar_obs.png",
            q_r,
            r"Natural real rate $(r_t^*)$ (Model 1) vs. ex-ante and ex-post real rates",
            [
                {"y": overlays["r_minus_pi_ex_post"], "style": "b*-", "kwargs": {"linewidth": 2.0}},
                {"y": overlays["r_minus_pi_ex_ante"], "style": "b:", "kwargs": {"linewidth": 1.0}},
            ],
            None,
        ),
        (
            "TSbar.png",
            q_ts,
            "Term premium estimated from Model 1 (baseline trendy VAR)",
            None,
            None,
        ),
        (
            "TSbar_obs.png",
            q_ts,
            r"Term premium vs observed difference $(r^L - r)$",
            [
                {"y": overlays["credit_spread"], "style": "b:", "kwargs": {"linewidth": 1.0}},
            ],
            None,
        ),
        (
            "Rscaled.png",
            q_r,
            r"Trend natural real rate $(r_t^*)$ (Model 1, zoomed scale)",
            None,
            (-0.5, 3.5),
        ),
        (
            "TSscaled.png",
            q_ts,
            "Trend credit-spread / convenience-yield proxy (zoomed scale)",
            None,
            (-0.5, 3.5),
        ),
    ]

    for filename, quantiles, title, overlay_list, ylim in specs:
        fig = fig_with_shaded(time, quantiles, title, overlays=overlay_list, ylim=ylim)
        save_figure(fig, fig_dir / filename)
        plt.close(fig)
        print(f"Wrote {filename}")


if __name__ == "__main__":
    recreate_figures()

