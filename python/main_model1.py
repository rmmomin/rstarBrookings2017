from __future__ import annotations

import math
import multiprocessing as mp
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:  # NumPy < 1.20 fallback
    from numpy.typing import NDArray
except ImportError:  # pragma: no cover - older environments
    NDArray = np.ndarray  # type: ignore

from scipy.linalg import solve_discrete_lyapunov

from .routines import (
    bvar,
    covariance_draw,
    kalman_filter,
    kalman_smoother_draw,
    plot_states_shaded,
    save_figure,
)

PYTHON_DIR = Path(__file__).resolve().parent
REPO_ROOT = PYTHON_DIR.parents[0]
LEGACY_TVAR_DIR = REPO_ROOT / "tvar"
PYTHON_DATA_DIR = PYTHON_DIR / "data"
OUTPUT_ROOT = PYTHON_DIR / "output"
DATA_OUTPUT_DIR = OUTPUT_ROOT / "Data"
FIGURES_OUTPUT_DIR = OUTPUT_ROOT / "Figures"


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError as exc:  # pragma: no cover - protective branch
        raise ValueError(f"{name} must be an integer, got {value!r}") from exc


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def matlab_datenum(dt: datetime) -> float:
    """Convert Python datetime to MATLAB serial day number."""
    ordinal = dt.toordinal() + 366
    seconds = (
        dt - datetime(dt.year, dt.month, dt.day)
    ).total_seconds()
    return ordinal + seconds / 86400.0


@dataclass
class SharedState:
    y: NDArray[np.float64]
    Y: NDArray[np.float64]
    time_datetimes: NDArray[np.datetime64]
    time_serial: NDArray[np.float64]
    mnemonics: Tuple[str, ...]
    C: NDArray[np.float64]
    r: int
    n: int
    p: int
    rn: int
    b0: NDArray[np.float64]
    df0tr: float
    SC0tr: NDArray[np.float64]
    S0tr: NDArray[np.float64]
    P0tr: NDArray[np.float64]
    psi: NDArray[np.float64]
    transition: NDArray[np.float64]
    measurement_cov: NDArray[np.float64]
    process_cov: NDArray[np.float64]
    initial_state: NDArray[np.float64]
    initial_covariance: NDArray[np.float64]
    save_level: str


def load_input_data(path: Path) -> Tuple[pd.DatetimeIndex, List[str], NDArray[np.float64]]:
    raw = pd.read_excel(path)
    time_index = pd.to_datetime(raw.iloc[:, 0])
    mnemonics = [str(col) for col in raw.columns[1:]]
    values = raw.iloc[:, 1:].to_numpy(dtype=float)
    return pd.DatetimeIndex(time_index), mnemonics, values


def prepare_data() -> SharedState:
    # Controls -------------------------------------------------------------
    data_candidates = [
        PYTHON_DATA_DIR / "DataCompleteLatest.xls",
        LEGACY_TVAR_DIR / "DataCompleteLatest.xls",
    ]
    data_path = next((path for path in data_candidates if path.exists()), None)
    if data_path is None:
        raise FileNotFoundError(
            "Missing input data. Expected DataCompleteLatest.xls under "
            f"{PYTHON_DATA_DIR} or {LEGACY_TVAR_DIR}."
        )
    time_index, mnemonics_full, values_full = load_input_data(data_path)

    first_year = 1960
    last_year = 2025
    selection = [0, 1, 2, 3, 4]

    mask = (time_index.year >= first_year) & (time_index.year <= last_year)
    time = time_index[mask]
    Y = values_full[mask][:, selection]
    y = Y.copy()
    mnemonics = tuple(mnemonics_full[i] for i in selection)

    T = len(time)
    bill_idx = [i for i, name in enumerate(mnemonics) if name.lower() == "bill"]
    bill_idx = bill_idx[0] if bill_idx else None

    zlb_periods = [
        (pd.Timestamp(2008, 12, 16), pd.Timestamp(2015, 12, 16)),
        (pd.Timestamp(2020, 3, 15), pd.Timestamp(2022, 3, 16)),
    ]
    is_zlb = np.zeros(T, dtype=bool)
    for start, end in zlb_periods:
        is_zlb |= (time >= start) & (time < end)
    if bill_idx is not None:
        y[is_zlb, bill_idx] = np.nan

    year1970_idx = np.where(time.year == 1970)[0]
    if year1970_idx.size:
        cutoff = year1970_idx[-1]
        if y.shape[1] > 1:
            y[: cutoff + 1, 1] = np.nan

    r = 3
    n = y.shape[1]
    p = 4
    rn = r + n * p

    Ctr = np.array(
        [
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 1.0],
        ]
    )
    Ccyc = np.zeros((n, n * p))
    Ccyc[:n, :n] = np.eye(n)
    C = np.hstack([Ctr, Ccyc])

    Atr = np.eye(r)
    Acyc = np.zeros((n * p, n * p))
    if p > 1:
        Acyc[n:, :-n] = np.eye(n * (p - 1))
    A = np.zeros((rn, rn))
    A[:r, :r] = Atr
    A[r:, r:] = Acyc

    R = np.eye(n) * 1e-12
    Q0cyc = np.zeros((n * p, n * p))
    psi = np.array([2.0, 1.0, 1.0, 0.5, 1.0])
    Q0cyc[:n, :n] = np.diag(psi)
    SC0tr = np.square(np.array([2.0, 1.0, 1.0])) / 400.0
    Q0tr = np.diag(SC0tr)
    Q = np.zeros((rn, rn))
    Q[:r, :r] = Q0tr
    Q[r:, r:] = Q0cyc

    S0tr = np.array([2.0, 0.5, 1.0])
    P0tr = np.eye(r)
    S0cyc = np.zeros(n * p)
    P0cyc = np.diag(np.tile(psi, p))

    S0 = np.concatenate([S0tr, S0cyc])
    P0 = np.zeros((rn, rn))
    P0[:r, :r] = P0tr
    P0[r:, r:] = P0cyc

    b0 = np.zeros((n * p, n))
    df0tr = 100.0

    time_serial = np.array([matlab_datenum(ts.to_pydatetime()) for ts in time])

    return SharedState(
        y=y,
        Y=Y,
        time_datetimes=time.to_numpy("datetime64[ns]"),
        time_serial=time_serial,
        mnemonics=mnemonics,
        C=C,
        r=r,
        n=n,
        p=p,
        rn=rn,
        b0=b0,
        df0tr=df0tr,
        SC0tr=SC0tr,
        S0tr=S0tr,
        P0tr=P0tr,
        psi=psi,
        transition=A,
        measurement_cov=R,
        process_cov=Q,
        initial_state=S0,
        initial_covariance=P0,
        save_level="lite",
    )


def run_chain(
    chain_id: int,
    seed: int,
    draws: int,
    thin: int,
    shared: SharedState,
    save_path: Optional[Path],
    collect: bool,
) -> Tuple[int, float]:
    rng = np.random.default_rng(seed)

    y = shared.y
    C = shared.C
    r = shared.r
    n = shared.n
    p = shared.p
    rn = shared.rn
    b0 = shared.b0
    df0tr = shared.df0tr
    SC0tr = shared.SC0tr
    S0tr = shared.S0tr
    P0tr = shared.P0tr
    psi = shared.psi
    save_level = shared.save_level

    A = shared.transition.copy()
    Q = shared.process_cov.copy()
    S0 = shared.initial_state.copy()
    P0 = shared.initial_covariance.copy()

    notrend = np.where(SC0tr < 1e-6)[0]

    Nkeep = draws // thin if collect else 0
    T = y.shape[0]
    states = np.empty((T, rn, Nkeep)) if collect else None
    trends = np.empty((T, n, Nkeep)) if collect else None
    trends_real = np.empty((T, n, Nkeep)) if collect else None
    cycles = np.empty((T, n, Nkeep)) if collect else None

    if collect and save_level.lower() == "full":
        loglik_store = np.empty(Nkeep)
        ss0_store = np.empty((r, Nkeep))
        AA = np.empty((rn, rn, Nkeep))
        QQ = np.empty((rn, rn, Nkeep))
        CC = np.empty((n, rn, Nkeep))
        RR = np.empty((n, n, Nkeep))
        p_acc = np.full(draws, np.nan)
    else:
        loglik_store = ss0_store = AA = QQ = CC = RR = p_acc = None

    keep_idx = 0
    start_time = time.perf_counter()

    for draw_idx in range(draws):
        kf = kalman_filter(
            y=y,
            measurement=C,
            measurement_cov=shared.measurement_cov,
            transition=A,
            transition_cov=Q,
            initial_state=S0,
            initial_covariance=P0,
        )
        loglik = kf.log_likelihood

        if collect and save_level.lower() == "full" and notrend.size:
            proposal = S0.copy()
            proposal[notrend] = S0[notrend] + rng.standard_normal(notrend.size)
            kf_new = kalman_filter(
                y=y,
                measurement=C,
                measurement_cov=shared.measurement_cov,
                transition=A,
                transition_cov=Q,
                initial_state=proposal,
                initial_covariance=P0,
            )
            loglik_new = kf_new.log_likelihood
            acc_prob = min(math.exp(loglik_new - loglik), 1.0)
            if rng.random() <= acc_prob:
                S0 = proposal
                kf = kf_new
                loglik = loglik_new
            p_acc[draw_idx] = acc_prob

        kc = kalman_smoother_draw(kf, rng)

        cyc_states = kc.states[:, r : r + n]
        initial_rows = []
        for lag in range(p):
            start = r + lag * n
            stop = start + n
            initial_rows.append(kc.initial_state[start:stop])
        if initial_rows:
            Ycyc = np.vstack([np.array(initial_rows), cyc_states])
        else:
            Ycyc = cyc_states

        beta, sigma = bvar(
            Ycyc,
            p,
            b0,
            psi,
            lam=0.2,
            draw=1,
            rng=rng,
        )
        A[r : r + n, r:] = beta.T
        Q[r : r + n, r : r + n] = sigma

        Ytr = np.vstack([kc.initial_state[:r], kc.states[:, :r]])
        diff_tr = np.diff(Ytr, axis=0)
        SCtr = covariance_draw(diff_tr, int(df0tr), np.diag(SC0tr), rng)
        Q[:r, :r] = SCtr

        Ac = A[r:, r:]
        Qc = Q[r:, r:]
        try:
            P0_cyc = solve_discrete_lyapunov(Ac, Qc)
        except np.linalg.LinAlgError:
            I = np.eye(Ac.size)
            K = I - np.kron(Ac, Ac)
            vecP, *_ = np.linalg.lstsq(K, Qc.reshape(-1, order="F"), rcond=None)
            P0_cyc = vecP.reshape(Ac.shape, order="F")
        P0[r:, r:] = P0_cyc

        if collect and (draw_idx + 1) % thin == 0:
            states[:, :, keep_idx] = kc.states
            trends[:, :, keep_idx] = kc.states[:, :r] @ C[:, :r].T
            if r > 1:
                trends_real[:, :, keep_idx] = kc.states[:, 1:r] @ C[:, 1:r].T
            else:
                trends_real[:, :, keep_idx] = trends[:, :, keep_idx]
            cycles[:, :, keep_idx] = cyc_states

            if save_level.lower() == "full":
                loglik_store[keep_idx] = loglik
                ss0_store[:, keep_idx] = S0[:r]
                AA[:, :, keep_idx] = A
                QQ[:, :, keep_idx] = Q
                CC[:, :, keep_idx] = C
                RR[:, :, keep_idx] = shared.measurement_cov

            keep_idx += 1

        if collect and (draw_idx + 1) % 1000 == 0:
            elapsed = time.perf_counter() - start_time
            print(
                f"[chain {chain_id}] {draw_idx + 1}/{draws}, elapsed {elapsed:.1f}s, kept {keep_idx}"
            )

    elapsed_total = time.perf_counter() - start_time

    if collect and keep_idx != Nkeep:
        states = states[:, :, :keep_idx]
        trends = trends[:, :, :keep_idx]
        trends_real = trends_real[:, :, :keep_idx]
        cycles = cycles[:, :, :keep_idx]
        if save_level.lower() == "full":
            loglik_store = loglik_store[:keep_idx]
            ss0_store = ss0_store[:, :keep_idx]
            AA = AA[:, :, :keep_idx]
            QQ = QQ[:, :, :keep_idx]
            CC = CC[:, :, :keep_idx]
            RR = RR[:, :, :keep_idx]

    if collect and save_path is not None:
        out_dict: Dict[str, NDArray[np.float64]] = {
            "CommonTrends": states[:, :r, :],
            "Trends": trends,
            "TrendsReal": trends_real,
            "Cycles": cycles,
            "SC0tr": SC0tr,
            "S0tr": S0tr,
            "P0tr": P0tr,
            "df0tr": np.array([df0tr]),
            "Psi": psi,
            "Time": shared.time_serial[:, None],
            "Y": shared.Y,
            "y": shared.y,
            "Mnem": np.array(shared.mnemonics, dtype=object),
            "THIN": np.array([thin]),
            "Ndraws": np.array([draws]),
        }
        if save_level.lower() == "full":
            out_dict.update(
                {
                    "AA": AA,
                    "QQ": QQ,
                    "CC": CC,
                    "RR": RR,
                    "LogLik": loglik_store,
                    "SS0": ss0_store,
                    "P_acc": p_acc,
                }
            )
        np.savez_compressed(str(save_path), **out_dict)

    return chain_id, elapsed_total


def combine_chains(
    chain_paths: Sequence[Path],
    save_level: str,
) -> Dict[str, NDArray[np.float64]]:
    combined: Dict[str, NDArray[np.float64]] = {}
    collect_full = save_level.lower() == "full"

    for path in chain_paths:
        with np.load(str(path), allow_pickle=True) as data:
            for key in ["CommonTrends", "Trends", "TrendsReal", "Cycles"]:
                if key not in data:
                    continue
                arr = data[key]
                if arr.ndim == 2:
                    arr = arr[:, :, None]
                if key in combined:
                    combined[key] = np.concatenate([combined[key], arr], axis=2)
                else:
                    combined[key] = arr
            if collect_full:
                for key in ["AA", "QQ", "CC", "RR"]:
                    if key not in data:
                        continue
                    arr = data[key]
                    if arr.ndim == 2:
                        arr = arr[:, :, None]
                    if key in combined:
                        combined[key] = np.concatenate([combined[key], arr], axis=2)
                    else:
                        combined[key] = arr
                for key in ["LogLik", "SS0", "P_acc"]:
                    if key not in data:
                        continue
                    arr = np.atleast_1d(data[key])
                    if key in combined:
                        combined[key] = np.concatenate(
                            [combined[key], arr], axis=arr.ndim - 1
                        )
                    else:
                        combined[key] = arr

    return combined


def discard_burn_in(array: NDArray[np.float64], discard: int) -> NDArray[np.float64]:
    if array.ndim < 3:
        raise ValueError("Expected (T, k, draws) array")
    return array[:, :, discard:]


def compute_quantiles(samples: NDArray[np.float64], quantiles: Sequence[float]) -> NDArray[np.float64]:
    sorted_samples = np.sort(samples, axis=2)
    M = sorted_samples.shape[2]
    indices = [min(max(int(math.ceil(q * M)) - 1, 0), M - 1) for q in quantiles]
    return sorted_samples[:, :, indices]


def post_process(
    samples: Dict[str, NDArray[np.float64]],
    shared: SharedState,
    data_dir: Path,
    fig_dir: Path,
) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    quant = [0.025, 0.16, 0.50, 0.84, 0.975]

    Mkeep = samples["CommonTrends"].shape[2]
    discard = math.ceil(Mkeep / 2)

    CommonTrends = discard_burn_in(samples["CommonTrends"], discard)
    Trends = discard_burn_in(samples["Trends"], discard)
    TrendsReal = discard_burn_in(samples["TrendsReal"], discard)
    Cycles = discard_burn_in(samples["Cycles"], discard)

    qCommonTrends = compute_quantiles(CommonTrends, quant)
    qTrends = compute_quantiles(Trends, quant)
    qTrendsReal = compute_quantiles(TrendsReal, quant)
    qCycles = compute_quantiles(Cycles, quant)

    Pi_bar = CommonTrends[:, 0, :]
    R_bar = CommonTrends[:, 1, :]
    Ts_bar = CommonTrends[:, 2, :]
    nominal_bar = Pi_bar + R_bar

    qPi_bar = compute_quantiles(Pi_bar[:, None, :], quant)[:, 0, :]
    qR_bar = compute_quantiles(R_bar[:, None, :], quant)[:, 0, :]
    qTs_bar = compute_quantiles(Ts_bar[:, None, :], quant)[:, 0, :]
    qNominal = compute_quantiles(nominal_bar[:, None, :], quant)[:, 0, :]

    time_dt = pd.to_datetime(shared.time_datetimes)
    df = pd.DataFrame(
        {
            "Date": time_dt,
            "Pi_bar_med": qPi_bar[:, 2],
            "Pi_bar_p2_5": qPi_bar[:, 0],
            "Pi_bar_p16": qPi_bar[:, 1],
            "Pi_bar_p84": qPi_bar[:, 3],
            "Pi_bar_p97_5": qPi_bar[:, 4],
            "R_bar_med": qR_bar[:, 2],
            "R_bar_p2_5": qR_bar[:, 0],
            "R_bar_p16": qR_bar[:, 1],
            "R_bar_p84": qR_bar[:, 3],
            "R_bar_p97_5": qR_bar[:, 4],
            "NR_bar_med": qNominal[:, 2],
            "NR_bar_p2_5": qNominal[:, 0],
            "NR_bar_p16": qNominal[:, 1],
            "NR_bar_p84": qNominal[:, 3],
            "NR_bar_p97_5": qNominal[:, 4],
            "Ts_bar_med": qTs_bar[:, 2],
            "Ts_bar_p2_5": qTs_bar[:, 0],
            "Ts_bar_p16": qTs_bar[:, 1],
            "Ts_bar_p84": qTs_bar[:, 3],
            "Ts_bar_p97_5": qTs_bar[:, 4],
        }
    )
    df.to_csv(data_dir / "OutMod1forCharts.csv", index=False)

    def quantile_frame(
        quantile_array: NDArray[np.float64],
        series_names: Sequence[str],
        filename: str,
    ) -> None:
        frames = []
        for idx, name in enumerate(series_names):
            frames.append(
                pd.DataFrame(
                    {
                        "Date": time_dt,
                        "Series": name,
                        "p2_5": quantile_array[:, idx, 0],
                        "p16": quantile_array[:, idx, 1],
                        "p50": quantile_array[:, idx, 2],
                        "p84": quantile_array[:, idx, 3],
                        "p97_5": quantile_array[:, idx, 4],
                    }
                )
            )
        pd.concat(frames, ignore_index=True).to_csv(data_dir / filename, index=False)

    quantile_frame(
        qCommonTrends,
        ["Pi_bar_trend", "R_bar_trend", "Ts_bar_trend"],
        "CommonTrends_quantiles.csv",
    )
    quantile_frame(
        np.stack([qPi_bar, qR_bar, qNominal, qTs_bar], axis=1),
        ["Pi_bar", "R_bar", "NR_bar", "Ts_bar"],
        "KeySeries_quantiles.csv",
    )
    quantile_frame(qTrends, shared.mnemonics, "ObservedTrends_quantiles.csv")
    quantile_frame(qTrendsReal, shared.mnemonics, "RealTrends_quantiles.csv")
    quantile_frame(qCycles, shared.mnemonics, "Cycles_quantiles.csv")

    y = shared.y

    def fig_with_shaded(
        data: NDArray[np.float64],
        title: str,
        overlays: Optional[List[Dict[str, object]]] = None,
        ylim: Optional[Tuple[float, float]] = None,
        subtitle: Optional[str] = None,
        ylabel: str = "Percent, annualized",
    ) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(10, 4))
        plot_states_shaded(time_dt, data, ax=ax)
        median_series = data[:, 2]
        last_value = median_series[-1]
        last_date = time_dt[-1]
        quarter_label = f"{pd.Period(last_date, freq='Q')}"
        if overlays:
            for overlay in overlays:
                series = np.asarray(overlay["y"], dtype=float)
                style = overlay.get("style", "b-")
                kwargs = overlay.get("kwargs", {})
                ax.plot(time_dt, series, style, **kwargs)
        if ylim:
            ax.set_ylim(*ylim)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontweight="bold", fontsize=14, loc="left")
        if subtitle:
            ax.set_title(subtitle, fontsize=11, loc="left", pad=24)
        ax.set_xlabel("Year")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.yaxis.grid(True, linestyle=":", linewidth=0.5, color="0.8")
        ax.annotate(
            f"{quarter_label}  {last_value:.2f}",
            xy=(last_date, last_value),
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

    figs = [
        (
            fig_with_shaded(
                qPi_bar,
                r"Trend inflation $(\pi_t^{\mathrm{trend}})$ estimated from Model 1",
                subtitle=None,
            ),
            "PIbar.png",
        ),
        (
            fig_with_shaded(
                qPi_bar,
                r"Observed inflation $(\pi_t)$ vs. survey inflation expectations $(\pi_t^e)$",
                subtitle=None,
                overlays=[
                    {"y": y[:, 1], "style": "b-", "kwargs": {"linewidth": 2.0}},
                    {"y": y[:, 0], "style": "b:", "kwargs": {"linewidth": 1.0}},
                ],
            ),
            "PIbar_obs.png",
        ),
        (
            fig_with_shaded(
                qR_bar,
                r"Natural real rate $(r_t^*)$ estimated from Model 1 (baseline trendy VAR)",
                subtitle=None,
            ),
            "Rbar.png",
        ),
        (
            fig_with_shaded(
                qNominal,
                r"Nominal neutral short rate $(i_t^* = r_t^* + \pi_t^{\mathrm{trend}})$ implied by Model 1",
                subtitle=None,
            ),
            "NRbar.png",
        ),
        (
            fig_with_shaded(
                qR_bar,
                r"Natural real rate $(r_t^*)$ (Model 1) vs. ex-ante and ex-post real rates",
                subtitle=None,
                overlays=[
                    {
                        "y": y[:, 3] - y[:, 1],
                        "style": "b*-",
                        "kwargs": {"linewidth": 2.0},
                    },
                    {
                        "y": y[:, 2] - y[:, 1],
                        "style": "b:",
                        "kwargs": {"linewidth": 1.0},
                    },
                ],
            ),
            "Rbar_obs.png",
        ),
        (
            fig_with_shaded(
                qTs_bar,
                "Trend credit-spread / convenience-yield proxy used in natural-rate decomposition",
                subtitle=None,
            ),
            "TSbar.png",
        ),
        (
            fig_with_shaded(
                qTs_bar,
                r"Trend credit-spread / convenience-yield proxy vs. observed credit spread $(r^L - r)$",
                subtitle=None,
                overlays=[
                    {
                        "y": y[:, 4] - y[:, 2],
                        "style": "b:",
                        "kwargs": {"linewidth": 1.0},
                    }
                ],
            ),
            "TSbar_obs.png",
        ),
        (
            fig_with_shaded(
                qR_bar,
                r"Trend natural real rate $(r_t^*)$ (Model 1, zoomed scale)",
                subtitle=None,
                ylim=(-0.5, 3.5),
            ),
            "Rscaled.png",
        ),
        (
            fig_with_shaded(
                qTs_bar,
                "Trend credit-spread / convenience-yield proxy (zoomed scale)",
                subtitle=None,
                ylim=(-0.5, 3.5),
            ),
            "TSscaled.png",
        ),
    ]

    for fig, name in figs:
        save_figure(fig, fig_dir / name)
        plt.close(fig)

    print(f"Post-processing complete: data in {data_dir}, figures in {fig_dir}")


def main() -> None:
    RunEstimation = _env_bool("RSTAR_RUN_ESTIMATION", True)
    OutputName = os.getenv("RSTAR_OUTPUT_NAME", "OutputModel1")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    run_output_dir = OUTPUT_ROOT / OutputName
    data_dir = run_output_dir / "Data"
    fig_dir = run_output_dir / "Figures"
    chains_dir = run_output_dir / "chains_out"
    combined_file = data_dir / "combined_draws.npz"

    data_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    chains_dir.mkdir(parents=True, exist_ok=True)

    shared = prepare_data()

    Ndraws = _env_int("RSTAR_NDRAWS", 100_000)
    NCHAINS = max(1, _env_int("RSTAR_NCHAINS", 8))
    THIN = max(1, _env_int("RSTAR_THIN", 10))
    Nbench = max(1, _env_int("RSTAR_NBENCH", 1000))
    draws_per_chain = max(1, Ndraws // NCHAINS)
    if draws_per_chain * NCHAINS != Ndraws:
        print(
            f"Adjusted draws per chain to {draws_per_chain} (requested {Ndraws} total)."
        )

    seeds = np.random.SeedSequence().spawn(NCHAINS)
    chain_seeds = [int(seed.generate_state(1)[0]) for seed in seeds]

    if RunEstimation:
        bench_times = []
        for chain_id, seed in enumerate(chain_seeds, start=1):
            _, duration = run_chain(
                chain_id,
                seed,
                draws=Nbench,
                thin=THIN,
                shared=shared,
                save_path=None,
                collect=False,
            )
            bench_times.append(duration)
            print(
                f"[chain {chain_id}] Benchmark: {Nbench} draws took {duration:.2f} sec ({duration / Nbench:.4f} sec/draw)"
            )

        sec_per_draw = float(np.mean(bench_times) / Nbench)
        serial_hours = sec_per_draw * Ndraws / 3600.0
        parallel_hours = serial_hours / NCHAINS + 0.3
        print("--- Runtime Estimates ---")
        print(f"Serial (100k draws):           ~{serial_hours:.2f} hours")
        print(
            f"Parallel ({NCHAINS} chains, 100k tot): ~{parallel_hours:.2f} hours (incl. overhead)"
        )
        print("-------------------------")

        chain_paths = [
            chains_dir / f"{OutputName}_chain{chain_id:02d}.npz"
            for chain_id in range(1, NCHAINS + 1)
        ]
        args = [
            (
                chain_id,
                chain_seeds[chain_id - 1],
                draws_per_chain,
                THIN,
                shared,
                chain_paths[chain_id - 1],
                True,
            )
            for chain_id in range(1, NCHAINS + 1)
        ]

        if NCHAINS == 1:
            results = [run_chain(*args[0])]
        else:
            with mp.get_context("spawn").Pool(processes=NCHAINS) as pool:
                results = pool.starmap(run_chain, args)

        total_time = sum(duration for _, duration in results)
        print(f"All chains finished; aggregate wall time {total_time:.1f} sec")

        combined = combine_chains(chain_paths, shared.save_level)

        combined = combine_chains(chain_paths, shared.save_level)
        combined_payload = {
            "CommonTrends": combined["CommonTrends"],
            "Trends": combined["Trends"],
            "TrendsReal": combined["TrendsReal"],
            "Cycles": combined["Cycles"],
            "Ndraws": np.array([Ndraws]),
            "Discard": np.array([math.ceil(combined["CommonTrends"].shape[2] / 2)]),
            "SC0tr": shared.SC0tr,
            "S0tr": shared.S0tr,
            "P0tr": shared.P0tr,
            "df0tr": np.array([shared.df0tr]),
            "Psi": shared.psi,
            "Time": shared.time_serial[:, None],
            "Y": shared.Y,
            "y": shared.y,
            "Mnem": np.array(shared.mnemonics, dtype=object),
            "NCHAINS": np.array([NCHAINS]),
            "THIN": np.array([THIN]),
            "draws_per_chain": np.array([draws_per_chain]),
            "Nbench": np.array([Nbench]),
            "bench_times": np.array(bench_times),
        }
        np.savez_compressed(str(combined_file), **combined_payload)
        samples = {
            "CommonTrends": combined["CommonTrends"],
            "Trends": combined["Trends"],
            "TrendsReal": combined["TrendsReal"],
            "Cycles": combined["Cycles"],
        }
    else:
        if not combined_file.exists():
            raise FileNotFoundError(
                f"{combined_file} not found; rerun with RSTAR_RUN_ESTIMATION=1 first"
            )
        with np.load(str(combined_file), allow_pickle=True) as stored:
            samples = {
                "CommonTrends": stored["CommonTrends"],
                "Trends": stored["Trends"],
                "TrendsReal": stored["TrendsReal"],
                "Cycles": stored["Cycles"],
            }

    post_process(samples, shared, data_dir, fig_dir)


if __name__ == "__main__":
    main()
