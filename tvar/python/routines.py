from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray


def cholred(matrix: NDArray[np.float64], tol: float = 1e-12) -> NDArray[np.float64]:
    """Return a square-root factor even when the matrix is only semidefinite."""
    values, vectors = np.linalg.eigh(matrix)
    clipped = np.clip(values, tol, None)
    return vectors @ np.diag(np.sqrt(clipped)) @ vectors.T


@dataclass
class KalmanResult:
    log_likelihood: float
    smoothed_states: NDArray[np.float64]
    smoothed_covariances: NDArray[np.float64]
    predicted_states: NDArray[np.float64]
    predicted_covariances: NDArray[np.float64]
    initial_state: NDArray[np.float64]
    initial_covariance: NDArray[np.float64]
    transition: NDArray[np.float64]
    transition_cov: NDArray[np.float64]
    measurement: NDArray[np.float64]
    measurement_cov: NDArray[np.float64]


def kalman_filter(
    y: NDArray[np.float64],
    measurement: NDArray[np.float64],
    measurement_cov: NDArray[np.float64],
    transition: NDArray[np.float64],
    transition_cov: NDArray[np.float64],
    initial_state: NDArray[np.float64],
    initial_covariance: NDArray[np.float64],
) -> KalmanResult:
    """Run a Kalman filter that handles missing observations."""
    T = y.shape[0]
    state_dim = transition.shape[0]

    S_prev = initial_state.copy()
    P_prev = initial_covariance.copy()

    loglik = 0.0
    S = np.full((T, state_dim), np.nan)
    P = np.full((state_dim, state_dim, T), np.nan)
    Sf = np.full((T, state_dim), np.nan)
    Pf = np.full((state_dim, state_dim, T), np.nan)

    for t in range(T):
        S_f = transition @ S_prev
        P_f = transition @ P_prev @ transition.T + transition_cov

        yt = y[t, :]
        obs_mask = ~np.isnan(yt)
        if not np.any(obs_mask):
            S[t, :] = S_f
            P[:, :, t] = P_f
            Sf[t, :] = S_f
            Pf[:, :, t] = P_f
            S_prev = S_f
            P_prev = P_f
            continue

        Ct = measurement[obs_mask, :]
        Rt = measurement_cov[np.ix_(obs_mask, obs_mask)]
        yt_obs = yt[obs_mask][:, None]
        y_forecast = Ct @ S_f[:, None]
        innovation_cov = Ct @ P_f @ Ct.T + Rt
        inv_innovation_cov = np.linalg.inv(innovation_cov)

        gain = P_f @ Ct.T @ inv_innovation_cov
        innovation = yt_obs - y_forecast
        S_t = S_f[:, None] + gain @ innovation
        P_t = P_f - gain @ Ct @ P_f

        sign, logdet = np.linalg.slogdet(innovation_cov)
        if sign <= 0:
            raise np.linalg.LinAlgError("Innovation covariance not positive definite")
        quad = float(innovation.T @ inv_innovation_cov @ innovation)
        loglik += -0.5 * (logdet + quad + obs_mask.sum() * np.log(2.0 * np.pi))

        S[t, :] = S_t.ravel()
        P[:, :, t] = P_t
        Sf[t, :] = S_f
        Pf[:, :, t] = P_f
        S_prev = S_t.ravel()
        P_prev = P_t

    return KalmanResult(
        log_likelihood=loglik,
        smoothed_states=S,
        smoothed_covariances=P,
        predicted_states=Sf,
        predicted_covariances=Pf,
        initial_state=initial_state,
        initial_covariance=initial_covariance,
        transition=transition,
        transition_cov=transition_cov,
        measurement=measurement,
        measurement_cov=measurement_cov,
    )


@dataclass
class KalmanDraw:
    states: NDArray[np.float64]
    initial_state: NDArray[np.float64]


def kalman_smoother_draw(
    kf: KalmanResult, rng: np.random.Generator
) -> KalmanDraw:
    """Simulation smoother (Carter-Kohn) draw of latent states."""
    S = kf.smoothed_states
    P = kf.smoothed_covariances
    Sf = kf.predicted_states
    Pf = kf.predicted_covariances
    A = kf.transition

    T, state_dim = S.shape
    draws = np.zeros_like(S)

    mean = S[-1, :]
    cov = P[:, :, -1]
    draw = mean + cholred(cov).T @ rng.standard_normal(state_dim)
    draws[-1, :] = draw

    for t in range(T - 2, -1, -1):
        Pf_inv = np.linalg.pinv(Pf[:, :, t + 1])
        mean_t = (
            S[t, :]
            + P[:, :, t] @ A.T @ Pf_inv @ (draws[t + 1, :] - Sf[t + 1, :])
        )
        cov_t = P[:, :, t] - P[:, :, t] @ A.T @ Pf_inv @ A @ P[:, :, t]
        draw = mean_t + cholred(cov_t).T @ rng.standard_normal(state_dim)
        draws[t, :] = draw

    Pf0_inv = np.linalg.inv(Pf[:, :, 0])
    mean0 = (
        kf.initial_state
        + kf.initial_covariance
        @ A.T
        @ Pf0_inv
        @ (draws[0, :] - Sf[0, :])
    )
    cov0 = (
        kf.initial_covariance
        - kf.initial_covariance
        @ A.T
        @ Pf0_inv
        @ A
        @ kf.initial_covariance
    )
    draw0 = mean0 + cholred(cov0).T @ rng.standard_normal(state_dim)

    return KalmanDraw(states=draws, initial_state=draw0)


def lag_matrix(data: NDArray[np.float64], lag: int) -> NDArray[np.float64]:
    """Create a lagged design matrix."""
    if lag <= 0:
        raise ValueError("Lag must be positive")
    T, n = data.shape
    lags = []
    for i in range(1, lag + 1):
        pad = np.full((i, n), np.nan)
        lags.append(np.vstack((pad, data[:-i, :])))
    return np.hstack(lags)


def bvar(
    y: NDArray[np.float64],
    lags: int,
    b_prior: NDArray[np.float64],
    psi: NDArray[np.float64],
    lam: float,
    draw: int,
    rng: np.random.Generator,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Minnesota prior Bayesian VAR with optional draw from the posterior."""
    x = lag_matrix(y, lags)[lags:, :]
    y_trimmed = y[lags:, :]
    T = y_trimmed.shape[0]
    n = y.shape[1]
    k = n * lags

    d = n + 2
    alpha = 2.0
    omega = np.zeros(k)
    for i in range(lags):
        start = i * n
        stop = (i + 1) * n
        scaling = (d - n - 1) * (lam ** 2) * (1.0 / ((i + 1) ** alpha))
        omega[start:stop] = scaling / psi

    prior_precision = np.diag(1.0 / omega)
    xTx = x.T @ x
    right = x.T @ y_trimmed + prior_precision @ b_prior
    beta_hat = np.linalg.solve(xTx + prior_precision, right)

    residuals = y_trimmed - x @ beta_hat
    diff = beta_hat - b_prior
    sigma_hat = (
        residuals.T @ residuals
        + np.diag(psi)
        + diff.T @ prior_precision @ diff
    ) / (T + d + n + 1)

    if draw != 1:
        return beta_hat, sigma_hat

    max_iter = 1000
    for _ in range(max_iter):
        eigvals, eigvecs = np.linalg.eigh(sigma_hat * (T + d + n + 1))
        inv_eigs = np.clip(np.abs(eigvals), 1e-12, None)
        sinv = eigvecs @ np.diag(1.0 / inv_eigs) @ eigvecs.T
        eta = rng.multivariate_normal(
            mean=np.zeros(n), cov=sinv, size=T + d
        )
        sigma_draw = np.linalg.inv(eta.T @ eta)
        sigma_draw = (sigma_draw + sigma_draw.T) / 2.0
        chol_sigma = cholred(sigma_draw)
        inv_information = np.linalg.inv(xTx + prior_precision)
        chol_inv_information = cholred(inv_information)
        noise = rng.standard_normal(size=beta_hat.shape)
        beta_draw = beta_hat + chol_inv_information.T @ noise @ chol_sigma

        companion = np.zeros((n * lags, n * lags))
        companion[:n, :] = beta_draw.T
        if lags > 1:
            companion[n:, :-n] = np.eye(n * (lags - 1))
        if np.all(np.abs(np.linalg.eigvals(companion)) < 1.0):
            return beta_draw, sigma_draw

    raise RuntimeError("Failed to draw stationary VAR parameters within limit")


def covariance_draw(
    z: NDArray[np.float64],
    df0: int,
    mS0: NDArray[np.float64],
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """Draw covariance matrix following CovarianceDraw.m."""
    n = z.shape[1]
    Sc0 = mS0 * (df0 + n + 1)
    S = z.T @ z + Sc0

    eigvals, eigvecs = np.linalg.eigh(S)
    inv_sqrt = np.diag(1.0 / np.sqrt(np.clip(np.abs(eigvals), 1e-12, None)))
    eta = rng.standard_normal(size=(z.shape[0] + df0, n)) @ inv_sqrt @ eigvecs.T
    sigma = np.linalg.inv(eta.T @ eta)
    return (sigma + sigma.T) / 2.0


def plot_states_shaded(
    time: NDArray[np.float64] | ArrayLike,
    quantiles: NDArray[np.float64],
    ax: Optional[plt.Axes] = None,
    color: Optional[Tuple[float, float, float]] = None,
    transparency: float = 0.5,
) -> plt.Axes:
    """Replicate PlotStatesShaded.m in Matplotlib."""
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    base_color = color or (0.8, 0.8, 0.8)
    time_arr = np.asarray(time)
    if np.issubdtype(time_arr.dtype, np.datetime64):
        time_nums = mdates.date2num(pd.to_datetime(time_arr))
    else:
        time_nums = time_arr.astype(float)
    q = np.asarray(quantiles)

    ax.fill_between(
        time_nums,
        q[:, 0],
        q[:, 4],
        color=np.multiply(base_color, 0.75),
        alpha=transparency,
        linewidth=0,
    )
    ax.fill_between(
        time_nums,
        q[:, 1],
        q[:, 3],
        color=np.multiply(base_color, 0.5),
        alpha=transparency,
        linewidth=0,
    )
    line_color = tuple(np.multiply(base_color, 0.65))
    ax.plot(time_nums, q[:, 2], linestyle="--", color=line_color, linewidth=1.5)
    ax.plot(time_nums, np.zeros_like(time_nums), color="k", linewidth=0.25)

    if np.issubdtype(time_arr.dtype, np.datetime64):
        ax.xaxis.set_major_locator(mdates.YearLocator(base=10))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.figure.autofmt_xdate()
    ax.set_xlim(time_nums[0], time_nums[-1])
    ax.grid(False)
    fig.set_facecolor("w")
    return ax


def save_pdf(fig: plt.Figure, output: Path | str) -> None:
    """Save a Matplotlib figure to PDF with tight bounds."""
    fig.savefig(output, format="pdf", bbox_inches="tight")
