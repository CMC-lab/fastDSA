"""
q_star.py

Data-driven q_star-SVHT eigendelay selector for fastDSA.

This implementation matches the final manuscript/chats algorithm:

1. Standardize a representative scalar time series.
2. Estimate an initial Hankel window q0 from:
   - integrated autocorrelation time,
   - a reliable dominant period from a Hann-windowed periodogram,
   - a conservative N^(1/3) fallback.
3. Build a pilot Hankel matrix H(q0).
4. Estimate the pilot rank r0 using SVHT.
5. Measure rank pressure / saturation p0 = r0 / q0.
6. Define the data-driven final Hankel window

       q_star = min(ceil(q0 * (1 + p0**2)), floor((N + 1) / 2)).

7. Build the final Hankel matrix H(q_star) and estimate r_star by SVHT.

There is intentionally no holdout NRMSE, no BIC, no grid search over q, and
no grid search over r inside this selector.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import math
import numpy as np
import torch

ArrayLike = Union[np.ndarray, torch.Tensor]


# =============================================================================
# Basic conversion utilities
# =============================================================================

def as_numpy(X: ArrayLike) -> np.ndarray:
    """Convert NumPy array or torch tensor to a float NumPy array."""
    if isinstance(X, torch.Tensor):
        X = X.detach().cpu().numpy()
    return np.asarray(X, dtype=float)


def infer_time_axis(X: np.ndarray, time_axis: Optional[int] = None) -> int:
    """
    Infer the time axis for q_star estimation.

    Recommended explicit settings:
        - time_axis=0 for data shaped (time, features)
        - time_axis=1 for data shaped (trials, time, features)

    If not provided:
        - 1D: axis 0
        - 2D: larger axis
        - >=3D: axis 1, common for (trials, time, features)
    """
    if time_axis is not None:
        axis = int(time_axis)
        if axis < 0:
            axis += X.ndim
        if axis < 0 or axis >= X.ndim:
            raise ValueError(f"time_axis={time_axis} is invalid for shape {X.shape}")
        return axis

    if X.ndim == 1:
        return 0
    if X.ndim == 2:
        return int(np.argmax(X.shape))
    return 1


def standardize_signal(x: ArrayLike, detrend: str = "mean") -> np.ndarray:
    """Return a finite, standardized 1D signal."""
    z = as_numpy(x).ravel()
    z = z[np.isfinite(z)]

    if z.size < 8:
        raise ValueError(f"Time series too short for q_star estimation: N={z.size}")

    if detrend == "mean":
        z = z - np.mean(z)
    elif detrend in (None, "none", "None"):
        pass
    else:
        raise ValueError("detrend must be 'mean' or 'none'")

    s = np.std(z)
    if s > 0:
        z = z / s
    return np.asarray(z, dtype=float)


def representative_signal(
    X: ArrayLike,
    time_axis: Optional[int] = None,
    max_channels: int = 128,
) -> np.ndarray:
    """
    Build a representative standardized scalar signal from arbitrary data.

    This scalar signal is used only for the q_star heuristics. The final DMD fit
    still uses the original full data.
    """
    arr = as_numpy(X)

    if arr.ndim == 0:
        raise ValueError("Cannot estimate q_star from a scalar input")

    if arr.ndim == 1:
        sig = arr.astype(float)
    else:
        axis = infer_time_axis(arr, time_axis=time_axis)
        moved = np.moveaxis(arr, axis, 0)
        T = moved.shape[0]
        flat = moved.reshape(T, -1)

        if flat.shape[1] > max_channels:
            idx = np.linspace(0, flat.shape[1] - 1, max_channels).astype(int)
            flat = flat[:, idx]

        flat = flat.astype(float)
        flat = flat - np.nanmean(flat, axis=0, keepdims=True)
        flat = flat / (np.nanstd(flat, axis=0, keepdims=True) + 1e-12)
        sig = np.nanmean(flat, axis=1)

    return standardize_signal(sig, detrend="mean")


# =============================================================================
# Autocorrelation component q_acf
# =============================================================================

def autocorrelation_fft(x: np.ndarray, max_lag: Optional[int] = None) -> np.ndarray:
    """Normalized autocorrelation computed by FFT."""
    z = np.asarray(x, dtype=float).ravel()
    z = z - np.mean(z)
    N = len(z)

    if N < 2 or np.std(z) < 1e-12:
        return np.ones(1, dtype=float)

    if max_lag is None:
        max_lag = N - 1
    max_lag = int(max(0, min(max_lag, N - 1)))

    nfft = 1 << int(np.ceil(np.log2(2 * N - 1)))
    fz = np.fft.rfft(z, n=nfft)
    acf = np.fft.irfft(fz * np.conj(fz), n=nfft)[:N]
    acf = acf / (acf[0] + 1e-12)
    return acf[: max_lag + 1]


# Backward-compatible alias.
def normalized_autocorrelation(sig: np.ndarray, max_lag: Optional[int] = None) -> np.ndarray:
    return autocorrelation_fft(sig, max_lag=max_lag)


def integrated_autocorrelation_time(acf: np.ndarray) -> Tuple[float, int]:
    """
    Integrated autocorrelation time from a normalized ACF.

    Let k0 be the first non-positive autocorrelation lag. Then

        tau_int = 1 + 2 * sum_{k=1}^{k0-1} max(rho(k), 0).
    """
    rho = np.asarray(acf, dtype=float).ravel()

    if rho.size <= 1:
        return 1.0, 1

    non_pos = np.where(rho[1:] <= 0.0)[0]
    if non_pos.size > 0:
        k0 = int(non_pos[0] + 1)
    else:
        k0 = int(rho.size)

    if k0 <= 1:
        tau_int = 1.0
    else:
        tau_int = float(1.0 + 2.0 * np.sum(np.maximum(rho[1:k0], 0.0)))

    return float(tau_int), int(k0)


def integrated_autocorrelation_window(
    sig: np.ndarray,
    max_lag: Optional[int] = None,
) -> Tuple[int, float, int, np.ndarray]:
    """
    Estimate q_acf from integrated autocorrelation time.

        q_acf = ceil(2 * tau_int).
    """
    acf = autocorrelation_fft(sig, max_lag=max_lag)
    tau_int, first_nonpositive_lag = integrated_autocorrelation_time(acf)
    q_acf = int(max(1, math.ceil(2.0 * tau_int)))
    return q_acf, tau_int, first_nonpositive_lag, acf


# Backward-compatible alias used by some earlier drafts.
def autocorrelation_window(sig: np.ndarray, max_lag: Optional[int] = None) -> Tuple[int, float, int, np.ndarray]:
    return integrated_autocorrelation_window(sig, max_lag=max_lag)


# =============================================================================
# Dominant-period component q_period
# =============================================================================

def dominant_period_samples(
    sig: np.ndarray,
    prominence_ratio: float = 8.0,
    min_period: int = 8,
    min_cycles: int = 3,
) -> Tuple[float, float, float, bool]:
    """
    Estimate the dominant period in samples from a Hann-windowed periodogram.

    Reliability criteria:
        - peak / median spectral power >= prominence_ratio
        - period >= min_period samples
        - at least min_cycles cycles fit in the signal

    Returns
    -------
    period_samples : float
    peak_frequency_per_sample : float
    peak_prominence_ratio : float
    reliable : bool
    """
    z = np.asarray(sig, dtype=float).ravel()
    z = z - np.mean(z)
    N = len(z)

    if N < max(8, int(min_period)) or np.std(z) < 1e-12:
        return 0.0, 0.0, 0.0, False

    win = np.hanning(N)
    zw = z * win

    spectrum = np.fft.rfft(zw)
    power = np.abs(spectrum) ** 2
    freqs = np.fft.rfftfreq(N, d=1.0)

    # Ignore DC.
    if power.size <= 2:
        return 0.0, 0.0, 0.0, False

    power = power.copy()
    power[0] = 0.0

    valid = freqs > 0
    idx_valid = np.where(valid)[0]
    if idx_valid.size == 0:
        return 0.0, 0.0, 0.0, False

    k_peak = int(idx_valid[np.argmax(power[idx_valid])])
    peak_power = float(power[k_peak])

    positive_power = power[idx_valid]
    median_power = float(np.median(positive_power[positive_power > 0])) if np.any(positive_power > 0) else 0.0
    peak_ratio = float(peak_power / (median_power + 1e-12))

    f_peak = float(freqs[k_peak])
    if f_peak <= 0:
        return 0.0, 0.0, peak_ratio, False

    period = float(1.0 / f_peak)

    reliable = (
        peak_ratio >= float(prominence_ratio)
        and period >= float(min_period)
        and (N / period) >= float(min_cycles)
    )

    return float(period), float(f_peak), float(peak_ratio), bool(reliable)


# Backward-compatible alias. Returns int period if reliable, otherwise None.
def dominant_period_lag(
    sig: np.ndarray,
    q_min: int = 8,
    q_max: Optional[int] = None,
    prominence_ratio: float = 8.0,
    min_cycles: int = 3,
) -> Optional[int]:
    period, _, _, reliable = dominant_period_samples(
        sig,
        prominence_ratio=prominence_ratio,
        min_period=max(8, int(q_min)),
        min_cycles=min_cycles,
    )
    if not reliable:
        return None
    q = int(math.ceil(period))
    if q_max is not None:
        q = min(q, int(q_max))
    return q


# =============================================================================
# Hankel/SVHT utilities
# =============================================================================

def q_min_default(N: int) -> int:
    """Reasonable lower bound for q when the caller does not provide q_min."""
    return 16 if int(N) < 256 else 32


def q_geom_for_N(N: int) -> int:
    """Geometrically admissible maximum window length floor((N + 1) / 2)."""
    return int(max(2, (int(N) + 1) // 2))


def hankel_matrix(x: np.ndarray, q: int) -> np.ndarray:
    """
    Construct scalar Hankel trajectory matrix H_q with q rows.

    H_q[:, j] = [x_j, x_{j+1}, ..., x_{j+q-1}]^T.
    """
    z = np.asarray(x, dtype=float).ravel()
    N = len(z)
    q = int(q)
    if q < 1:
        raise ValueError("q must be positive")
    if q > N:
        raise ValueError(f"q={q} exceeds signal length N={N}")
    L = N - q + 1
    if L < 2:
        raise ValueError(f"Too few Hankel columns for q={q}, N={N}")

    # Use sliding_window_view when available; copy to keep downstream SVD safe.
    try:
        H = np.lib.stride_tricks.sliding_window_view(z, q).T.copy()
    except Exception:
        H = np.vstack([z[i : i + L] for i in range(q)])
    return np.asarray(H, dtype=float)


def hankel_pair_view(x: np.ndarray, q: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return one-step shifted scalar Hankel pair (H0, H1).

    This is included for backward compatibility with earlier helper code.
    """
    H = hankel_matrix(x, q)
    if H.shape[1] < 2:
        raise ValueError(f"Too few Hankel columns for shifted pair: shape={H.shape}")
    return H[:, :-1], H[:, 1:]


def omega_beta(beta: float) -> float:
    """Gavish-Donoho cubic approximation for SVHT."""
    beta = float(beta)
    return float(0.56 * beta**3 - 0.95 * beta**2 + 1.82 * beta + 1.43)


def svht_threshold_from_shape(shape: Tuple[int, int], sv: np.ndarray) -> float:
    """SVHT threshold from matrix shape and singular values."""
    m, n = int(shape[0]), int(shape[1])
    beta = min(m, n) / max(m, n)
    return float(omega_beta(beta) * np.median(np.asarray(sv, dtype=float)))


def svht_threshold(X: np.ndarray, sv: Optional[np.ndarray] = None) -> float:
    """Singular Value Hard Thresholding threshold for matrix X."""
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"svht_threshold expects a 2D matrix; got {X.shape}")
    if sv is None:
        sv = np.linalg.svd(X, compute_uv=False, full_matrices=False)
    return svht_threshold_from_shape(X.shape, sv)


def svht_rank(
    X_or_shape: Union[np.ndarray, Tuple[int, int]],
    sv: Optional[np.ndarray] = None,
    min_rank: int = 1,
    max_rank: Optional[int] = None,
) -> Tuple[int, float, np.ndarray]:
    """
    Estimate rank by SVHT.

    Parameters
    ----------
    X_or_shape:
        Either a 2D matrix or the matrix shape. If a shape is supplied, `sv`
        must also be supplied.
    sv:
        Optional precomputed singular values.
    """
    if isinstance(X_or_shape, tuple):
        shape = (int(X_or_shape[0]), int(X_or_shape[1]))
        if sv is None:
            raise ValueError("sv must be provided when X_or_shape is a shape")
        sv_arr = np.asarray(sv, dtype=float)
    else:
        X = np.asarray(X_or_shape, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"svht_rank expects a 2D matrix; got {X.shape}")
        shape = X.shape
        sv_arr = np.linalg.svd(X, compute_uv=False, full_matrices=False) if sv is None else np.asarray(sv, dtype=float)

    tau = svht_threshold_from_shape(shape, sv_arr)
    r = int(np.sum(sv_arr > tau))

    if max_rank is None:
        max_rank = len(sv_arr)

    r = max(int(min_rank), r)
    r = min(int(max_rank), r)

    return int(r), float(tau), sv_arr


def evaluate_hankel_svht(
    sig: np.ndarray,
    q: int,
    min_rank: int = 1,
    max_rank_cap: Optional[int] = None,
) -> Dict[str, Any]:
    """Build H_q and estimate SVHT rank/energy/saturation."""
    H = hankel_matrix(sig, q)
    sv = np.linalg.svd(H, compute_uv=False, full_matrices=False)

    max_rank = len(sv)
    if max_rank_cap is not None:
        max_rank = min(max_rank, int(max_rank_cap))

    r, tau, sv = svht_rank(H.shape, sv=sv, min_rank=min_rank, max_rank=max_rank)
    p = float(r) / float(max(int(q), 1))
    energy = float(np.sum(sv[:r] ** 2) / (np.sum(sv ** 2) + 1e-12))

    return {
        "q": int(q),
        "r": int(r),
        "p_saturation": float(p),
        "SVHT_threshold": float(tau),
        "energy_retained": float(energy),
        "embedded_rows": int(H.shape[0]),
        "embedded_cols": int(H.shape[1]),
        "n_singular_values": int(len(sv)),
        "n_above_SVHT": int(np.sum(sv > tau)),
    }


# Backward-compatible name.
def evaluate_pilot_q(
    X: ArrayLike,
    q: int,
    delay_interval: int = 1,
    min_rank: int = 1,
    max_rank_cap: Optional[int] = None,
    device: Union[str, torch.device] = "cpu",
) -> Dict[str, Any]:
    sig = representative_signal(X, time_axis=None)
    # delay_interval is intentionally ignored for q_star, which uses consecutive samples.
    return evaluate_hankel_svht(sig, q=q, min_rank=min_rank, max_rank_cap=max_rank_cap)


# =============================================================================
# q0 and q_star selection
# =============================================================================

def estimate_q0(
    X: ArrayLike,
    q_min: Optional[int] = None,
    c_prop: float = 2.0,
    q_prop_exponent: float = 1.0 / 3.0,
    max_acf_lag_fraction: float = 0.25,
    period_prominence_ratio: float = 8.0,
    dominant_period_min_cycles: int = 3,
    min_period: int = 8,
    time_axis: Optional[int] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Estimate initial q0 from autocorrelation, dominant period, and N^(1/3)."""
    sig = representative_signal(X, time_axis=time_axis)
    N = int(len(sig))

    q_min_eff = q_min_default(N) if q_min is None else int(q_min)
    q_geom = q_geom_for_N(N)

    max_acf_lag = int(max(8, min(math.floor(float(max_acf_lag_fraction) * N), N - 1)))
    q_acf, tau_int, first_nonpositive_lag, acf = integrated_autocorrelation_window(
        sig,
        max_lag=max_acf_lag,
    )

    P_dom, f_peak, peak_prominence, period_reliable = dominant_period_samples(
        sig,
        prominence_ratio=float(period_prominence_ratio),
        min_period=int(min_period),
        min_cycles=int(dominant_period_min_cycles),
    )
    q_period = int(math.ceil(P_dom)) if period_reliable else 0

    q_prop = int(math.ceil(float(c_prop) * (N ** float(q_prop_exponent))))

    q_pre = int(max(q_min_eff, q_acf, q_period, q_prop))
    q0 = int(min(max(q_pre, q_min_eff), q_geom))

    out = {
        "N": int(N),
        "q_min": int(q_min_eff),
        "q_geom": int(q_geom),
        "q_acf": int(q_acf),
        "q_period": int(q_period),
        "q_prop": int(q_prop),
        "q_pre": int(q_pre),
        "q0": int(q0),
        "tau_int_samples": float(tau_int),
        "first_nonpositive_lag": int(first_nonpositive_lag),
        "max_acf_lag": int(max_acf_lag),
        "dominant_period_samples": float(P_dom),
        "peak_frequency_per_sample": float(f_peak),
        "peak_prominence_ratio": float(peak_prominence),
        "period_reliable": bool(period_reliable),
    }

    if verbose:
        print(
            "[q_star] q0 estimate:",
            {k: out[k] for k in ["N", "q_acf", "q_period", "q_prop", "q0", "q_geom"]},
        )

    return out


def data_driven_q_hard_from_svht(
    x: ArrayLike,
    q0: int,
    min_rank: int = 1,
    max_rank_cap: Optional[int] = None,
    time_axis: Optional[int] = None,
) -> Tuple[int, Dict[str, Any]]:
    """
    Data-driven replacement for fixed q_hard.

    Build a pilot Hankel matrix at q0, estimate the SVHT rank r0, compute
    p0 = r0/q0, and set

        q_hard_data = min(ceil(q0 * (1 + p0**2)), floor((N + 1) / 2)).
    """
    sig = representative_signal(x, time_axis=time_axis)
    N = int(len(sig))
    q_geom = q_geom_for_N(N)
    q0 = int(min(max(2, int(q0)), q_geom))

    pilot = evaluate_hankel_svht(sig, q=q0, min_rank=min_rank, max_rank_cap=max_rank_cap)
    r0 = int(pilot["r"])
    p0 = float(r0) / float(max(q0, 1))

    q_hard_data = int(math.ceil(q0 * (1.0 + p0**2)))
    q_hard_data = int(min(q_hard_data, q_geom))

    info = {
        "q0": int(q0),
        "r0_svht": int(r0),
        "rank_pressure": float(p0),
        "p0": float(p0),
        "SVHT_threshold_pilot": float(pilot["SVHT_threshold"]),
        "energy_retained_pilot": float(pilot["energy_retained"]),
        "q_hard_data": int(q_hard_data),
        "q_geom": int(q_geom),
        "pilot_embedded_rows": int(pilot["embedded_rows"]),
        "pilot_embedded_cols": int(pilot["embedded_cols"]),
        "pilot_n_singular_values": int(pilot["n_singular_values"]),
        "pilot_n_above_SVHT": int(pilot["n_above_SVHT"]),
    }

    return int(q_hard_data), info


def select_q_star_for_dataset(
    X: ArrayLike,
    delay_interval: int = 1,
    q_min: Optional[int] = None,
    q_max_abs: Optional[int] = None,  # kept for API compatibility; intentionally unused in final rule
    c_sqrtN: Optional[float] = None,  # kept for API compatibility; intentionally unused in final rule
    max_fraction: Optional[float] = None,  # kept for API compatibility; intentionally unused in final rule
    c_n13: Optional[float] = None,  # backward-compatible name for c_prop
    c_prop: float = 2.0,
    q_prop_exponent: float = 1.0 / 3.0,
    max_acf_lag_fraction: float = 0.25,
    period_prominence_ratio: float = 8.0,
    dominant_period_min_cycles: int = 3,
    min_period: int = 8,
    train_ratio: Optional[float] = None,  # kept for API compatibility; intentionally unused
    min_rank: int = 1,
    max_rank_cap: Optional[int] = None,
    acf_threshold: Optional[float] = None,  # kept for API compatibility; intentionally unused
    saturation_threshold: Optional[float] = None,  # kept for API compatibility; intentionally unused
    expand_factor: Optional[float] = None,  # kept for API compatibility; intentionally unused
    max_expansions: Optional[int] = None,  # kept for API compatibility; intentionally unused
    time_axis: Optional[int] = None,
    device: Union[str, torch.device] = "cpu",  # kept for API compatibility; not used
    verbose: bool = False,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Select q_star and r_star without holdout error or grid search.

    The final rule is one-step and data-driven:

        q0 = clip(max(q_min, q_acf, q_period, q_prop), q_min, q_geom)
        p0 = r0/q0, where r0 is SVHT rank of H(q0)
        q_star = min(ceil(q0 * (1 + p0**2)), q_geom)

    Returns
    -------
    chosen : dict
        q_star/r_star and diagnostics.
    diagnostics : list[dict]
        Two rows: pilot-q diagnostics and final-q diagnostics.
    """
    if c_n13 is not None:
        c_prop = float(c_n13)

    sig = representative_signal(X, time_axis=time_axis)

    q0_info = estimate_q0(
        sig,
        q_min=q_min,
        c_prop=float(c_prop),
        q_prop_exponent=float(q_prop_exponent),
        max_acf_lag_fraction=float(max_acf_lag_fraction),
        period_prominence_ratio=float(period_prominence_ratio),
        dominant_period_min_cycles=int(dominant_period_min_cycles),
        min_period=int(min_period),
        time_axis=0,
        verbose=verbose,
    )

    q0 = int(q0_info["q0"])
    q_star, hard_info = data_driven_q_hard_from_svht(
        sig,
        q0=q0,
        min_rank=int(min_rank),
        max_rank_cap=max_rank_cap,
        time_axis=0,
    )

    final_eval = evaluate_hankel_svht(
        sig,
        q=q_star,
        min_rank=int(min_rank),
        max_rank_cap=max_rank_cap,
    )

    chosen = {
        **q0_info,
        **hard_info,
        "q_star": int(q_star),
        "r_star": int(final_eval["r"]),
        "SVHT_threshold": float(final_eval["SVHT_threshold"]),
        "energy_retained": float(final_eval["energy_retained"]),
        "final_embedded_rows": int(final_eval["embedded_rows"]),
        "final_embedded_cols": int(final_eval["embedded_cols"]),
        "final_n_singular_values": int(final_eval["n_singular_values"]),
        "final_n_above_SVHT": int(final_eval["n_above_SVHT"]),
        "selection_rule": "q_star = min(ceil(q0 * (1 + (r0/q0)^2)), floor((N + 1)/2)); r_star by SVHT on H(q_star)",
    }

    diagnostics = []
    pilot_row = {
        "stage": "pilot_q0",
        **{k: v for k, v in hard_info.items() if k not in ("q_geom",)},
    }
    diagnostics.append(pilot_row)
    diagnostics.append({"stage": "final_q_star", **final_eval})

    if verbose:
        print(
            f"[q_star] q0={q0}, r0={hard_info['r0_svht']}, "
            f"p0={hard_info['rank_pressure']:.3f}, q_star={q_star}, "
            f"r_star={chosen['r_star']}"
        )

    return chosen, diagnostics


# Backward-compatible alternate name from the recovered notes.
def acp_choose_q_star_data_driven_hardcap(
    x: ArrayLike,
    q_prop_constant: float = 2.0,
    q_prop_exponent: float = 1.0 / 3.0,
    max_acf_lag_fraction: float = 0.25,
    period_prominence_ratio: float = 8.0,
    dominant_period_min_cycles: int = 3,
    train_ratio: float = 0.7,
    min_rank: int = 1,
) -> Tuple[int, Dict[str, Any]]:
    chosen, _ = select_q_star_for_dataset(
        x,
        c_prop=q_prop_constant,
        q_prop_exponent=q_prop_exponent,
        max_acf_lag_fraction=max_acf_lag_fraction,
        period_prominence_ratio=period_prominence_ratio,
        dominant_period_min_cycles=dominant_period_min_cycles,
        train_ratio=train_ratio,
        min_rank=min_rank,
    )
    return int(chosen["q_star"]), chosen


def _median_int(values: Sequence[int]) -> int:
    return int(round(float(np.median(np.asarray(values, dtype=float)))))


def select_q_star_for_collection(
    data: Sequence[Sequence[ArrayLike]],
    delay_interval: int = 1,
    q_min: Optional[int] = None,
    c_prop: float = 2.0,
    q_prop_exponent: float = 1.0 / 3.0,
    max_acf_lag_fraction: float = 0.25,
    period_prominence_ratio: float = 8.0,
    dominant_period_min_cycles: int = 3,
    min_period: int = 8,
    min_rank: int = 1,
    max_rank_cap: Optional[int] = None,
    shared_q: bool = True,
    shared_q_strategy: str = "median",
    rank_strategy: str = "max",
    time_axis: Optional[int] = None,
    device: Union[str, torch.device] = "cpu",
    verbose: bool = False,
    **legacy_kwargs: Any,
) -> Dict[str, Any]:
    """
    Select q/r for a nested fastDSA data collection.

    Parameters
    ----------
    data:
        Nested list with shape [data_block][data_index].
    shared_q:
        If True, replace per-dataset q values by one shared q. This is usually
        cleaner for operator comparison.
    shared_q_strategy:
        'median', 'max', or 'min'. Used only when shared_q=True.
    rank_strategy:
        'max' or 'median'. A shared rank is recommended because similarity
        comparison is simplest when all operators have the same dimension.

    Notes
    -----
    `legacy_kwargs` are accepted to avoid breaking older callers. They are not
    used by the final q_star rule if they correspond to grid-search/BIC/RMSE
    parameters.
    """
    selected_rows: List[Dict[str, Any]] = []
    diagnostic_rows: List[Dict[str, Any]] = []

    q_nested: List[List[int]] = []
    r_nested: List[List[int]] = []
    delay_nested: List[List[int]] = []

    for i, dat in enumerate(data):
        q_row: List[int] = []
        r_row: List[int] = []
        delay_row: List[int] = []

        for j, X in enumerate(dat):
            chosen, diagnostics = select_q_star_for_dataset(
                X,
                delay_interval=int(delay_interval),
                q_min=q_min,
                c_prop=float(c_prop),
                q_prop_exponent=float(q_prop_exponent),
                max_acf_lag_fraction=float(max_acf_lag_fraction),
                period_prominence_ratio=float(period_prominence_ratio),
                dominant_period_min_cycles=int(dominant_period_min_cycles),
                min_period=int(min_period),
                min_rank=int(min_rank),
                max_rank_cap=max_rank_cap,
                time_axis=time_axis,
                device=device,
                verbose=verbose,
            )

            chosen = chosen.copy()
            chosen["data_block"] = int(i)
            chosen["data_index"] = int(j)
            selected_rows.append(chosen)

            for row in diagnostics:
                row = row.copy()
                row["data_block"] = int(i)
                row["data_index"] = int(j)
                diagnostic_rows.append(row)

            q_row.append(int(chosen["q_star"]))
            r_row.append(int(chosen["r_star"]))
            delay_row.append(int(delay_interval))

        q_nested.append(q_row)
        r_nested.append(r_row)
        delay_nested.append(delay_row)

    if shared_q:
        all_q = [int(row["q_star"]) for row in selected_rows]
        strategy = str(shared_q_strategy).lower()
        if strategy == "max":
            q_global = int(max(all_q))
        elif strategy == "min":
            q_global = int(min(all_q))
        elif strategy == "median":
            q_global = _median_int(all_q)
        else:
            raise ValueError("shared_q_strategy must be 'median', 'max', or 'min'")
        q_nested = [[q_global for _ in row] for row in q_nested]
    else:
        q_global = None

    all_r = [int(row["r_star"]) for row in selected_rows]
    r_strategy = str(rank_strategy).lower()
    if r_strategy == "max":
        r_global = int(max(all_r))
    elif r_strategy == "median":
        r_global = _median_int(all_r)
    else:
        raise ValueError("rank_strategy must be 'max' or 'median'")
    r_nested = [[r_global for _ in row] for row in r_nested]

    return {
        "n_delays": q_nested,
        "delay_interval": delay_nested,
        "rank": r_nested,
        "q_global": q_global,
        "rank_global": r_global,
        "selected_rows": selected_rows,
        "diagnostic_rows": diagnostic_rows,
    }
