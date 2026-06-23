"""
q_star.py

Data-driven q_star-SVHT eigendelay selector for fastDSA.

Algorithm:

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


The collection-level functions are designed to be called from fastDSA.py before
DMD/KernelDMD objects are constructed.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import math
import numpy as np
import torch

try:  # package import
    from fastDSA.dmd import embed_signal_torch
except Exception:  # local relative import
    try:
        from .dmd import embed_signal_torch
    except Exception:  # standalone fallback
        embed_signal_torch = None

ArrayLike = Union[np.ndarray, torch.Tensor]


# =============================================================================
# Basic conversions
# =============================================================================

def as_numpy(X: ArrayLike) -> np.ndarray:
    """Convert NumPy array or torch tensor to a float NumPy array."""
    if isinstance(X, torch.Tensor):
        X = X.detach().cpu().numpy()
    return np.asarray(X, dtype=float)


def as_torch_float(X: ArrayLike, device: Union[str, torch.device] = "cpu") -> torch.Tensor:
    """Convert NumPy array or torch tensor to a float32 torch tensor."""
    if isinstance(X, torch.Tensor):
        return X.to(device=device, dtype=torch.float32)
    return torch.as_tensor(X, dtype=torch.float32, device=device)


def to_2d_numpy(D: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    """
    Convert an embedded matrix to 2D NumPy.

    If the embedding returns more than two dimensions, all leading dimensions are
    flattened into the feature axis and the last dimension is kept as snapshots.
    """
    if isinstance(D, torch.Tensor):
        D_np = D.detach().cpu().numpy()
    else:
        D_np = np.asarray(D)

    D_np = np.asarray(D_np, dtype=float)

    if D_np.ndim == 1:
        D_np = D_np.reshape(1, -1)
    elif D_np.ndim > 2:
        D_np = D_np.reshape(-1, D_np.shape[-1])

    if D_np.ndim != 2:
        raise ValueError(f"Expected a 2D embedded matrix, got shape {D_np.shape}")

    return D_np


# =============================================================================
# Representative scalar signal for q_star heuristics
# =============================================================================

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
    """
    Convert to a standardized 1D signal.

    Parameters
    ----------
    x:
        Input scalar signal.
    detrend:
        Currently supports "mean" or "none".
    """
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
    Build a representative standardized 1D signal from arbitrary data.

    This signal is used only for q0/q_star heuristics. The final DMD fit in
    fastDSA still uses the original full data.
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
# Autocorrelation and period components for q0
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


# Backward-compatible name.
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
    tau_int, k0 = integrated_autocorrelation_time(acf)
    q_acf = int(max(1, math.ceil(2.0 * tau_int)))
    return q_acf, tau_int, k0, acf


# Backward-compatible alias used by earlier drafts.
def autocorrelation_decorrelation_lag(
    sig: np.ndarray,
    max_lag: Optional[int] = None,
    threshold: float = 0.0,
) -> Optional[int]:
    q_acf, _, _, _ = integrated_autocorrelation_window(sig, max_lag=max_lag)
    return int(q_acf)


def dominant_period_samples(
    sig: np.ndarray,
    prominence_ratio: float = 8.0,
    min_cycles: float = 3.0,
    min_period: int = 8,
) -> Tuple[float, float, float, bool]:
    """
    Estimate dominant period from a Hann-windowed periodogram.

    Returns
    -------
    P_dom : float
        Dominant period in samples. np.nan if unreliable.
    f_peak : float
        Peak frequency in cycles/sample. np.nan if unreliable.
    peak_prominence : float
        Peak-to-median spectral power ratio.
    period_reliable : bool
        True if all reliability criteria are satisfied:
            - peak-to-median power ratio >= prominence_ratio,
            - period >= min_period samples,
            - at least min_cycles fit in the time series.
    """
    z = np.asarray(sig, dtype=float).ravel()
    z = z - np.mean(z)
    N = len(z)

    if N < max(8, int(min_period)) or np.std(z) < 1e-12:
        return np.nan, np.nan, 0.0, False

    win = np.hanning(N)
    zw = z * win
    power = np.abs(np.fft.rfft(zw)) ** 2

    if power.size <= 2:
        return np.nan, np.nan, 0.0, False

    power[0] = 0.0  # ignore DC
    k_vals = np.arange(power.size)

    valid = k_vals > 0

    # P = N / k. Require P >= min_period -> k <= N/min_period.
    valid &= k_vals <= max(1, int(math.floor(N / max(1, int(min_period)))))

    # Require at least min_cycles in the observation -> P <= N/min_cycles -> k >= min_cycles.
    valid &= k_vals >= max(1, int(math.ceil(float(min_cycles))))

    idx = np.where(valid)[0]
    if idx.size == 0:
        return np.nan, np.nan, 0.0, False

    local_power = power[idx]
    k_star = int(idx[int(np.argmax(local_power))])
    peak_power = float(power[k_star])
    background = float(np.median(local_power) + 1e-12)
    peak_prominence = peak_power / background

    P_dom = float(N / k_star)
    f_peak = float(k_star / N)

    reliable = (
        np.isfinite(peak_prominence)
        and peak_prominence >= float(prominence_ratio)
        and P_dom >= float(min_period)
        and (N / max(P_dom, 1.0)) >= float(min_cycles)
    )

    if not reliable:
        return float(P_dom), float(f_peak), float(peak_prominence), False

    return float(P_dom), float(f_peak), float(peak_prominence), True


# Backward-compatible wrapper.
def dominant_period_lag(
    sig: np.ndarray,
    q_min: int = 20,
    q_max: Optional[int] = None,
    peak_to_median_threshold: float = 8.0,
    min_period: int = 8,
    min_cycles: float = 3.0,
) -> Optional[int]:
    P_dom, _, _, reliable = dominant_period_samples(
        sig,
        prominence_ratio=peak_to_median_threshold,
        min_cycles=min_cycles,
        min_period=min_period,
    )
    if not reliable:
        return None
    q = int(math.ceil(P_dom))
    if q_max is not None and q > int(q_max):
        return None
    if q < int(min_period):
        return None
    return q


# =============================================================================
# Hankel and SVHT utilities
# =============================================================================

def q_geom_for_N(N: int, delay_interval: int = 1, q_max_abs: Optional[int] = None) -> int:
    """
    Geometrically admissible maximum Hankel window.

    For consecutive delays, q_geom(N)=floor((N+1)/2).
    If delay_interval > 1, the cap is adjusted so that a comparable number of
    snapshots remains after embedding.
    """
    N = int(N)
    d = max(1, int(delay_interval))

    if N <= 3:
        qmax = max(1, N - 1)
    elif d == 1:
        qmax = int(math.floor((N + 1) / 2))
    else:
        # Require approximately: N - (q - 1)d >= q.
        qmax = int(math.floor((N + d) / (d + 1)))
        qmax = max(1, qmax)

    if q_max_abs is not None:
        qmax = min(qmax, int(q_max_abs))

    return int(max(1, qmax))


# Backward-compatible max-q function. The sqrt cap arguments are intentionally
# ignored to match the final data-driven hard-cap manuscript version.
def q_max_for_N(
    N: int,
    q_min: int = 20,
    q_max_abs: Optional[int] = None,
    c_sqrtN: float = 2.0,
    max_fraction: float = 0.5,
    delay_interval: int = 1,
) -> int:
    return q_geom_for_N(N, delay_interval=delay_interval, q_max_abs=q_max_abs)


def hankel_view(x: np.ndarray, q: int) -> np.ndarray:
    """
    Build a consecutive-delay Hankel matrix from a scalar time series.

    H_q has shape (q, N-q+1):
        row i contains x[i : i + N-q+1].
    """
    z = np.asarray(x, dtype=float).ravel()
    N = len(z)
    q = int(q)

    if q < 1:
        raise ValueError("q must be >= 1")
    if q > N:
        raise ValueError(f"q={q} exceeds signal length N={N}")

    L = N - q + 1
    if L < 1:
        raise ValueError(f"No Hankel columns for N={N}, q={q}")

    return np.vstack([z[i : i + L] for i in range(q)])


def hankel_pair_view(x: np.ndarray, q: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build one-step-shifted Hankel matrices H0 and H1.

    H0 uses columns up to x_{N-1}; H1 is shifted forward by one sample.
    Both have shape (q, N-q).
    """
    z = np.asarray(x, dtype=float).ravel()
    N = len(z)
    q = int(q)
    L = N - q

    if q < 1:
        raise ValueError("q must be >= 1")
    if L < 2:
        raise ValueError(f"Too few Hankel snapshots for N={N}, q={q}")

    H0 = np.vstack([z[i : i + L] for i in range(q)])
    H1 = np.vstack([z[i + 1 : i + L + 1] for i in range(q)])
    return H0, H1


def svht_threshold_from_shape(shape: Tuple[int, int], sv: np.ndarray) -> float:
    """SVHT threshold from matrix shape and singular values."""
    m, n = int(shape[0]), int(shape[1])
    beta = min(m, n) / max(m, n)
    omega = 0.56 * beta**3 - 0.95 * beta**2 + 1.82 * beta + 1.43
    return float(omega * np.median(np.asarray(sv, dtype=float)))


def svht_threshold(X: Union[np.ndarray, Tuple[int, int]], sv: Optional[np.ndarray] = None) -> float:
    """
    Singular Value Hard Thresholding threshold.

    Parameters
    ----------
    X:
        Either a matrix or a shape tuple (m, n).
    sv:
        Singular values. Required when X is a shape tuple.
    """
    if isinstance(X, tuple):
        if sv is None:
            raise ValueError("sv must be provided when X is a shape tuple")
        return svht_threshold_from_shape(X, sv)

    X2 = to_2d_numpy(X)
    if sv is None:
        sv = np.linalg.svd(X2, compute_uv=False, full_matrices=False)
    return svht_threshold_from_shape(X2.shape, sv)


def svht_rank(
    X_or_shape: Union[np.ndarray, Tuple[int, int]],
    sv: Optional[np.ndarray] = None,
    min_rank: int = 1,
    max_rank: Optional[int] = None,
) -> Tuple[int, float, np.ndarray]:
    """
    Estimate rank with SVHT.

    Supports both:
        svht_rank(matrix, min_rank=...)
        svht_rank(matrix.shape, singular_values, min_rank=...)
    """
    if isinstance(X_or_shape, tuple):
        if sv is None:
            raise ValueError("sv must be provided when X_or_shape is a shape tuple")
        shape = (int(X_or_shape[0]), int(X_or_shape[1]))
        singular_values = np.asarray(sv, dtype=float)
    else:
        X = to_2d_numpy(X_or_shape)
        shape = X.shape
        if sv is None:
            singular_values = np.linalg.svd(X, compute_uv=False, full_matrices=False)
        else:
            singular_values = np.asarray(sv, dtype=float)

    tau = svht_threshold_from_shape(shape, singular_values)
    r = int(np.sum(singular_values > tau))

    if max_rank is None:
        max_rank = len(singular_values)

    r = max(int(min_rank), r)
    r = min(int(max_rank), r)
    return int(r), float(tau), singular_values


# =============================================================================
# q0 and q_star selection matching the final manuscript text
# =============================================================================

def estimate_q0(
    X: ArrayLike,
    q_min: int = 20,
    q_max_abs: Optional[int] = None,
    c_sqrtN: float = 2.0,
    max_fraction: float = 0.5,
    c_n13: float = 2.0,
    acf_threshold: float = 0.0,
    time_axis: Optional[int] = None,
    delay_interval: int = 1,
    period_peak_to_median_threshold: float = 8.0,
    period_min_samples: int = 8,
    period_min_cycles: float = 3.0,
    max_acf_lag_fraction: float = 0.25,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Estimate initial q0.

        q_pre = max(q_min, q_acf, q_period, q_prop)
        q0    = clip(q_pre, q_min, q_geom(N)).

    q_acf is based on integrated autocorrelation time.
    q_period is used only if the dominant period is reliable.
    q_prop = ceil(c_n13 * N^(1/3)).
    """
    z = representative_signal(X, time_axis=time_axis)
    N = int(len(z))

    q_geom = q_geom_for_N(N, delay_interval=int(delay_interval), q_max_abs=q_max_abs)
    q_min_eff = int(min(max(1, int(q_min)), q_geom))

    # Autocorrelation component.
    max_acf_lag = int(
        max(
            8,
            min(
                math.floor(float(max_acf_lag_fraction) * N),
                N - 1,
                q_geom,
            ),
        )
    )
    q_acf, tau_int, first_nonpositive_lag, _acf = integrated_autocorrelation_window(
        z,
        max_lag=max_acf_lag,
    )

    # Dominant period component.
    P_dom, f_peak, peak_prominence, period_reliable = dominant_period_samples(
        z,
        prominence_ratio=float(period_peak_to_median_threshold),
        min_cycles=float(period_min_cycles),
        min_period=int(period_min_samples),
    )
    q_period = int(math.ceil(P_dom)) if period_reliable else 0
    if q_period > q_geom:
        period_reliable = False
        q_period = 0

    # Data-size fallback.
    q_prop = int(max(1, math.ceil(float(c_n13) * (N ** (1.0 / 3.0)))))

    q_pre = int(max(q_min_eff, int(q_acf), int(q_period), int(q_prop)))
    q0 = int(np.clip(q_pre, q_min_eff, q_geom))

    out = {
        "N": int(N),
        "q_min": int(q_min),
        "q_min_effective": int(q_min_eff),
        "q_geom": int(q_geom),
        "q_max": int(q_geom),  # backward-compatible field
        "q_acf": int(q_acf),
        "tau_int_samples": float(tau_int),
        "tau_int": float(tau_int),  # backward-compatible field
        "first_nonpositive_lag": int(first_nonpositive_lag),
        "acf_k0": int(first_nonpositive_lag),
        "max_acf_lag": int(max_acf_lag),
        "q_period": int(q_period),
        "dominant_period_samples": float(P_dom) if np.isfinite(P_dom) else np.nan,
        "peak_frequency_per_sample": float(f_peak) if np.isfinite(f_peak) else np.nan,
        "peak_prominence_ratio": float(peak_prominence),
        "period_reliable": bool(period_reliable),
        "period_used": bool(q_period > 0),
        "q_prop": int(q_prop),
        "q_n13": int(q_prop),  # backward-compatible field
        "q_pre": int(q_pre),
        "q0": int(q0),
        "q0_initial": int(q0),
        "q0_raw": int(q_pre),
        "period_peak_to_median_threshold": float(period_peak_to_median_threshold),
        "period_min_samples": int(period_min_samples),
        "period_min_cycles": float(period_min_cycles),
        "q_max_abs": None if q_max_abs is None else int(q_max_abs),
        # Accepted but intentionally not used in the final manuscript version.
        "c_sqrtN_unused": float(c_sqrtN),
        "max_fraction_unused": float(max_fraction),
        "acf_threshold_unused": float(acf_threshold),
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
    train_ratio: float = 0.7,
    min_rank: int = 2,
) -> Tuple[int, Dict[str, Any]]:
    """
    Data-driven replacement for a fixed q_hard.

    This function follows the final manuscript/chats rule:

        1. Build a pilot Hankel matrix at q0.
        2. Estimate SVHT rank r0 on the training part of H0.
        3. Compute rank pressure p0 = r0 / q0.
        4. q_hard_data = ceil(q0 * (1 + p0^2)).
        5. Clip by q_geom = floor((N + 1) / 2).

    Returns
    -------
    q_hard_data:
        The data-driven final/admissible Hankel window.
    info:
        Diagnostic information.
    """
    z = standardize_signal(x, detrend="mean")
    N = int(len(z))

    q_geom = int((N + 1) // 2)
    q0 = int(min(max(2, int(q0)), q_geom))

    H0, _H1 = hankel_pair_view(z, q0)
    split = int(float(train_ratio) * H0.shape[1])
    split = int(np.clip(split, 2, H0.shape[1]))
    H0tr = H0[:, :split]

    S = np.linalg.svd(H0tr, compute_uv=False, full_matrices=False)
    r0, threshold, S = svht_rank(H0tr.shape, S, min_rank=int(min_rank))

    p0 = float(r0) / float(max(q0, 1))
    q_hard_data = int(math.ceil(float(q0) * (1.0 + p0**2)))
    q_hard_data = int(min(q_hard_data, q_geom))

    energy = float(np.sum(S[:r0] ** 2) / (np.sum(S**2) + 1e-12))

    info = {
        "q0": int(q0),
        "r0_svht": int(r0),
        "r0": int(r0),
        "rank_pressure": float(p0),
        "p0": float(p0),
        "SVHT_threshold": float(threshold),
        "q_hard_data": int(q_hard_data),
        "q_geom": int(q_geom),
        "n_singular_values": int(len(S)),
        "energy_retained_pilot": float(energy),
        "pilot_hankel_rows": int(H0tr.shape[0]),
        "pilot_hankel_cols": int(H0tr.shape[1]),
    }

    return int(q_hard_data), info


def acp_choose_q_star_data_driven_hardcap(
    x: ArrayLike,
    q_prop_constant: float = 2.0,
    q_prop_exponent: float = 1.0 / 3.0,
    max_acf_lag_fraction: float = 0.25,
    period_prominence_ratio: float = 8.0,
    dominant_period_min_cycles: float = 3.0,
    train_ratio: float = 0.7,
    min_rank: int = 2,
    q_min: Optional[int] = None,
    period_min_samples: int = 8,
) -> Tuple[int, Dict[str, Any]]:
    """
    Fully data-driven q_star selector from the recovered final algorithm.

    This is the compact scalar-signal API matching the pasted chat:
        q0 from integrated ACF, dominant period, and N^(1/3), then
        q_star = min(ceil(q0 * (1 + (r0/q0)^2)), floor((N+1)/2)).
    """
    z = standardize_signal(x, detrend="mean")
    N = int(len(z))

    if q_min is None:
        q_min_use = 16 if N < 256 else 32
    else:
        q_min_use = int(q_min)

    # Reuse estimate_q0 with c_n13 and period settings from this compact API.
    q0_info = estimate_q0(
        z,
        q_min=q_min_use,
        c_n13=float(q_prop_constant),
        period_peak_to_median_threshold=float(period_prominence_ratio),
        period_min_samples=int(period_min_samples),
        period_min_cycles=float(dominant_period_min_cycles),
        max_acf_lag_fraction=float(max_acf_lag_fraction),
    )

    q_star, hard_info = data_driven_q_hard_from_svht(
        z,
        q0=int(q0_info["q0"]),
        train_ratio=float(train_ratio),
        min_rank=int(min_rank),
    )

    info = {
        "q_star": int(q_star),
        **q0_info,
        **hard_info,
    }

    return int(q_star), info


def embed_dataset_to_numpy(
    X: ArrayLike,
    q: int,
    delay_interval: int = 1,
    device: Union[str, torch.device] = "cpu",
) -> np.ndarray:
    """
    Use fastDSA's existing embedding and return a 2D NumPy matrix.

    If fastDSA.dmd.embed_signal_torch is unavailable, fall back to a scalar
    representative Hankel matrix. This fallback is mainly for standalone tests.
    """
    if embed_signal_torch is not None:
        Xt = as_torch_float(X, device=device)
        D = embed_signal_torch(Xt, int(q), int(delay_interval))
        return to_2d_numpy(D)

    # Standalone fallback.
    z = representative_signal(X)
    if int(delay_interval) != 1:
        z = z[:: int(delay_interval)]
    return hankel_view(z, int(q))


def evaluate_hankel_q(
    X: ArrayLike,
    q: int,
    delay_interval: int = 1,
    min_rank: int = 1,
    max_rank_cap: Optional[int] = None,
    device: Union[str, torch.device] = "cpu",
) -> Dict[str, Any]:
    """
    Build final embedding at q and estimate SVHT rank.

    In the fastDSA pipeline this is evaluated on the full input embedding, so
    the returned r can be used as the DMD rank. For scalar data this matches the
    final Hankel SVD described in the manuscript.
    """
    q = int(q)
    D = embed_dataset_to_numpy(X, q=q, delay_interval=int(delay_interval), device=device)

    if D.shape[1] < 2:
        raise ValueError(f"Too few snapshots for q={q}: embedded shape={D.shape}")

    sv = np.linalg.svd(D, compute_uv=False, full_matrices=False)

    # For scalar eigendelay rank, r <= q. For multivariate embeddings this cap
    # keeps the effective rank tied to the delay basis rather than channel count.
    max_rank = min(len(sv), q)
    if max_rank_cap is not None:
        max_rank = min(max_rank, int(max_rank_cap))

    r, tau, sv = svht_rank(D.shape, sv, min_rank=int(min_rank), max_rank=max_rank)
    p = float(r) / float(max(q, 1))
    energy = float(np.sum(sv[:r] ** 2) / (np.sum(sv**2) + 1e-12))

    return {
        "q": int(q),
        "r": int(r),
        "p_saturation": float(p),
        "SVHT_threshold": float(tau),
        "energy_retained": float(energy),
        "embedded_rows": int(D.shape[0]),
        "embedded_cols": int(D.shape[1]),
        "n_singular_values": int(len(sv)),
        "n_above_SVHT": int(np.sum(sv > tau)),
    }


# Backward-compatible alias.
def evaluate_pilot_q(*args, **kwargs) -> Dict[str, Any]:
    return evaluate_hankel_q(*args, **kwargs)


def select_q_star_for_dataset(
    X: ArrayLike,
    delay_interval: int = 1,
    q_min: int = 20,
    q_max_abs: Optional[int] = None,
    c_sqrtN: float = 2.0,
    max_fraction: float = 0.5,
    c_n13: float = 2.0,
    acf_threshold: float = 0.0,
    saturation_threshold: float = 0.80,
    expand_factor: float = 1.50,
    max_expansions: int = 8,
    min_rank: int = 1,
    max_rank_cap: Optional[int] = None,
    time_axis: Optional[int] = None,
    device: Union[str, torch.device] = "cpu",
    verbose: bool = False,
    period_peak_to_median_threshold: float = 8.0,
    period_min_samples: int = 8,
    period_min_cycles: float = 3.0,
    train_ratio: float = 0.7,
    max_acf_lag_fraction: float = 0.25,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Select q_star and r_star without holdout error or grid search.

    Final rule:
        q0 = clip(max(q_min, q_acf, q_period, q_prop), q_min, q_geom(N))
        r0 = SVHT-rank(H_q0)
        p0 = r0/q0
        q_star = min(ceil(q0 * (1 + p0^2)), q_geom(N))
        r_star = SVHT-rank(H_q_star)

    Notes
    -----
    saturation_threshold, expand_factor, max_expansions, c_sqrtN,
    max_fraction, and acf_threshold are accepted for backward compatibility
    with earlier fastDSA.py calls. They are not used in the final one-step
    rank-pressure rule, except that q_max_abs remains available as an optional
    safety cap.
    """
    # q0 from a representative scalar signal.
    q0_info = estimate_q0(
        X,
        q_min=int(q_min),
        q_max_abs=q_max_abs,
        c_sqrtN=float(c_sqrtN),
        max_fraction=float(max_fraction),
        c_n13=float(c_n13),
        acf_threshold=float(acf_threshold),
        time_axis=time_axis,
        delay_interval=int(delay_interval),
        period_peak_to_median_threshold=float(period_peak_to_median_threshold),
        period_min_samples=int(period_min_samples),
        period_min_cycles=float(period_min_cycles),
        max_acf_lag_fraction=float(max_acf_lag_fraction),
        verbose=verbose,
    )

    # Pilot rank-pressure from the same representative scalar signal used to
    # define q0, matching the manuscript derivation.
    z = representative_signal(X, time_axis=time_axis)
    q_star_scalar, hard_info = data_driven_q_hard_from_svht(
        z,
        q0=int(q0_info["q0"]),
        train_ratio=float(train_ratio),
        min_rank=int(max(1, min_rank)),
    )

    q_star = int(q_star_scalar)

    # Final rank used for DMD fitting. For scalar data, this is exactly the
    # final H(q_star) SVHT rank. For multivariate data, this uses the full
    # fastDSA embedding at q_star.
    final = evaluate_hankel_q(
        X,
        q=q_star,
        delay_interval=int(delay_interval),
        min_rank=int(min_rank),
        max_rank_cap=max_rank_cap,
        device=device,
    )

    pilot_row = {
        **q0_info,
        **hard_info,
        "stage": "pilot_q0",
        "q": int(q0_info["q0"]),
        "r": int(hard_info["r0"]),
        "r0": int(hard_info["r0"]),
        "r0_svht": int(hard_info["r0"]),
        "p_saturation": float(hard_info["p0"]),
        "p0": float(hard_info["p0"]),
        "rank_pressure": float(hard_info["p0"]),
    }

    chosen = {
        **q0_info,
        **hard_info,
        **final,
        "stage": "final_q_star",
        "q_star": int(q_star),
        "r_star": int(final["r"]),
        "p_star": float(final["p_saturation"]),
        "q0_initial": int(q0_info["q0"]),
        "q0": int(q0_info["q0"]),
        "r0": int(hard_info["r0"]),
        "r0_svht": int(hard_info["r0"]),
        "p0": float(hard_info["p0"]),
        "rank_pressure": float(hard_info["p0"]),
        "q_hard_data": int(hard_info["q_hard_data"]),
        "hit_q_geom": bool(q_star >= int(q0_info["q_geom"])),
        "hit_q_max": bool(q_star >= int(q0_info["q_geom"])),
        "selection_rule": (
            "q0=clip(max(q_min,q_acf,q_period,q_prop),q_min,q_geom); "
            "r0=SVHT(H_q0); p0=r0/q0; "
            "q_star=min(ceil(q0*(1+p0^2)),q_geom); r_star=SVHT(H_q_star)"
        ),
        "unused_backward_compatibility_args": {
            "saturation_threshold": float(saturation_threshold),
            "expand_factor": float(expand_factor),
            "max_expansions": int(max_expansions),
        },
    }

    final_row = chosen.copy()
    diagnostics: List[Dict[str, Any]] = [pilot_row, final_row]

    if verbose:
        print(
            f"[q_star] q0={chosen['q0']}, r0={chosen['r0']}, "
            f"p0={chosen['p0']:.3f}, q_star={chosen['q_star']}, "
            f"r_star={chosen['r_star']}, q_geom={chosen['q_geom']}"
        )

    return chosen, diagnostics


# =============================================================================
# Collection-level fastDSA integration
# =============================================================================

def _shared_value(values: Sequence[int], strategy: str, allowed_max: Optional[int] = None) -> int:
    vals = np.asarray(values, dtype=int)
    if vals.size == 0:
        raise ValueError("No values provided")

    strategy = str(strategy).lower()
    if strategy == "max":
        out = int(np.max(vals))
    elif strategy == "median":
        out = int(np.median(vals))
    elif strategy == "min":
        out = int(np.min(vals))
    else:
        raise ValueError("strategy must be 'max', 'median', or 'min'")

    if allowed_max is not None:
        out = min(out, int(allowed_max))
    return int(out)


def select_q_star_for_collection(
    data: Sequence[Sequence[ArrayLike]],
    delay_interval: int = 1,
    q_min: int = 20,
    q_max_abs: Optional[int] = None,
    c_sqrtN: float = 2.0,
    max_fraction: float = 0.5,
    c_n13: float = 2.0,
    acf_threshold: float = 0.0,
    saturation_threshold: float = 0.80,
    expand_factor: float = 1.50,
    max_expansions: int = 8,
    min_rank: int = 1,
    max_rank_cap: Optional[int] = None,
    time_axis: Optional[int] = None,
    shared_q: bool = True,
    shared_q_strategy: str = "max",
    rank_strategy: str = "max",
    device: Union[str, torch.device] = "cpu",
    verbose: bool = False,
    period_peak_to_median_threshold: float = 8.0,
    period_min_samples: int = 8,
    period_min_cycles: float = 3.0,
    train_ratio: float = 0.7,
) -> Dict[str, Any]:
    """
    Select q_star/r_star for fastDSA's nested [data_block][data_index] data.

    Returns nested lists for n_delays, delay_interval, and rank that can be used
    directly by fastDSA.py.
    """
    selected_rows: List[Dict[str, Any]] = []
    diagnostic_rows: List[Dict[str, Any]] = []

    # Per-dataset q_star.
    for i, dat in enumerate(data):
        for j, X in enumerate(dat):
            chosen, diagnostics = select_q_star_for_dataset(
                X,
                delay_interval=int(delay_interval),
                q_min=int(q_min),
                q_max_abs=q_max_abs,
                c_sqrtN=float(c_sqrtN),
                max_fraction=float(max_fraction),
                c_n13=float(c_n13),
                acf_threshold=float(acf_threshold),
                saturation_threshold=float(saturation_threshold),
                expand_factor=float(expand_factor),
                max_expansions=int(max_expansions),
                min_rank=int(min_rank),
                max_rank_cap=max_rank_cap,
                time_axis=time_axis,
                device=device,
                verbose=verbose,
                period_peak_to_median_threshold=float(period_peak_to_median_threshold),
                period_min_samples=int(period_min_samples),
                period_min_cycles=float(period_min_cycles),
                train_ratio=float(train_ratio),
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

    # Shared q keeps operators more directly comparable.
    if shared_q:
        q_values = [int(row["q_star"]) for row in selected_rows]
        q_global = _shared_value(q_values, strategy=shared_q_strategy)
    else:
        q_global = None

    final_rows: List[Dict[str, Any]] = []
    q_nested: List[List[int]] = []
    r_nested_raw: List[List[int]] = []
    delay_nested: List[List[int]] = []

    for i, dat in enumerate(data):
        q_row: List[int] = []
        r_row: List[int] = []
        delay_row: List[int] = []

        for j, X in enumerate(dat):
            base = next(
                row for row in selected_rows
                if row["data_block"] == i and row["data_index"] == j
            )

            if shared_q:
                q_use = int(q_global)
                row_rank = evaluate_hankel_q(
                    X,
                    q=q_use,
                    delay_interval=int(delay_interval),
                    min_rank=int(min_rank),
                    max_rank_cap=max_rank_cap,
                    device=device,
                )
                row = base.copy()
                row.update({
                    "shared_q_recomputed_rank": True,
                    "q_used_for_fit": int(q_use),
                    "r_used_before_global_rank": int(row_rank["r"]),
                    "p_used_for_fit": float(row_rank["p_saturation"]),
                    "energy_retained_used_for_fit": float(row_rank["energy_retained"]),
                    "embedded_rows_used_for_fit": int(row_rank["embedded_rows"]),
                    "embedded_cols_used_for_fit": int(row_rank["embedded_cols"]),
                    "SVHT_threshold_used_for_fit": float(row_rank["SVHT_threshold"]),
                    "q_star": int(q_use),
                    "r_star": int(row_rank["r"]),
                })
            else:
                row = base.copy()
                q_use = int(row["q_star"])

            final_rows.append(row)
            q_row.append(int(q_use))
            r_row.append(int(row["r_star"]))
            delay_row.append(int(delay_interval))

        q_nested.append(q_row)
        r_nested_raw.append(r_row)
        delay_nested.append(delay_row)

    # Shared rank is recommended because SimilarityTransformDist compares A matrices.
    all_r = [int(row["r_star"]) for row in final_rows]
    r_global = _shared_value(all_r, strategy=rank_strategy)
    r_nested = [[int(r_global) for _ in row] for row in r_nested_raw]

    return {
        "n_delays": q_nested,
        "delay_interval": delay_nested,
        "rank": r_nested,
        "q_global": None if q_global is None else int(q_global),
        "rank_global": int(r_global),
        "selected_rows": final_rows,
        "diagnostic_rows": diagnostic_rows,
    }


__all__ = [
    "standardize_signal",
    "representative_signal",
    "autocorrelation_fft",
    "integrated_autocorrelation_time",
    "dominant_period_samples",
    "hankel_view",
    "hankel_pair_view",
    "svht_threshold",
    "svht_rank",
    "estimate_q0",
    "data_driven_q_hard_from_svht",
    "acp_choose_q_star_data_driven_hardcap",
    "select_q_star_for_dataset",
    "select_q_star_for_collection",
]
