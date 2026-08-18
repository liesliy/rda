"""MAD-based normalization utilities.

Implements robust z-score normalization using Median Absolute Deviation
(MAD) instead of standard deviation, which is less sensitive to outliers
in robot trajectory data.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


def mad(values: np.ndarray) -> float:
    """Compute Median Absolute Deviation of a 1-D array.

    Args:
        values: 1-D array of numeric values.

    Returns:
        MAD value. Returns 0.0 for empty arrays.
    """
    if values.size == 0:
        return 0.0
    med = float(np.median(values))
    return float(np.median(np.abs(values - med)))


def percentile_stats(values: np.ndarray) -> Dict[str, float]:
    """Compute p05, p25, median, p75, p95 for a 1-D array.

    Args:
        values: 1-D array of numeric values.

    Returns:
        Dict with keys ``p05``, ``p25``, ``median``, ``p75``, ``p95``.
        All zeros for empty input.
    """
    if values.size == 0:
        return {"p05": 0.0, "p25": 0.0, "median": 0.0, "p75": 0.0, "p95": 0.0}
    pcts = np.percentile(values, [5, 25, 50, 75, 95])
    return {
        "p05": float(pcts[0]),
        "p25": float(pcts[1]),
        "median": float(pcts[2]),
        "p75": float(pcts[3]),
        "p95": float(pcts[4]),
    }


def robust_z_scores(values: np.ndarray) -> np.ndarray:
    """Compute robust (MAD-based) z-scores for a 1-D array.

    Scaled by 0.6745 so that the result is comparable to standard
    z-scores for normally distributed data.

    Args:
        values: 1-D array of raw values.

    Returns:
        Array of robust z-scores with same shape as input.
        All zeros if MAD is zero.
    """
    if values.size == 0:
        return np.array([], dtype=np.float64)
    med = float(np.median(values))
    m = mad(values)
    if m == 0.0:
        return np.zeros_like(values, dtype=np.float64)
    return 0.6745 * (values - med) / m


def bad_z_scores(values: np.ndarray, direction: str) -> np.ndarray:
    """Compute direction-aware bad z-scores for a 1-D array.

    For "higher_is_worse" metrics, positive z-scores indicate anomalies
    (higher values = more anomalous), so bad_z = raw_z.
    For "lower_is_worse" metrics, negative z-scores indicate anomalies
    (lower values = more anomalous), so bad_z = -raw_z.

    After this transformation, positive bad_z always means "more anomalous"
    regardless of the original metric direction.

    Args:
        values: 1-D array of raw values.
        direction: One of ``"higher_is_worse"`` or ``"lower_is_worse"``.

    Returns:
        Array of bad z-scores with same shape as input.
        All zeros if MAD is zero.

    Raises:
        ValueError: If direction is not a recognized value.
    """
    raw_z = robust_z_scores(values)
    if direction == "higher_is_worse":
        return raw_z
    elif direction == "lower_is_worse":
        return -raw_z
    else:
        raise ValueError(
            f"Unknown direction '{direction}'. "
            "Must be 'higher_is_worse' or 'lower_is_worse'."
        )


def compute_pca_loadings(
    z_matrix: np.ndarray,
    metric_names: List[str],
) -> Tuple[List[Dict[str, float]], List[float]]:
    """Compute principal component loadings from a z-score matrix.

    Returns up to min(2, n_features) components for use in behavioral
    scoring. The first component captures the dominant anomaly direction,
    while the second captures orthogonal variance.

    Args:
        z_matrix: Array of shape ``(n_episodes, n_metrics)`` containing
            robust z-scores (already normalized and direction-adjusted).
        metric_names: List of metric names in column order.

    Returns:
        Tuple of ``(components_list, explained_variance_ratios)`` where:
        - ``components_list`` is a list of dicts, each mapping metric names
          to their loading weights for that component.
        - ``explained_variance_ratios`` is a list of floats indicating
          how much variance each component explains.
        Returns empty lists if PCA cannot be computed.
    """
    n_samples, n_features = z_matrix.shape
    if n_samples < 2 or n_features < 2:
        return [], []

    # Use SVD-based PCA on the covariance matrix for numerical stability.
    # Center the data (z-scores are already roughly centered, but
    # centering explicitly is correct regardless).
    X = z_matrix - np.mean(z_matrix, axis=0)

    try:
        # Full SVD: U (n×n), S (min(n,m)), Vt (m×m)
        U, S, Vt = np.linalg.svd(X, full_matrices=False)
    except np.linalg.LinAlgError:
        return [], []

    if S.size == 0 or S[0] == 0.0:
        return [], []

    # Explained variance ratio: singular values squared / sum of all
    total_variance = float(np.sum(S ** 2))
    if total_variance == 0.0:
        return [], []

    # Number of components to return: min(2, n_features)
    n_components = min(2, n_features)

    components_list: List[Dict[str, float]] = []
    explained_variance_ratios: List[float] = []

    for i in range(n_components):
        if i >= len(S) or S[i] == 0.0:
            break

        evr_i = float((S[i] ** 2) / total_variance)
        explained_variance_ratios.append(evr_i)

        # Loadings = right singular vector Vt[i]
        loadings = Vt[i].copy()
        # Normalize sign so the largest absolute loading is positive
        max_idx = int(np.argmax(np.abs(loadings)))
        if loadings[max_idx] < 0:
            loadings = -loadings

        loadings_dict: Dict[str, float] = {}
        for name, val in zip(metric_names, loadings):
            loadings_dict[name] = float(val)
        components_list.append(loadings_dict)

    return components_list, explained_variance_ratios
