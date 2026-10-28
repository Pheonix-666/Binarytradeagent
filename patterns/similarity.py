"""Pattern similarity — vectorized correlation and pure-numpy constrained DTW.

Strict Rules:
  - Pure numpy only, zero compiled dependencies (Windows compatible).
  - Similarity = clip(correlation * 100, 0, 100). Negative correlation scores 0.
  - Guard against divide-by-zero on flat windows (score 0.0).
  - Sakoe-Chiba constrained DTW for fast refinement of top matches.
"""

from __future__ import annotations

import numpy as np


# ── Final Similarity Combination Rule ────────────────────────────────
# final_similarity = (1.0 - dtw_weight) * corr_similarity + dtw_weight * dtw_similarity
#
# Correlation (1 - dtw_weight) captures linear profile alignment and overall slope.
# DTW (dtw_weight) refines top matches by penalizing elastic time-warping deformation
# within the Sakoe-Chiba constraint band, accommodating minor timing differences in peaks/troughs.
# If a price window is flat (zero variance), similarity is 0.0 with zero division guarded.
DEFAULT_DTW_WEIGHT: float = 0.3
DEFAULT_SAKOE_CHIBA_RADIUS: int = 6


def resample_to_points(arr: np.ndarray, num_points: int = 64) -> np.ndarray:
    """Resample 1-D array to num_points using linear interpolation."""
    n = len(arr)
    if n == num_points:
        return np.asarray(arr, dtype=np.float64)
    if n < 2:
        return np.full(num_points, arr[0] if n == 1 else 0.0, dtype=np.float64)

    orig_indices = np.linspace(0.0, 1.0, n)
    target_indices = np.linspace(0.0, 1.0, num_points)
    return np.interp(target_indices, orig_indices, arr).astype(np.float64)


def z_normalize(arr: np.ndarray, eps: float = 1e-8) -> tuple[np.ndarray, bool]:
    """Z-normalize array: (arr - mean) / std.

    Guards against divide-by-zero: if std < eps (flat series), returns all zeros
    and is_flat=True.
    """
    arr = np.asarray(arr, dtype=np.float64)
    std = float(np.std(arr))
    if std < eps:
        return np.zeros_like(arr, dtype=np.float64), True
    mean = float(np.mean(arr))
    return (arr - mean) / std, False


def compute_vectorized_correlation(
    templates_matrix: np.ndarray,
    query_normalized: np.ndarray,
    is_flat: bool = False,
) -> np.ndarray:
    """Compute Pearson correlation of query against all templates in templates_matrix.

    Assumes templates_matrix has shape (M, K) with z-normalized rows,
    and query_normalized has shape (K,) with mean 0 and std 1.

    Returns similarity scores in [0.0, 100.0] of shape (M,).
    Negative correlation scores 0. Flat windows score 0.
    """
    if is_flat or templates_matrix.size == 0:
        return np.zeros(len(templates_matrix), dtype=np.float64)

    k = templates_matrix.shape[1]
    # dot product divided by k gives Pearson r because both vectors are z-normalized
    corrs = np.dot(templates_matrix, query_normalized) / float(k)
    # Similarity = clip(correlation * 100, 0, 100). Negative correlation scores 0.
    return np.clip(corrs * 100.0, 0.0, 100.0)


def dtw_sakoe_chiba(
    q: np.ndarray,
    t: np.ndarray,
    radius: int = DEFAULT_SAKOE_CHIBA_RADIUS,
) -> float:
    """Compute Dynamic Time Warping distance with Sakoe-Chiba band constraint.

    Pure numpy implementation. Both q and t must be 1-D arrays of equal length K.
    Constraint: |i - j| <= radius.

    Returns normalized average step distance.
    """
    k = len(q)
    if k == 0:
        return 0.0
    if k == 1:
        return float(abs(q[0] - t[0]))

    r = max(1, radius)
    # Precompute pairwise absolute differences via vectorized broadcasting
    diff = np.abs(q[:, None] - t[None, :])

    # Cost matrix initialized to infinity
    cost = np.full((k + 1, k + 1), np.inf, dtype=np.float64)
    cost[0, 0] = 0.0

    for i in range(1, k + 1):
        j_min = max(1, i - r)
        j_max = min(k, i + r)
        d_row = diff[i - 1]
        for j in range(j_min, j_max + 1):
            cost[i, j] = d_row[j - 1] + min(
                cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1]
            )

    total_dist = cost[k, k]
    if np.isinf(total_dist):
        return 999.0

    # Normalized average step distance
    return float(total_dist / (2.0 * k))


def dtw_distance_to_similarity(dist_norm: float) -> float:
    """Convert normalized DTW step distance to a 0-100 similarity score.

    For identical z-normalized series, dist_norm == 0.0 -> similarity = 100.0.
    For distant/opposite series, dist_norm >= 1.5 -> similarity = 0.0.
    """
    if dist_norm >= 1.5:
        return 0.0
    sim = (1.0 - (dist_norm / 1.5)) * 100.0
    return float(np.clip(sim, 0.0, 100.0))


def compute_combined_similarity(
    corr_similarity: float,
    dtw_similarity: float,
    dtw_weight: float = DEFAULT_DTW_WEIGHT,
    is_flat: bool = False,
) -> float:
    """Combine correlation similarity and DTW similarity according to the defined rule.

    Formula:
      final_similarity = (1.0 - dtw_weight) * corr_similarity + dtw_weight * dtw_similarity

    Rules:
      - Negative correlation scores 0.0 (if corr_similarity <= 0, final_similarity = 0.0).
      - Flat windows score 0.0.
    """
    if is_flat or corr_similarity <= 0.0:
        return 0.0
    final_sim = (1.0 - dtw_weight) * corr_similarity + dtw_weight * dtw_similarity
    return float(np.clip(final_sim, 0.0, 100.0))
