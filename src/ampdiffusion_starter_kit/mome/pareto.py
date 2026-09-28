"""Stage 4: 2D Pareto dominance and hypervolume computation.

Provides the core multi-objective primitives used by the MOME archive:
dominance testing and hypervolume contribution for pruning overfull fronts.
"""

from __future__ import annotations

import numpy as np


def dominates(a_obj: np.ndarray, b_obj: np.ndarray) -> bool:
    """Return True if candidate A dominates candidate B in the 2D objective space.

    A dominates B iff:
        - A is >= B on ALL objectives, AND
        - A is strictly > B on at least one objective.

    Both inputs are 1D arrays of shape (2,) representing [efficacy, safety].
    """
    return bool(np.all(a_obj >= b_obj) and np.any(a_obj > b_obj))


def compute_2d_hypervolume(front_obj: np.ndarray, ref_point: np.ndarray) -> float:
    """Compute the hypervolume (area) dominated by a 2D Pareto front.

    Uses the standard sweep-line algorithm: sort points by the first objective
    ascending, compute suffix-max of the second objective, and sum rectangular
    strips from the reference point.

    Args:
        front_obj: shape (N, 2), each row is [efficacy, safety].
        ref_point: shape (2,), the reference point (worst-case values).

    Returns:
        The dominated hypervolume (area). Returns 0.0 if the front is empty.
    """
    if len(front_obj) == 0:
        return 0.0

    # Only keep points that are strictly better than the reference in both dims
    mask = np.all(front_obj > ref_point, axis=1)
    valid = front_obj[mask]
    if len(valid) == 0:
        return 0.0

    # Sort by first objective (efficacy) ascending
    pts = valid[np.argsort(valid[:, 0])]
    n = len(pts)

    # Precompute suffix max of second objective (safety)
    suffix_max_y = np.empty(n)
    suffix_max_y[-1] = pts[-1, 1]
    for i in range(n - 2, -1, -1):
        suffix_max_y[i] = max(pts[i, 1], suffix_max_y[i + 1])

    # Sweep: each point i contributes a strip from x_{i-1} to x_i with
    # height = suffix_max_y[i] - ref_y.  x_{-1} is ref_x.
    area = 0.0
    prev_x = ref_point[0]
    for i in range(n):
        x_width = pts[i, 0] - prev_x
        y_height = suffix_max_y[i] - ref_point[1]
        if x_width > 0 and y_height > 0:
            area += x_width * y_height
        prev_x = pts[i, 0]

    return area


def hypervolume_contribution_2d(
    front_obj: np.ndarray, index: int, ref_point: np.ndarray
) -> float:
    """Compute the exclusive hypervolume contribution of one point in a 2D front.

    HVC(i) = HV(front) - HV(front without point i).

    Args:
        front_obj: shape (N, 2), the full local Pareto front objectives.
        index: index of the candidate whose contribution to compute.
        ref_point: shape (2,), reference point.

    Returns:
        The hypervolume that would be lost if this candidate were removed.
    """
    hv_full = compute_2d_hypervolume(front_obj, ref_point)
    hv_without = compute_2d_hypervolume(np.delete(front_obj, index, axis=0), ref_point)
    return hv_full - hv_without
