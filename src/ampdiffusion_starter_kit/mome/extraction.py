"""Stage 6: Final extraction via Compromise Programming.

Picks one champion per filled cell using a distance-to-ideal score in the
normalized (Efficacy, Safety) space, with Half-life and Solubility as
weighted bonuses. Then selects the globally best 100 champions.
"""

from __future__ import annotations

import numpy as np

from ampdiffusion_starter_kit.mome.candidate import PeptideCandidate

if False:  # TYPE_CHECKING
    from ampdiffusion_starter_kit.mome.archive import MOMEArchive


def compute_global_bounds(
    candidates: list[PeptideCandidate],
) -> tuple[dict[str, float], dict[str, float]]:
    """Compute global min and max for the 4 score dimensions.

    Returns (global_min, global_max) dicts keyed by field name.
    """
    fields = ["efficacy", "safety", "half_life", "solubility"]
    g_min: dict[str, float] = {}
    g_max: dict[str, float] = {}
    for f in fields:
        values = [getattr(c, f) for c in candidates]
        g_min[f] = min(values)
        g_max[f] = max(values)
    return g_min, g_max


def _normalize(value: float, lo: float, hi: float) -> float:
    """Normalize a value to [0, 1] given global bounds. Returns 0.5 if lo == hi."""
    if hi - lo < 1e-12:
        return 0.5
    return (value - lo) / (hi - lo)


def extract_champion(
    front: list[PeptideCandidate],
    g_min: dict[str, float],
    g_max: dict[str, float],
    half_life_weight: float = 0.1,
    solubility_weight: float = 0.05,
) -> tuple[PeptideCandidate, float]:
    """Pick the best candidate from a cell's Pareto front via Compromise Programming.

    Score = -distance_to_ideal(norm_eff, norm_safety)
            + half_life_weight * norm_half_life
            + solubility_weight * norm_solubility

    Returns (champion, score).
    """
    best_cand = front[0]
    best_score = -float("inf")

    for c in front:
        ne = _normalize(c.efficacy, g_min["efficacy"], g_max["efficacy"])
        ns = _normalize(c.safety, g_min["safety"], g_max["safety"])
        nh = _normalize(c.half_life, g_min["half_life"], g_max["half_life"])
        nl = _normalize(c.solubility, g_min["solubility"], g_max["solubility"])

        # Distance to the ideal point (1, 1) in normalized space
        dist = np.sqrt((1.0 - ne) ** 2 + (1.0 - ns) ** 2)
        score = -dist + half_life_weight * nh + solubility_weight * nl

        if score > best_score:
            best_score = score
            best_cand = c

    return best_cand, best_score


def extract_top_100(
    archive: MOMEArchive,
    all_candidates: list[PeptideCandidate],
    top_k: int = 100,
) -> list[PeptideCandidate]:
    """Extract the final top-k candidates from the MOME archive.

    Steps:
        1. Compute global normalization bounds from ALL candidates.
        2. For each filled cell, pick its champion via Compromise Programming.
        3. Sort champions by score descending.
        4. Return the top top_k.
    """
    g_min, g_max = compute_global_bounds(all_candidates)

    champions: list[tuple[PeptideCandidate, float]] = []
    for cell_idx, front in archive.cells.items():
        champ, score = extract_champion(front, g_min, g_max)
        champions.append((champ, score))

    # Sort by score descending (higher = better)
    champions.sort(key=lambda x: x[1], reverse=True)

    n_filled = len(champions)
    selected = [c for c, _ in champions[:top_k]]

    print(f"Extraction: {n_filled} filled cells -> selected top {len(selected)} champions")
    if n_filled < top_k:
        print(f"  WARNING: only {n_filled} filled cells, fewer than {top_k} requested!")

    return selected
