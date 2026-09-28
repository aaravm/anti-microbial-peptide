"""Stage 8: Analytics and visualization.

Exports metrics to CSV, prints head-to-head comparisons, and generates
behavioral scatter plots to prove MOME's diversity advantage.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ampdiffusion_starter_kit.mome.candidate import PeptideCandidate


def export_metrics_csv(candidates: list[PeptideCandidate], path: Path) -> None:
    """Write a CSV with all scores and behavioral features for the selected candidates.

    Columns: sequence, charge, hydrophobicity, length, efficacy, safety,
             half_life, solubility, confidence, immunogenicity.
    """
    rows = []
    for c in candidates:
        rows.append({
            "sequence": c.sequence,
            "charge": c.charge,
            "hydrophobicity": c.hydrophobicity,
            "length": c.length,
            "efficacy": c.efficacy,
            "safety": c.safety,
            "half_life": c.half_life,
            "solubility": c.solubility,
            "confidence": c.confidence,
            "immunogenicity": c.immunogenicity,
        })
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    print(f"Exported metrics for {len(candidates)} candidates -> {path}")


def print_comparison(
    mome_top: list[PeptideCandidate],
    baseline_top: list[PeptideCandidate],
) -> None:
    """Print a formatted table comparing mean scores of MOME vs. Baseline."""

    fields = ["efficacy", "safety", "half_life", "solubility", "confidence", "immunogenicity"]
    labels = ["Efficacy (-MIC)", "Safety", "Half-life", "Solubility", "Confidence", "Immunogenicity"]

    def mean_of(cands: list[PeptideCandidate], field: str) -> float:
        return float(np.mean([getattr(c, field) for c in cands]))

    print("\n" + "=" * 65)
    print(f"{'Metric':<20} {'MOME':>12} {'Baseline':>12} {'Delta':>12}")
    print("-" * 65)
    for label, field in zip(labels, fields):
        m_val = mean_of(mome_top, field)
        b_val = mean_of(baseline_top, field)
        delta = m_val - b_val
        sign = "+" if delta >= 0 else ""
        print(f"{label:<20} {m_val:>12.3f} {b_val:>12.3f} {sign}{delta:>11.3f}")
    print("=" * 65 + "\n")


def plot_behavioral_scatter(
    mome_top: list[PeptideCandidate],
    baseline_top: list[PeptideCandidate] | None = None,
    save_path: Path | None = None,
) -> None:
    """Create a 2D scatter plot of Charge (x) vs Hydrophobicity (y).

    MOME candidates are shown as blue dots; baseline (if provided) as red dots.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")  # Non-interactive backend for headless environments
        import matplotlib.pyplot as plt
    except ImportError:
        print("WARNING: matplotlib not installed; skipping scatter plot.")
        return

    fig, ax = plt.subplots(figsize=(10, 7))

    if baseline_top:
        bx = [c.charge for c in baseline_top]
        by = [c.hydrophobicity for c in baseline_top]
        ax.scatter(bx, by, c="red", alpha=0.6, s=50, label="Baseline Top-100", edgecolors="darkred")

    mx = [c.charge for c in mome_top]
    my = [c.hydrophobicity for c in mome_top]
    ax.scatter(mx, my, c="royalblue", alpha=0.7, s=50, label="MOME Top-100", edgecolors="navy")

    ax.set_xlabel("Net Charge (pH 7.0)", fontsize=12)
    ax.set_ylabel("Hydrophobicity (GRAVY)", fontsize=12)
    ax.set_title("Behavioral Diversity: MOME vs Baseline", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Scatter plot saved -> {save_path}")
    else:
        plt.show()
    plt.close(fig)
