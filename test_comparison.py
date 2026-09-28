"""End-to-end tester: compare MOME top-100 vs Baseline top-100.

Reads both FASTA files, scores them through the same mock scorers and
behavioral descriptors, then prints:
  1. Head-to-head mean score comparison
  2. Diversity statistics (spread across the behavioral space)
  3. Per-objective win/loss counts
  4. Pareto front analysis (how many MOME candidates dominate baseline ones)
  5. Scatter plots (2D panels for all 3 behavioral dimension pairs)

Usage:
    uv run python test_comparison.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path("src").resolve()))

from ampdiffusion_starter_kit.mome.candidate import PeptideCandidate, create_candidate_mock
from ampdiffusion_starter_kit.mome.filters import compute_behavioral_descriptors, read_fasta_sequences
from ampdiffusion_starter_kit.mome.pareto import dominates


# ── Paths ──────────────────────────────────────────────────────────────
MOME_FASTA = Path("generate_broad_spectrum/top.fasta")
BASELINE_FASTA = Path("generate_broad_spectrum/top_og.fasta")
OUTPUT_DIR = Path("generate_broad_spectrum")


# ── Helpers ────────────────────────────────────────────────────────────

def score_sequences(sequences: list[str]) -> list[PeptideCandidate]:
    """Score sequences with mock scorers + compute behavioral descriptors."""
    candidates = []
    for seq in sequences:
        charge, hydro, length = compute_behavioral_descriptors(seq)
        cand = create_candidate_mock(seq, charge, hydro)
        candidates.append(cand)
    return candidates


def print_header(title: str) -> None:
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")


def print_section(title: str) -> None:
    print(f"\n  --- {title} ---")


# ── Test 1: Mean Score Comparison ──────────────────────────────────────

def test_mean_scores(mome: list[PeptideCandidate], baseline: list[PeptideCandidate]) -> None:
    """Compare mean scores across all objectives."""
    print_section("TEST 1: Mean Score Comparison")

    fields = [
        ("Efficacy (-MIC)", "efficacy", True),     # higher = better
        ("Safety", "safety", True),                 # higher = better
        ("Half-life", "half_life", True),            # higher = better
        ("Solubility", "solubility", True),          # higher = better
        ("Confidence", "confidence", True),          # higher = better
        ("Immunogenicity", "immunogenicity", True),  # higher = better (1=safe)
    ]

    mome_wins = 0
    baseline_wins = 0

    print(f"\n  {'Metric':<22} {'MOME':>10} {'Baseline':>10} {'Delta':>10} {'Winner':>10}")
    print(f"  {'-' * 62}")

    for label, field, higher_better in fields:
        m_val = float(np.mean([getattr(c, field) for c in mome]))
        b_val = float(np.mean([getattr(c, field) for c in baseline]))
        delta = m_val - b_val
        sign = "+" if delta >= 0 else ""

        if (higher_better and delta > 0) or (not higher_better and delta < 0):
            winner = "MOME ✓"
            mome_wins += 1
        elif delta == 0:
            winner = "TIE"
        else:
            winner = "BASE ✓"
            baseline_wins += 1

        print(f"  {label:<22} {m_val:>10.3f} {b_val:>10.3f} {sign}{delta:>9.3f} {winner:>10}")

    print(f"\n  Score: MOME wins {mome_wins}/{len(fields)}, Baseline wins {baseline_wins}/{len(fields)}")


# ── Test 2: Diversity Statistics ───────────────────────────────────────

def test_diversity(mome: list[PeptideCandidate], baseline: list[PeptideCandidate]) -> None:
    """Compare spread across the 3D behavioral space."""
    print_section("TEST 2: Behavioral Diversity")

    dims = [
        ("Charge", "charge"),
        ("Hydrophobicity", "hydrophobicity"),
        ("Length", "length"),
    ]

    print(f"\n  {'Dimension':<18} {'':>5} {'Range':>10} {'Std':>10} {'Unique':>8}")
    print(f"  {'-' * 55}")

    for label, field in dims:
        for name, cands in [("MOME", mome), ("BASE", baseline)]:
            vals = [getattr(c, field) for c in cands]
            rng = max(vals) - min(vals)
            std = float(np.std(vals))
            uniq = len(set(vals)) if field == "length" else len(set(f"{v:.4f}" for v in vals))
            print(f"  {label if name == 'MOME' else '':<18} {name:>5} {rng:>10.3f} {std:>10.3f} {uniq:>8}")

    # Overall spread: volume of the bounding box in behavioral space
    def bbox_volume(cands: list[PeptideCandidate]) -> float:
        c = [x.charge for x in cands]
        h = [x.hydrophobicity for x in cands]
        l = [x.length for x in cands]
        return (max(c) - min(c)) * (max(h) - min(h)) * (max(l) - min(l))

    m_vol = bbox_volume(mome)
    b_vol = bbox_volume(baseline)
    ratio = m_vol / b_vol if b_vol > 0 else float("inf")
    print(f"\n  Bounding box volume:  MOME = {m_vol:.2f},  Baseline = {b_vol:.2f}")
    print(f"  MOME covers {ratio:.1f}x the behavioral space of Baseline")


# ── Test 3: Pairwise Pareto Dominance ──────────────────────────────────

def test_pareto_dominance(mome: list[PeptideCandidate], baseline: list[PeptideCandidate]) -> None:
    """Count how many MOME candidates dominate baseline ones and vice versa."""
    print_section("TEST 3: Pairwise Pareto Dominance (Efficacy × Safety)")

    # For each MOME candidate, check if it dominates any baseline candidate
    mome_dominates_count = 0
    baseline_dominates_count = 0

    for m in mome:
        for b in baseline:
            if dominates(m.objective_vector, b.objective_vector):
                mome_dominates_count += 1
            if dominates(b.objective_vector, m.objective_vector):
                baseline_dominates_count += 1

    total_pairs = len(mome) * len(baseline)
    print(f"\n  Total candidate pairs compared: {total_pairs}")
    print(f"  MOME dominates Baseline:     {mome_dominates_count:>6} pairs ({100*mome_dominates_count/total_pairs:.1f}%)")
    print(f"  Baseline dominates MOME:     {baseline_dominates_count:>6} pairs ({100*baseline_dominates_count/total_pairs:.1f}%)")
    print(f"  Non-dominated (trade-offs):  {total_pairs - mome_dominates_count - baseline_dominates_count:>6} pairs")


# ── Test 4: Sequence Overlap ──────────────────────────────────────────

def test_overlap(mome: list[PeptideCandidate], baseline: list[PeptideCandidate]) -> None:
    """Check how many sequences are shared between the two sets."""
    print_section("TEST 4: Sequence Overlap")

    mome_seqs = set(c.sequence for c in mome)
    base_seqs = set(c.sequence for c in baseline)
    overlap = mome_seqs & base_seqs

    print(f"\n  MOME unique sequences:     {len(mome_seqs)}")
    print(f"  Baseline unique sequences: {len(base_seqs)}")
    print(f"  Shared sequences:          {len(overlap)} ({100*len(overlap)/len(mome_seqs):.1f}%)")


# ── Test 5: Scatter Plots (all 3 pairwise combinations) ───────────────

def test_scatter_plots(mome: list[PeptideCandidate], baseline: list[PeptideCandidate]) -> None:
    """Generate 2x2 panel of scatter plots for all behavioral dimension pairs."""
    print_section("TEST 5: Generating Scatter Plots")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  WARNING: matplotlib not installed; skipping plots.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    pairs = [
        ("charge", "hydrophobicity", "Net Charge", "Hydrophobicity (GRAVY)"),
        ("charge", "length", "Net Charge", "Length"),
        ("hydrophobicity", "length", "Hydrophobicity (GRAVY)", "Length"),
    ]

    for ax, (xf, yf, xlabel, ylabel) in zip(axes, pairs):
        # Baseline first (so MOME dots are on top)
        bx = [getattr(c, xf) for c in baseline]
        by = [getattr(c, yf) for c in baseline]
        ax.scatter(bx, by, c="red", alpha=0.5, s=40, label="Baseline", edgecolors="darkred", linewidths=0.5)

        mx = [getattr(c, xf) for c in mome]
        my = [getattr(c, yf) for c in mome]
        ax.scatter(mx, my, c="royalblue", alpha=0.6, s=40, label="MOME", edgecolors="navy", linewidths=0.5)

        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Behavioral Space: MOME vs Baseline (all 3 dimension pairs)", fontsize=13)
    plt.tight_layout()

    save_path = OUTPUT_DIR / "comparison_scatter_panel.png"
    fig.savefig(save_path, dpi=150)
    print(f"  Saved -> {save_path}")
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────

def main() -> None:
    if not MOME_FASTA.exists():
        sys.exit(f"ERROR: MOME output not found at {MOME_FASTA}. Run the MOME pipeline first.")
    if not BASELINE_FASTA.exists():
        sys.exit(f"ERROR: Baseline output not found at {BASELINE_FASTA}.")

    print_header("MOME vs BASELINE — Full Comparison")

    print("\nLoading and scoring MOME candidates...")
    mome_seqs = read_fasta_sequences(MOME_FASTA)
    mome = score_sequences(mome_seqs)
    print(f"  Loaded {len(mome)} MOME candidates")

    print("Loading and scoring Baseline candidates...")
    baseline_seqs = read_fasta_sequences(BASELINE_FASTA)
    baseline = score_sequences(baseline_seqs)
    print(f"  Loaded {len(baseline)} Baseline candidates")

    test_mean_scores(mome, baseline)
    test_diversity(mome, baseline)
    test_pareto_dominance(mome, baseline)
    test_overlap(mome, baseline)
    test_scatter_plots(mome, baseline)

    print_header("ALL TESTS COMPLETE")


if __name__ == "__main__":
    main()
