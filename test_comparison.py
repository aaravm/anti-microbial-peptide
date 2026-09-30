"""End-to-end tester: compare two peptide sets (e.g. MOME top-100 vs Baseline top-100).

Scores both sets through the Deep Ensemble (for real Efficacy & Confidence)
and physicochemical behavioral descriptors, then prints:
  1. Head-to-head mean score comparison
  2. Diversity statistics (spread across the 3D behavioral space)
  3. Per-objective win/loss counts
  4. Pareto front analysis (pairwise dominance in Efficacy × Safety)
  5. Sequence overlap
  6. Scatter plots (3-panel figure showing all behavioral dimension pairs)

Usage:
    # Default: compares generate_broad_spectrum/top.fasta vs top_og.fasta
    uv run python test_comparison.py

    # Custom paths (e.g. comparing Deep Ensemble run vs mock run):
    uv run python test_comparison.py \
        --mome-fasta generate_broad_spectrum/top.fasta \
        --baseline-fasta generate_broad_spectrum/top_mock.fasta
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path("src").resolve()))

from ampdiffusion_starter_kit.mome.candidate import PeptideCandidate, calculate_biophysics
from ampdiffusion_starter_kit.scoring import apex_score
from ampdiffusion_starter_kit.mome.filters import compute_behavioral_descriptors, read_fasta_sequences
from ampdiffusion_starter_kit.mome.pareto import dominates

ROOT = Path(__file__).resolve().parent

# ── Helpers ────────────────────────────────────────────────────────────

def score_sequences(
    sequences: list[str],
    use_ensemble: bool = True,
    output_dir: Path | None = None,
) -> list[PeptideCandidate]:
    """Score sequences with either the APEX Ensemble or mock scorers."""
    if not sequences:
        return []

    if output_dir is None:
        output_dir = Path("generate_broad_spectrum")

    descriptors = [compute_behavioral_descriptors(seq) for seq in sequences]

    if use_ensemble:
        mean_mics, confidences = apex_score(
            seqs=sequences, 
            apex_dir=ROOT / "apex", 
            workdir=output_dir / "apex_work"
        )
        candidates = []
        for seq, (charge, hydro, length) in zip(sequences, descriptors):
            cand = calculate_biophysics(seq, charge, hydro)
            cand.efficacy = float(-mean_mics[seq])
            cand.confidence = float(confidences[seq])
            candidates.append(cand)
        return candidates
    else:
        candidates = []
        for seq, (c, h, l) in zip(sequences, descriptors):
            cand = calculate_biophysics(seq, c, h)
            cand.efficacy = -10.0
            candidates.append(cand)
        return candidates


def print_header(title: str) -> None:
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")


def print_section(title: str) -> None:
    print(f"\n  --- {title} ---")


# ── Test 1: Mean Score Comparison ──────────────────────────────────────

def test_mean_scores(
    mome: list[PeptideCandidate],
    baseline: list[PeptideCandidate],
    label_mome: str = "Candidate 1",
    label_base: str = "Candidate 2",
) -> None:
    """Compare mean scores across all objectives."""
    print_section("TEST 1: Mean Score Comparison")

    fields = [
        ("Efficacy (-MIC)", "efficacy", True),
        ("Safety", "safety", True),
        ("Half-life", "half_life", True),
        ("Solubility", "solubility", True),
        ("Confidence", "confidence", True),
        ("Immunogenicity", "immunogenicity", True),
    ]

    mome_wins = 0
    baseline_wins = 0

    col1 = label_mome[:10]
    col2 = label_base[:10]

    print(f"\n  {'Metric':<22} {col1:>10} {col2:>10} {'Delta':>10} {'Advantage':>12}")
    print(f"  {'-' * 66}")

    for label, field, higher_better in fields:
        m_val = float(np.mean([getattr(c, field) for c in mome]))
        b_val = float(np.mean([getattr(c, field) for c in baseline]))
        delta = m_val - b_val
        sign = "+" if delta >= 0 else ""

        if (higher_better and delta > 0) or (not higher_better and delta < 0):
            winner = f"{col1} ✓"
            mome_wins += 1
        elif delta == 0:
            winner = "TIE"
        else:
            winner = f"{col2} ✓"
            baseline_wins += 1

        print(f"  {label:<22} {m_val:>10.3f} {b_val:>10.3f} {sign}{delta:>9.3f} {winner:>12}")

    print(f"\n  Summary: {col1} leads in {mome_wins}/{len(fields)}, {col2} leads in {baseline_wins}/{len(fields)}")


# ── Test 2: Diversity Statistics ───────────────────────────────────────

def test_diversity(
    mome: list[PeptideCandidate],
    baseline: list[PeptideCandidate],
    label_mome: str = "Set 1",
    label_base: str = "Set 2",
) -> None:
    """Compare spread across the 3D behavioral space."""
    print_section("TEST 2: Behavioral Diversity")

    dims = [
        ("Charge", "charge"),
        ("Hydrophobicity", "hydrophobicity"),
        ("Length", "length"),
    ]

    print(f"\n  {'Dimension':<18} {'Set':>8} {'Range':>10} {'Std':>10} {'Unique':>8}")
    print(f"  {'-' * 58}")

    for label, field in dims:
        for name, cands in [(label_mome, mome), (label_base, baseline)]:
            vals = [getattr(c, field) for c in cands]
            rng = max(vals) - min(vals)
            std = float(np.std(vals))
            uniq = len(set(vals)) if field == "length" else len(set(f"{v:.4f}" for v in vals))
            print(f"  {label if name == label_mome else '':<18} {name[:8]:>8} {rng:>10.3f} {std:>10.3f} {uniq:>8}")

    def bbox_volume(cands: list[PeptideCandidate]) -> float:
        c = [x.charge for x in cands]
        h = [x.hydrophobicity for x in cands]
        l = [x.length for x in cands]
        return (max(c) - min(c)) * (max(h) - min(h)) * (max(l) - min(l))

    m_vol = bbox_volume(mome)
    b_vol = bbox_volume(baseline)
    ratio = m_vol / b_vol if b_vol > 0 else float("inf")
    print(f"\n  Bounding box volume:  {label_mome} = {m_vol:.2f},  {label_base} = {b_vol:.2f}")
    if ratio >= 1.0:
        print(f"  {label_mome} covers {ratio:.1f}x the behavioral volume of {label_base}")
    else:
        print(f"  {label_base} covers {(1.0/ratio):.1f}x the behavioral volume of {label_mome}")


# ── Test 3: Pairwise Pareto Dominance ──────────────────────────────────

def test_pareto_dominance(
    mome: list[PeptideCandidate],
    baseline: list[PeptideCandidate],
    label_mome: str = "Set 1",
    label_base: str = "Set 2",
) -> None:
    """Count how many candidates dominate each other in Efficacy × Safety."""
    print_section("TEST 3: Pairwise Pareto Dominance (Efficacy × Safety)")

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
    print(f"  {label_mome} dominates {label_base}:     {mome_dominates_count:>6} pairs ({100*mome_dominates_count/total_pairs:.1f}%)")
    print(f"  {label_base} dominates {label_mome}:     {baseline_dominates_count:>6} pairs ({100*baseline_dominates_count/total_pairs:.1f}%)")
    print(f"  Non-dominated (trade-offs):      {total_pairs - mome_dominates_count - baseline_dominates_count:>6} pairs")


# ── Test 4: Sequence Overlap ──────────────────────────────────────────

def test_overlap(
    mome: list[PeptideCandidate],
    baseline: list[PeptideCandidate],
    label_mome: str = "Set 1",
    label_base: str = "Set 2",
) -> None:
    """Check how many sequences are shared between the two sets."""
    print_section("TEST 4: Sequence Overlap")

    mome_seqs = set(c.sequence for c in mome)
    base_seqs = set(c.sequence for c in baseline)
    overlap = mome_seqs & base_seqs

    print(f"\n  {label_mome} unique sequences: {len(mome_seqs)}")
    print(f"  {label_base} unique sequences: {len(base_seqs)}")
    print(f"  Shared sequences:        {len(overlap)} ({100*len(overlap)/len(mome_seqs):.1f}%)")


# ── Test 5: Scatter Plots (all 3 pairwise combinations) ───────────────

def test_scatter_plots(
    mome: list[PeptideCandidate],
    baseline: list[PeptideCandidate],
    output_dir: Path,
    label_mome: str = "Set 1",
    label_base: str = "Set 2",
) -> None:
    """Generate panel of scatter plots for all behavioral dimension pairs."""
    print_section("TEST 5: Generating Scatter Plots")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  WARNING: matplotlib not installed; skipping plots.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    pairs = [
        ("charge", "hydrophobicity", "Net Charge", "Hydrophobicity (GRAVY)"),
        ("charge", "length", "Net Charge", "Length"),
        ("hydrophobicity", "length", "Hydrophobicity (GRAVY)", "Length"),
    ]

    for ax, (xf, yf, xlabel, ylabel) in zip(axes, pairs):
        bx = [getattr(c, xf) for c in baseline]
        by = [getattr(c, yf) for c in baseline]
        ax.scatter(bx, by, c="red", alpha=0.5, s=40, label=label_base, edgecolors="darkred", linewidths=0.5)

        mx = [getattr(c, xf) for c in mome]
        my = [getattr(c, yf) for c in mome]
        ax.scatter(mx, my, c="royalblue", alpha=0.6, s=40, label=label_mome, edgecolors="navy", linewidths=0.5)

        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"Behavioral Space: {label_mome} vs {label_base}", fontsize=13)
    plt.tight_layout()

    save_path = output_dir / "comparison_scatter_panel.png"
    fig.savefig(save_path, dpi=150)
    print(f"  Saved -> {save_path}")
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two top-100 peptide FASTA files.")
    parser.add_argument(
        "--mome-fasta",
        type=Path,
        default=Path("generate_broad_spectrum/top.fasta"),
        help="Path to first/MOME FASTA (default: generate_broad_spectrum/top.fasta)",
    )
    parser.add_argument(
        "--baseline-fasta",
        type=Path,
        default=Path("generate_broad_spectrum/top_round1.fasta"),
        help="Path to second/Baseline FASTA (default: generate_broad_spectrum/top_round1.fasta)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("generate_broad_spectrum"),
        help="Output directory for comparison plots (default: generate_broad_spectrum)",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        default=False,
        help="Use mock scorers instead of Deep Ensemble",
    )
    args = parser.parse_args()

    if not args.mome_fasta.exists():
        sys.exit(f"ERROR: File not found: {args.mome_fasta}")
    if not args.baseline_fasta.exists():
        sys.exit(f"ERROR: File not found: {args.baseline_fasta}")

    label_mome = args.mome_fasta.stem.upper()
    label_base = args.baseline_fasta.stem.upper()

    print_header(f"Comparison: {label_mome} ({args.mome_fasta.name}) vs {label_base} ({args.baseline_fasta.name})")
    print(f"Scoring mode: {'MOCK' if args.mock else 'REAL APEX ENSEMBLE'}")

    print(f"\nLoading and scoring {args.mome_fasta.name}...")
    mome_seqs = read_fasta_sequences(args.mome_fasta)
    mome = score_sequences(mome_seqs, use_ensemble=not args.mock, output_dir=args.output_dir)
    print(f"  Loaded {len(mome)} candidates")

    print(f"\nLoading and scoring {args.baseline_fasta.name}...")
    baseline_seqs = read_fasta_sequences(args.baseline_fasta)
    baseline = score_sequences(baseline_seqs, use_ensemble=not args.mock, output_dir=args.output_dir)
    print(f"  Loaded {len(baseline)} candidates")

    test_mean_scores(mome, baseline, label_mome, label_base)
    test_diversity(mome, baseline, label_mome, label_base)
    test_pareto_dominance(mome, baseline, label_mome, label_base)
    test_overlap(mome, baseline, label_mome, label_base)
    test_scatter_plots(mome, baseline, args.output_dir, label_mome, label_base)

    print_header("ALL TESTS COMPLETE")


if __name__ == "__main__":
    main()
