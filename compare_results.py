"""Compare MOME top-100 against the baseline top-100.

Scores both sets through the same scorers and behavioral descriptors,
then prints a head-to-head comparison table and generates scatter plots.

Usage:
    uv run python compare_results.py \
        --mome-fasta generate_broad_spectrum/top.fasta \
        --baseline-fasta generate_broad_spectrum/top_baseline.fasta
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path("src").resolve()))

from ampdiffusion_starter_kit.mome.candidate import PeptideCandidate, create_candidate_mock
from ampdiffusion_starter_kit.mome.filters import (
    compute_behavioral_descriptors,
    read_fasta_sequences,
)
from ampdiffusion_starter_kit.mome.analytics import (
    export_metrics_csv,
    plot_behavioral_scatter,
    print_comparison,
)


def score_sequences(sequences: list[str]) -> list[PeptideCandidate]:
    """Score a list of sequences using mock scorers and behavioral descriptors."""
    candidates = []
    for seq in sequences:
        charge, hydro, length = compute_behavioral_descriptors(seq)
        cand = create_candidate_mock(seq, charge, hydro)
        candidates.append(cand)
    return candidates


def print_diversity_stats(label: str, candidates: list[PeptideCandidate]) -> None:
    """Print diversity statistics for a set of candidates."""
    charges = [c.charge for c in candidates]
    hydros = [c.hydrophobicity for c in candidates]
    lengths = [c.length for c in candidates]

    print(f"\n  {label} — Diversity Stats:")
    print(f"    Charge:          min={min(charges):>7.2f}  max={max(charges):>7.2f}  "
          f"std={np.std(charges):>6.2f}  range={max(charges)-min(charges):.2f}")
    print(f"    Hydrophobicity:  min={min(hydros):>7.2f}  max={max(hydros):>7.2f}  "
          f"std={np.std(hydros):>6.2f}  range={max(hydros)-min(hydros):.2f}")
    print(f"    Length:           min={min(lengths):>7d}  max={max(lengths):>7d}  "
          f"std={np.std(lengths):>6.2f}  range={max(lengths)-min(lengths)}")
    print(f"    Unique lengths: {len(set(lengths))}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare MOME vs Baseline top-100")
    parser.add_argument(
        "--mome-fasta", type=Path, required=True,
        help="Path to MOME top.fasta",
    )
    parser.add_argument(
        "--baseline-fasta", type=Path, default=None,
        help="Path to baseline top.fasta (optional; if missing, only MOME stats are shown)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("generate_broad_spectrum"),
        help="Directory to save comparison outputs",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ── Score MOME candidates ──────────────────────────────────────────
    print("Loading and scoring MOME candidates...")
    mome_seqs = read_fasta_sequences(args.mome_fasta)
    mome_candidates = score_sequences(mome_seqs)
    print(f"  MOME: {len(mome_candidates)} sequences")

    # ── Score Baseline candidates (if available) ───────────────────────
    baseline_candidates: list[PeptideCandidate] | None = None
    if args.baseline_fasta and args.baseline_fasta.exists():
        print("Loading and scoring Baseline candidates...")
        baseline_seqs = read_fasta_sequences(args.baseline_fasta)
        baseline_candidates = score_sequences(baseline_seqs)
        print(f"  Baseline: {len(baseline_candidates)} sequences")

    # ── Head-to-Head Comparison ────────────────────────────────────────
    if baseline_candidates:
        print_comparison(mome_candidates, baseline_candidates)

        # Diversity stats
        print("=" * 65)
        print("DIVERSITY COMPARISON")
        print("=" * 65)
        print_diversity_stats("MOME", mome_candidates)
        print_diversity_stats("Baseline", baseline_candidates)

        # Check sequence overlap
        mome_set = set(c.sequence for c in mome_candidates)
        baseline_set = set(c.sequence for c in baseline_candidates)
        overlap = mome_set & baseline_set
        print(f"\n  Sequence overlap: {len(overlap)} / {len(mome_set)} "
              f"({100*len(overlap)/len(mome_set):.1f}%)")

        # Scatter plot
        scatter_path = args.output_dir / "comparison_scatter.png"
        plot_behavioral_scatter(mome_candidates, baseline_candidates, save_path=scatter_path)

        # Export both CSVs
        export_metrics_csv(mome_candidates, args.output_dir / "mome_top100_metrics.csv")
        export_metrics_csv(baseline_candidates, args.output_dir / "baseline_top100_metrics.csv")
    else:
        # MOME only
        print("\n  No baseline provided. Showing MOME stats only.")
        print_diversity_stats("MOME", mome_candidates)
        export_metrics_csv(mome_candidates, args.output_dir / "mome_top100_metrics.csv")

        scatter_path = args.output_dir / "mome_scatter.png"
        plot_behavioral_scatter(mome_candidates, save_path=scatter_path)

    print("\nDone!")


if __name__ == "__main__":
    main()
