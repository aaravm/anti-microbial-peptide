"""MOME pipeline orchestrator.

Ties together all 8 stages: hard filters, behavioral descriptors, mock/real
scoring, CVT grid, MOME archiving, Compromise Programming extraction,
and analytics.

Usage (mock scorers for development)::

    uv run python -m ampdiffusion_starter_kit.mome.pipeline \
        --library-fasta generate_broad_spectrum/library.fasta \
        --mock
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ampdiffusion_starter_kit.mome.analytics import (
    export_metrics_csv,
    plot_behavioral_scatter,
)
from ampdiffusion_starter_kit.mome.archive import MOMEArchive
from ampdiffusion_starter_kit.mome.candidate import (
    PeptideCandidate,
    calculate_biophysics,
)
from ampdiffusion_starter_kit.mome.cvt import CVTGrid
from ampdiffusion_starter_kit.scoring import apex_score
from ampdiffusion_starter_kit.mome.extraction import extract_top_100
from ampdiffusion_starter_kit.mome.filters import (
    apply_hard_filters,
    compute_behavioral_descriptors,
    passes_penalty_thresholds,
    read_fasta_sequences,
)

ROOT = Path(__file__).resolve().parents[3]


def _write_fasta(sequences: list[str], path: Path) -> None:
    """Write sequences to a FASTA file."""
    with open(path, "w") as f:
        for i, seq in enumerate(sequences, start=1):
            f.write(f">seq{i}\n{seq}\n")


def run_mome_pipeline(
    library_fasta: Path,
    antibacterial_fasta: Path,
    output_dir: Path,
    n_cells: int = 500,
    top_k: int = 100,
    max_front_size: int = 25,
    use_mock_scorers: bool = True,
) -> list[PeptideCandidate]:
    """Run the full MOME pipeline.

    Steps:
        1. Read library sequences.
        2. Read antibacterial references.
        3. Apply hard filters (valid sequence + novelty check).
        4. Compute behavioral descriptors for all valid sequences.
        5. Score candidates (mock or real).
        6. Apply penalty thresholds (confidence + immunogenicity).
        7. Fit CVT grid on behavioral vectors.
        8. Create MOME archive and add all candidates.
        9. Extract top 100 via Compromise Programming.
        10. Export metrics CSV and generate scatter plot.
        11. Write top.fasta.

    Returns the top_k selected candidates.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1 & 2: Load data ──────────────────────────────────────────
    print("=" * 60)
    print("MOME Pipeline")
    print("=" * 60)
    print(f"\n[1/10] Reading library from {library_fasta}...")
    sequences = read_fasta_sequences(library_fasta)
    # Deduplicate
    sequences = list(set(sequences))
    print(f"  Loaded {len(sequences)} unique sequences")

    print(f"\n[2/10] Reading antibacterial references from {antibacterial_fasta}...")
    references = set(read_fasta_sequences(antibacterial_fasta))
    print(f"  Loaded {len(references)} reference sequences")

    # ── Step 3: Hard filters ───────────────────────────────────────────
    print(f"\n[3/10] Applying hard filters...")
    valid_seqs = apply_hard_filters(sequences, references)

    # ── Step 4: Behavioral descriptors ─────────────────────────────────
    print(f"\n[4/10] Computing behavioral descriptors for {len(valid_seqs)} sequences...")
    descriptors: list[tuple[float, float, int]] = []
    for i, seq in enumerate(valid_seqs):
        descriptors.append(compute_behavioral_descriptors(seq))
        if (i + 1) % 5000 == 0 or (i + 1) == len(valid_seqs):
            print(f"  computed {i + 1}/{len(valid_seqs)} descriptors")

    # ── Step 5: Score candidates ───────────────────────────────────────
    print(f"\n[5/10] Scoring candidates ({'MOCK' if use_mock_scorers else 'APEX ENSEMBLE'})...")
    candidates: list[PeptideCandidate] = []
    if use_mock_scorers:
        for seq, (charge, hydro, length) in zip(valid_seqs, descriptors):
            # In mock mode, we just leave the efficacy at 0.0 or randomly fill it
            # But the user isn't really using mock mode anymore.
            cand = calculate_biophysics(seq, charge, hydro)
            cand.efficacy = -10.0 # dummy
            candidates.append(cand)
    else:
        print("  Evaluating efficacy and confidence via APEX Ensemble...")
        mean_mics, confidences = apex_score(
            seqs=valid_seqs, 
            apex_dir=ROOT / "apex", 
            workdir=output_dir / "apex_work"
        )
        for i, (seq, (charge, hydro, length)) in enumerate(zip(valid_seqs, descriptors)):
            cand = calculate_biophysics(seq, charge, hydro)
            cand.efficacy = float(-mean_mics[seq])
            cand.confidence = float(confidences[seq])
            candidates.append(cand)
    print(f"  Scored {len(candidates)} candidates")

    # ── Step 6: Penalty thresholds ─────────────────────────────────────
    print(f"\n[6/10] Applying penalty thresholds (confidence >= 0.3, immuno >= 0.3, MIC <= 200)...")
    pre_count = len(candidates)
    candidates = [
        c for c in candidates
        if passes_penalty_thresholds(c.confidence, c.immunogenicity, c.efficacy)
    ]
    print(f"  Kept {len(candidates)}/{pre_count} after penalty filters")

    # ── Step 7: Fit CVT grid ───────────────────────────────────────────
    print(f"\n[7/10] Fitting CVT grid with {n_cells} cells...")
    behavior_vectors = np.array([c.behavior_vector for c in candidates])
    grid = CVTGrid(n_cells=n_cells).fit(behavior_vectors)

    # ── Step 8: MOME archiving ─────────────────────────────────────────
    print(f"\n[8/10] Building MOME archive (max front size = {max_front_size})...")
    archive = MOMEArchive(grid, max_front_size=max_front_size)
    added = archive.add_batch(candidates)
    print(f"  Added {added}/{len(candidates)} candidates to archive")
    print(f"  {archive.summary()}")

    # ── Step 9: Extract top-k ──────────────────────────────────────────
    print(f"\n[9/10] Extracting top {top_k} via Compromise Programming...")
    top = extract_top_100(archive, candidates, top_k=top_k)

    # ── Step 10: Analytics ─────────────────────────────────────────────
    print(f"\n[10/10] Exporting analytics...")
    csv_path = output_dir / "top100_metrics.csv"
    export_metrics_csv(top, csv_path)

    scatter_path = output_dir / "behavioral_scatter.png"
    plot_behavioral_scatter(top, save_path=scatter_path)

    # Write FASTA
    fasta_path = output_dir / "top.fasta"
    _write_fasta([c.sequence for c in top], fasta_path)
    print(f"Wrote {len(top)} sequences -> {fasta_path}")

    print("\n" + "=" * 60)
    print("MOME Pipeline complete!")
    print("=" * 60)

    return top


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the MOME selection pipeline on a pre-generated library."
    )
    parser.add_argument(
        "--library-fasta", type=Path, required=True,
        help="Path to the merged library FASTA (e.g. generate_broad_spectrum/library.fasta)",
    )
    parser.add_argument(
        "--antibacterial-fasta", type=Path,
        default=ROOT / "data" / "antibacterial.fasta",
        help="Path to the antibacterial reference FASTA",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("generate_broad_spectrum"),
        help="Output directory for top.fasta and metrics",
    )
    parser.add_argument("--n-cells", type=int, default=500, help="Number of CVT cells")
    parser.add_argument("--top-k", type=int, default=100, help="Number of final candidates")
    parser.add_argument(
        "--mock",
        dest="mock",
        action="store_true",
        default=False,
        help="Use mock scorers for development (default: False, uses APEX Ensemble)",
    )
    parser.add_argument(
        "--no-mock",
        dest="mock",
        action="store_false",
        help="Use APEX Ensemble for efficacy and confidence (default)",
    )

    args = parser.parse_args()
    run_mome_pipeline(
        library_fasta=args.library_fasta,
        antibacterial_fasta=args.antibacterial_fasta,
        output_dir=args.output_dir,
        n_cells=args.n_cells,
        top_k=args.top_k,
        use_mock_scorers=args.mock,
    )


if __name__ == "__main__":
    main()
