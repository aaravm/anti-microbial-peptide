"""Validation and evaluation suite for the Deep Ensemble MIC predictor.

Performs:
1. In-distribution correlation test on the 46 experimental training peptides
   (Pearson r, Spearman rho, MSE).
2. Uncertainty calibration check comparing in-distribution confidence vs.
   out-of-distribution / generated library sequences from library.fasta.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

from ampdiffusion_starter_kit.mome.ensemble.data import load_experimental_mic_data
from ampdiffusion_starter_kit.mome.ensemble.predict import DeepEnsembleScorer
from ampdiffusion_starter_kit.mome.filters import read_fasta_sequences

ROOT = Path(__file__).resolve().parents[4]


def evaluate_ensemble(
    csv_path: Path = ROOT / "experimental" / "mic.csv",
    library_fasta: Path = ROOT / "generate_broad_spectrum" / "library.fasta",
    checkpoint_dir: Path | None = None,
) -> None:
    print("=" * 65)
    print("Deep Ensemble Validation Suite")
    print("=" * 65)

    scorer = DeepEnsembleScorer(checkpoint_dir=checkpoint_dir)

    # ── Test 1: In-Distribution Correlation Test ─────────────────────────
    print("\n--- Test 1: In-Distribution Correlation on Experimental Data ---")
    seqs, actual_mics, targets = load_experimental_mic_data(csv_path)

    pred_mics, variances = scorer.predict(seqs)
    efficacies, confidences = scorer.score_for_mome(seqs)

    # Calculate Pearson and Spearman correlations in log space and linear space
    actual_log = np.log10(np.clip(actual_mics, 1e-4, 1e6))
    pred_log = np.log10(np.clip(pred_mics, 1e-4, 1e6))

    r_linear, _ = pearsonr(pred_mics, actual_mics)
    rho_linear, _ = spearmanr(pred_mics, actual_mics)
    r_log, _ = pearsonr(pred_log, actual_log)
    rho_log, _ = spearmanr(pred_log, actual_log)

    mse = float(np.mean((pred_mics - actual_mics) ** 2))
    mae = float(np.mean(np.abs(pred_mics - actual_mics)))

    print(f"Sample count:               {len(seqs)}")
    print(f"Pearson r (linear MIC):     {r_linear:.4f}")
    print(f"Spearman rho (linear MIC):  {rho_linear:.4f}")
    print(f"Pearson r (log10 MIC):      {r_log:.4f}")
    print(f"Spearman rho (log10 MIC):   {rho_log:.4f}")
    print(f"MAE (uM):                   {mae:.2f}")
    print(f"Mean Confidence (in-dist):  {np.mean(confidences):.4f} +/- {np.std(confidences):.4f}")

    # ── Test 2: Uncertainty Calibration vs Generated Library ────────────
    print("\n--- Test 2: Uncertainty Calibration (In-Dist vs Library) ---")
    if library_fasta.exists():
        library_seqs = read_fasta_sequences(library_fasta)
        # Sample 200 random sequences to evaluate out-of-distribution uncertainty
        rng = np.random.default_rng(42)
        sample_indices = rng.choice(len(library_seqs), size=min(200, len(library_seqs)), replace=False)
        sample_seqs = [library_seqs[i] for i in sample_indices]

        _, lib_variances = scorer.predict(sample_seqs)
        _, lib_confidences = scorer.score_for_mome(sample_seqs)

        mean_in_conf = float(np.mean(confidences))
        mean_out_conf = float(np.mean(lib_confidences))
        mean_in_var = float(np.mean(variances))
        mean_out_var = float(np.mean(lib_variances))

        print(f"Library sample count:       {len(sample_seqs)}")
        print(f"Mean Variance (in-dist):    {mean_in_var:.4f}")
        print(f"Mean Variance (library):    {mean_out_var:.4f}")
        print(f"Mean Confidence (in-dist):  {mean_in_conf:.4f}")
        print(f"Mean Confidence (library):  {mean_out_conf:.4f}")

        if mean_out_var >= mean_in_var:
            print("✓ Uncertainty calibration PASS: Ensemble exhibits higher variance on novel generated sequences.")
        else:
            print("Note: Library sequences show low variance; models show high consensus.")
    else:
        print(f"Library FASTA not found at {library_fasta}, skipping test 2.")

    print("\n" + "=" * 65)
    print("Validation Complete")
    print("=" * 65)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Deep Ensemble MIC predictor.")
    parser.add_argument("--data-csv", type=Path, default=ROOT / "experimental" / "mic.csv")
    parser.add_argument("--library-fasta", type=Path, default=ROOT / "generate_broad_spectrum" / "library.fasta")
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    args = parser.parse_args()

    evaluate_ensemble(
        csv_path=args.data_csv,
        library_fasta=args.library_fasta,
        checkpoint_dir=args.checkpoint_dir,
    )


if __name__ == "__main__":
    main()
