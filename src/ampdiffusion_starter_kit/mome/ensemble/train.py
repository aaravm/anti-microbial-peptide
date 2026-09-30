"""Training pipeline for Deep Ensemble of MIC regression models.

Trains N ensemble members with different random seeds and distinct train/validation
splits from experimental/mic.csv, saving the best checkpoint for each ensemble member.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ampdiffusion_starter_kit.mome.ensemble.data import (
    PeptideDataset,
    encode_sequences,
    load_experimental_mic_data,
    target_to_mic,
)
from ampdiffusion_starter_kit.mome.ensemble.model import AMPEnsembleModel

DEFAULT_SEEDS = [42, 101, 314, 777, 999]


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_single_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 300,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 40,
    device: str = "cpu",
) -> Tuple[dict, float]:
    """Train a single model using AdamW and Huber loss with early stopping on validation loss."""
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    criterion = nn.SmoothL1Loss()  # Huber loss is robust to outlier MIC values

    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0

    for epoch in range(1, epochs + 1):
        # Training phase
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * len(X_batch)
        scheduler.step()
        train_loss /= len(train_loader.dataset)

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_val, y_val in val_loader:
                X_val, y_val = X_val.to(device), y_val.to(device)
                val_preds = model(X_val)
                v_loss = criterion(val_preds, y_val)
                val_loss += v_loss.item() * len(X_val)
        val_loss /= len(val_loader.dataset)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience and epoch >= 100:
            break

    return best_state, best_val_loss


def train_ensemble(
    csv_path: Path,
    checkpoint_dir: Path,
    seeds: List[int] = DEFAULT_SEEDS,
    val_ratio: float = 0.2,
    epochs: int = 350,
    lr: float = 1e-3,
    batch_size: int = 16,
    device: str = "cpu",
) -> List[Path]:
    """Train N ensemble models and save weights to checkpoint_dir."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    seqs, mean_mics, targets = load_experimental_mic_data(csv_path)
    encoded_all = encode_sequences(seqs)
    n_samples = len(seqs)
    print(f"Loaded {n_samples} unique peptide measurements from {csv_path}")

    saved_checkpoints: List[Path] = []

    for i, seed in enumerate(seeds):
        set_seed(seed)
        print(f"\n--- Training Ensemble Member {i+1}/{len(seeds)} (seed={seed}) ---")

        # Shuffle indices deterministically for this seed
        indices = np.random.permutation(n_samples)
        val_size = max(1, int(n_samples * val_ratio))
        val_idx = indices[:val_size]
        train_idx = indices[val_size:]

        train_ds = PeptideDataset(encoded_all[train_idx], targets[train_idx])
        val_ds = PeptideDataset(encoded_all[val_idx], targets[val_idx])

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=len(val_ds), shuffle=False)

        model = AMPEnsembleModel()
        best_state, best_val = train_single_model(
            model,
            train_loader,
            val_loader,
            epochs=epochs,
            lr=lr,
            device=device,
        )

        ckpt_path = checkpoint_dir / f"model_seed_{seed}.pt"
        torch.save(
            {
                "seed": seed,
                "model_state_dict": best_state,
                "best_val_loss": best_val,
                "model_kwargs": {
                    "embed_dim": 32,
                    "hidden_dim": 64,
                    "num_layers": 1,
                    "latent_dim": 32,
                },
            },
            ckpt_path,
        )
        saved_checkpoints.append(ckpt_path)
        print(f"Saved ensemble member {i+1} -> {ckpt_path} (Val Loss: {best_val:.4f})")

    return saved_checkpoints


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Deep Ensemble for MIC prediction.")
    parser.add_argument(
        "--data-csv",
        type=Path,
        default=Path("experimental/mic.csv"),
        help="Path to experimental MIC CSV file.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("src/ampdiffusion_starter_kit/mome/ensemble/checkpoints"),
        help="Output directory for trained model weights.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=350,
        help="Max training epochs per model.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=DEFAULT_SEEDS,
        help="Random seeds for ensemble members.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Compute device ('cpu' or 'cuda').",
    )
    args = parser.parse_args()

    train_ensemble(
        csv_path=args.data_csv,
        checkpoint_dir=args.checkpoint_dir,
        seeds=args.seeds,
        epochs=args.epochs,
        device=args.device,
    )


if __name__ == "__main__":
    main()
