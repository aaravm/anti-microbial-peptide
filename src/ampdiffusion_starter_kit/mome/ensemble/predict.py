"""Inference and confidence estimation using Deep Ensemble.

Loads trained ensemble checkpoints, predicts MIC distributions across ensemble
members, and computes mean efficacy and calibrated confidence scores for the MOME pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch

from ampdiffusion_starter_kit.mome.ensemble.data import encode_sequences, target_to_mic
from ampdiffusion_starter_kit.mome.ensemble.model import AMPEnsembleModel

DEFAULT_CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoints"


class DeepEnsembleScorer:
    """Predictive scoring and uncertainty quantification using Deep Ensembles."""

    def __init__(self, checkpoint_dir: Path | str | None = None, device: str = "cpu"):
        self.device = device
        if checkpoint_dir is None:
            self.checkpoint_dir = DEFAULT_CHECKPOINT_DIR
        else:
            self.checkpoint_dir = Path(checkpoint_dir)

        self.models: List[AMPEnsembleModel] = []
        self._load_checkpoints()

    def _load_checkpoints(self) -> None:
        """Find and load all .pt model weights in the checkpoint directory."""
        if not self.checkpoint_dir.exists():
            raise FileNotFoundError(
                f"Checkpoint directory not found: {self.checkpoint_dir}. "
                "Please run `uv run python -m ampdiffusion_starter_kit.mome.ensemble.train` first."
            )

        ckpt_files = sorted(self.checkpoint_dir.glob("model_seed_*.pt"))
        if not ckpt_files:
            # Also try matching any .pt file
            ckpt_files = sorted(self.checkpoint_dir.glob("*.pt"))

        if not ckpt_files:
            raise FileNotFoundError(
                f"No checkpoint files found in {self.checkpoint_dir}. "
                "Please train the ensemble first."
            )

        self.models = []
        for ckpt_path in ckpt_files:
            data = torch.load(ckpt_path, map_location=self.device, weights_only=False)
            kwargs = data.get("model_kwargs", {})
            model = AMPEnsembleModel(**kwargs)
            model.load_state_dict(data["model_state_dict"])
            model.to(self.device)
            model.eval()
            self.models.append(model)

        print(f"Loaded {len(self.models)} deep ensemble model(s) from {self.checkpoint_dir}")

    @torch.no_grad()
    def predict(
        self, sequences: List[str], batch_size: int = 256
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Predict MIC in uM and ensemble variance for a list of peptide sequences.

        Args:
            sequences: List of peptide amino acid strings
            batch_size: Evaluation batch size

        Returns:
            mean_mic: shape (N,) - mean predicted MIC in uM
            target_variance: shape (N,) - variance of predictions across ensemble members
        """
        if not sequences:
            return np.array([], dtype=np.float32), np.array([], dtype=np.float32)

        encoded = encode_sequences(sequences)
        n_samples = len(sequences)
        n_models = len(self.models)

        # Store predictions: shape (n_models, n_samples)
        all_targets = np.zeros((n_models, n_samples), dtype=np.float32)

        for m_idx, model in enumerate(self.models):
            preds_list = []
            for start in range(0, n_samples, batch_size):
                batch_x = torch.tensor(encoded[start : start + batch_size], dtype=torch.long, device=self.device)
                batch_pred = model(batch_x).squeeze(-1).cpu().numpy()
                preds_list.append(batch_pred)
            all_targets[m_idx] = np.concatenate(preds_list) if len(preds_list) > 1 else preds_list[0]

        mean_targets = np.mean(all_targets, axis=0)
        target_variance = np.var(all_targets, axis=0) if n_models > 1 else np.zeros(n_samples, dtype=np.float32)

        mean_mic = target_to_mic(mean_targets)
        return mean_mic, target_variance

    def score_for_mome(
        self, sequences: List[str], batch_size: int = 256
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute (efficacy, confidence) formatted specifically for the MOME pipeline.

        - Efficacy = -mean_mic (higher is better, e.g. -1.0 uM > -64.0 uM)
        - Confidence = 1 / (1 + variance), bounded in [0, 1] (1 = high agreement)

        Returns:
            efficacy: shape (N,)
            confidence: shape (N,)
        """
        mean_mic, variance = self.predict(sequences, batch_size=batch_size)
        efficacy = -mean_mic.astype(np.float32)
        confidence = (1.0 / (1.0 + variance)).astype(np.float32)
        return efficacy, confidence
