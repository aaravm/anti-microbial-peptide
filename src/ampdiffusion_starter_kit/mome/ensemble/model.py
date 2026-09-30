"""Neural network architecture for peptide MIC regression and ensemble uncertainty.

A compact Bidirectional GRU with self-attention pooling and multilayer
perceptron head, designed to train stably on small peptide datasets while
providing calibrated predictive distributions in an ensemble.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ampdiffusion_starter_kit.mome.ensemble.data import MAX_LEN, VOCAB_SIZE


class AttentionPooling(nn.Module):
    """Self-attention pooling over variable-length sequence representations with padding masking."""

    def __init__(self, in_dim: int):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(in_dim, in_dim // 2),
            nn.Tanh(),
            nn.Linear(in_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Args:

        x: (batch_size, seq_len, in_dim)
        mask: (batch_size, seq_len) boolean tensor where True indicates valid
        token
        """
        scores = self.attn(x)  # (batch_size, seq_len, 1)
        if mask is not None:
            # Mask out padding tokens with a large negative value
            scores = scores.masked_fill(~mask.unsqueeze(-1), -1e9)
        weights = F.softmax(scores, dim=1)  # (batch_size, seq_len, 1)
        pooled = torch.sum(weights * x, dim=1)  # (batch_size, in_dim)
        return pooled


class AMPEnsembleModel(nn.Module):
    """Compact BiGRU + Attention regression model for peptide efficacy prediction."""

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        embed_dim: int = 32,
        hidden_dim: int = 64,
        num_layers: int = 1,
        latent_dim: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        gru_out_dim = hidden_dim * 2

        self.layer_norm = nn.LayerNorm(gru_out_dim)
        self.attn_pool = AttentionPooling(gru_out_dim)

        self.head = nn.Sequential(
            nn.Linear(gru_out_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim, latent_dim // 2),
            nn.LayerNorm(latent_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (batch_size, seq_len) token IDs

        Returns:
            (batch_size, 1) predicted regression target: -log10(MIC / 1e6)
        """
        mask = x != 0  # Valid tokens are non-zero
        emb = self.embedding(x)  # (batch_size, seq_len, embed_dim)
        out, _ = self.gru(emb)
        out = self.layer_norm(out)
        pooled = self.attn_pool(out, mask=mask)
        pred = self.head(pooled)
        return pred
