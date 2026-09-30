"""Data loading and preprocessing for Deep Ensemble MIC prediction.

Loads experimental MIC measurements, computes mean MIC per peptide across
pathogens, and prepares tokenized sequence tensors for PyTorch models.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

MAX_LEN = 52  # 1 start token + up to 50 aa + 1 end token
STANDARD_AA = "ACDEFGHIKLMNPQRSTVWY"


def make_vocab() -> Tuple[Dict[str, int], Dict[int, str]]:
    """Create token-to-index and index-to-token mappings.

    0: pad
    1: start token
    2: end token
    3-22: standard 20 amino acids
    """
    word2idx = {"0": 0, "1": 1, "2": 2}
    for i, aa in enumerate(STANDARD_AA, start=3):
        word2idx[aa] = i
    idx2word = {idx: token for token, idx in word2idx.items()}
    return word2idx, idx2word


VOCAB, IDX2WORD = make_vocab()
VOCAB_SIZE = len(VOCAB)  # 23


def encode_sequences(seq_list: List[str], max_len: int = MAX_LEN) -> np.ndarray:
    """Encode peptide string sequences into integer token arrays padded to max_len."""
    X = np.zeros((len(seq_list), max_len), dtype=np.int64)
    for i, seq in enumerate(seq_list):
        clean_seq = seq.strip().upper()
        # Truncate if longer than max_len - 2 to leave space for start ('1') and end ('2')
        if len(clean_seq) > max_len - 2:
            clean_seq = clean_seq[: max_len - 2]
        token_str = "1" + clean_seq + "2"
        for j, char in enumerate(token_str):
            if char in VOCAB:
                X[i, j] = VOCAB[char]
            else:
                X[i, j] = 0  # Unknown / pad
    return X


def mic_to_target(mic_um: float | np.ndarray) -> float | np.ndarray:
    """Convert MIC (in uM) to APEX-style regression target: -log10(MIC / 1e6).

    Lower MIC (more potent) yields higher target value.
    For example:
        MIC = 64 uM -> 4.1938
        MIC = 1 uM  -> 6.0000
        MIC = 0.5 uM -> 6.3010
    """
    mic_arr = np.asarray(mic_um, dtype=np.float32)
    # Clip to prevent log10(0)
    mic_arr = np.clip(mic_arr, 1e-6, 1e6)
    target = -np.log10(mic_arr / 1e6)
    if isinstance(mic_um, (float, int)):
        return float(target)
    return target


def target_to_mic(target: float | np.ndarray) -> float | np.ndarray:
    """Invert regression target back to MIC in uM: MIC = 10^(6 - target)."""
    target_arr = np.asarray(target, dtype=np.float32)
    mic = 10.0 ** (6.0 - target_arr)
    if isinstance(target, (float, int)):
        return float(mic)
    return mic


def load_experimental_mic_data(csv_path: Path) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """Load experimental/mic.csv and aggregate mean MIC per unique peptide.

    Returns:
        sequences: List of peptide sequence strings
        mean_mics: 1D numpy array of mean MIC values in uM
        targets: 1D numpy array of APEX regression targets
    """
    df = pd.read_csv(csv_path)

    # Clean and convert mic column to numeric, treating right-censored (>64) as 64
    df["mic"] = pd.to_numeric(df["mic"], errors="coerce").fillna(64.0)
    df["sequence"] = df["sequence"].astype(str).str.strip().str.upper()

    # Aggregate by sequence
    grouped = df.groupby("sequence")["mic"].mean().reset_index()

    sequences = grouped["sequence"].tolist()
    mean_mics = grouped["mic"].to_numpy(dtype=np.float32)
    targets = mic_to_target(mean_mics).astype(np.float32)

    return sequences, mean_mics, targets


class PeptideDataset(Dataset):
    """PyTorch Dataset for peptide sequence inputs and scalar regression targets."""

    def __init__(self, encoded_seqs: np.ndarray, targets: np.ndarray | None = None):
        self.X = torch.tensor(encoded_seqs, dtype=torch.long)
        if targets is not None:
            self.y = torch.tensor(targets, dtype=torch.float32).unsqueeze(-1)
        else:
            self.y = None

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]
