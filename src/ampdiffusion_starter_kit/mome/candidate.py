"""Stage 1: PeptideCandidate dataclass and mock scorers.

Defines the core data structure for the MOME pipeline and provides deterministic
mock scoring functions for development and testing.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field

import numpy as np


@dataclass
class PeptideCandidate:
    """A single peptide candidate with behavioral features and objective scores.

    Behavioral features define WHERE in the MAP-Elites grid the candidate sits.
    Objective scores define HOW GOOD the candidate is within its cell.

    Conventions (all "higher = better" internally):
        efficacy:       negated MIC (e.g. -10 means MIC=10 uM; higher = more potent)
        safety:         min(HC50, Cytotox_IC50); higher = safer
        half_life:      hours; higher = better
        solubility:     [0, 1]; higher = better
        confidence:     [0, 1]; 1 = model agrees, 0 = high uncertainty
        immunogenicity: [0, 1]; 1 = non-immunogenic (safe), 0 = highly immunogenic
    """

    # Identity
    sequence: str

    # Behavioral features (define the grid position)
    charge: float = 0.0
    hydrophobicity: float = 0.0
    length: int = 0

    # Core 2D Pareto objectives
    efficacy: float = 0.0   # -MIC, higher = better
    safety: float = 0.0     # higher = safer

    # Tie-breaker bonuses (used in final extraction)
    half_life: float = 0.0
    solubility: float = 0.0

    # Penalty scores (used as hard-threshold filters)
    confidence: float = 1.0
    immunogenicity: float = 1.0

    @property
    def behavior_vector(self) -> np.ndarray:
        """Return [charge, hydrophobicity, length] as a numpy array."""
        return np.array([self.charge, self.hydrophobicity, self.length], dtype=np.float64)

    @property
    def objective_vector(self) -> np.ndarray:
        """Return [efficacy, safety] as a numpy array (the 2D Pareto axes)."""
        return np.array([self.efficacy, self.safety], dtype=np.float64)

    def __repr__(self) -> str:
        seq_display = self.sequence[:15] + "..." if len(self.sequence) > 15 else self.sequence
        return (
            f"PeptideCandidate(seq='{seq_display}', "
            f"eff={self.efficacy:.2f}, safety={self.safety:.2f})"
        )


# ---------------------------------------------------------------------------
# Mock scorers — deterministic via sequence-hash seeding
# ---------------------------------------------------------------------------

def _seq_rng(seq: str, salt: str = "") -> random.Random:
    """Create a deterministic RNG seeded by the sequence + salt."""
    h = hashlib.sha256((seq + salt).encode()).hexdigest()
    return random.Random(int(h, 16) % (2**32))


def mock_score_efficacy(seq: str) -> float:
    """Mock efficacy: negated MIC in [-200, -1]. Higher = more potent."""
    return _seq_rng(seq, "efficacy").uniform(-200.0, -1.0)


def mock_score_safety(seq: str) -> float:
    """Mock safety score in [0, 100]. Higher = safer."""
    return _seq_rng(seq, "safety").uniform(0.0, 100.0)


def mock_score_half_life(seq: str) -> float:
    """Mock half-life in [0, 24] hours."""
    return _seq_rng(seq, "half_life").uniform(0.0, 24.0)


def mock_score_solubility(seq: str) -> float:
    """Mock solubility in [0, 1]."""
    return _seq_rng(seq, "solubility").uniform(0.0, 1.0)


def mock_score_confidence(seq: str) -> float:
    """Mock model confidence in [0, 1]. 1 = confident."""
    return _seq_rng(seq, "confidence").uniform(0.0, 1.0)


def mock_score_immunogenicity(seq: str) -> float:
    """Mock immunogenicity safety in [0, 1]. 1 = non-immunogenic."""
    return _seq_rng(seq, "immunogenicity").uniform(0.0, 1.0)


def create_candidate_mock(seq: str, charge: float, hydrophobicity: float) -> PeptideCandidate:
    """Create a PeptideCandidate with mock scores and precomputed behavioral features."""
    return PeptideCandidate(
        sequence=seq,
        charge=charge,
        hydrophobicity=hydrophobicity,
        length=len(seq),
        efficacy=mock_score_efficacy(seq),
        safety=mock_score_safety(seq),
        half_life=mock_score_half_life(seq),
        solubility=mock_score_solubility(seq),
        confidence=mock_score_confidence(seq),
        immunogenicity=mock_score_immunogenicity(seq),
    )
