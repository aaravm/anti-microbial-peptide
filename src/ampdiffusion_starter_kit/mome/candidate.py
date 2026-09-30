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

from Bio.SeqUtils.ProtParam import ProteinAnalysis

# ---------------------------------------------------------------------------
# Biophysics-based heuristics (replacing mock scorers)
# ---------------------------------------------------------------------------

def calculate_biophysics(seq: str, charge: float, hydrophobicity: float) -> PeptideCandidate:
    """Create a PeptideCandidate with real biophysical heuristics for safety/stability."""
    pa = ProteinAnalysis(seq)
    
    # Half-life proxy: aliphatic index (higher = more thermostable)
    try:
        aliphatic = pa.aliphatic_index()
    except Exception:
        aliphatic = 0.0
        
    # Safety proxy: instability index (lower = more stable in vivo / less likely to degrade randomly into toxic fragments)
    # MOME maximizes, so we negate it.
    try:
        instability = pa.instability_index()
    except Exception:
        instability = 100.0
    safety_score = -instability
    
    # Solubility proxy: GRAVY score (lower = more hydrophilic/soluble)
    # MOME maximizes, so we negate it.
    try:
        gravy = pa.gravy()
    except Exception:
        gravy = 0.0
    solubility_score = -gravy
    
    # Immunogenicity: longer peptides are generally more immunogenic.
    # Score [0, 1] where 1 is safe (short).
    immuno_score = max(0.0, 1.0 - (len(seq) / 50.0))
    
    return PeptideCandidate(
        sequence=seq,
        charge=charge,
        hydrophobicity=hydrophobicity,
        length=len(seq),
        efficacy=0.0,  # Filled by APEX or mock later
        safety=safety_score,
        half_life=aliphatic,
        solubility=solubility_score,
        confidence=1.0, # Filled by APEX or mock later
        immunogenicity=immuno_score,
    )
