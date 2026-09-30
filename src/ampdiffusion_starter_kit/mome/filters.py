"""Stage 2: Hard filters and behavioral descriptor computation.

Applies sequence validity checks, novelty guards, and penalty thresholds.
Computes physicochemical behavioral descriptors (charge, hydrophobicity, length)
using BioPython's ProtParam.
"""

from __future__ import annotations

from pathlib import Path

import Levenshtein
from Bio.SeqUtils.ProtParam import ProteinAnalysis

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STANDARD_AA_SET = set("ACDEFGHIKLMNPQRSTVWY")
MIN_LENGTH = 8
MAX_LENGTH = 50
SIMILARITY_THRESHOLD = 0.8   # Levenshtein ratio; challenge hard rule
CONFIDENCE_THRESHOLD = 0.3   # reject if model confidence < this
IMMUNOGENICITY_THRESHOLD = 0.3  # reject if immunogenicity safety < this


# ---------------------------------------------------------------------------
# Sequence validity
# ---------------------------------------------------------------------------

def is_valid_sequence(seq: str) -> bool:
    """Check that the sequence uses only the 20 canonical amino acids and is 8-50 residues."""
    return MIN_LENGTH <= len(seq) <= MAX_LENGTH and set(seq).issubset(STANDARD_AA_SET)


def passes_novelty_check(
    seq: str, references: set[str], threshold: float = SIMILARITY_THRESHOLD
) -> bool:
    """Return True if the sequence is sufficiently novel vs. all references.

    A sequence passes if its Levenshtein ratio to EVERY reference is <= threshold.
    """
    for ref in references:
        if Levenshtein.ratio(seq, ref) > threshold:
            return False
    return True


def passes_penalty_thresholds(
    confidence: float,
    immunogenicity: float,
    efficacy: float,
    conf_thresh: float = CONFIDENCE_THRESHOLD,
    immuno_thresh: float = IMMUNOGENICITY_THRESHOLD,
    efficacy_thresh: float = -200.0,
) -> bool:
    """Return True if the candidate passes all penalty hard thresholds."""
    return (
        confidence >= conf_thresh
        and immunogenicity >= immuno_thresh
        and efficacy >= efficacy_thresh
    )


# ---------------------------------------------------------------------------
# Behavioral descriptors
# ---------------------------------------------------------------------------

def compute_charge(seq: str) -> float:
    """Net charge at pH 7.0 via BioPython ProteinAnalysis."""
    return ProteinAnalysis(seq).charge_at_pH(7.0)


def compute_hydrophobicity(seq: str) -> float:
    """GRAVY (Grand Average of Hydropathicity) via BioPython ProteinAnalysis."""
    return ProteinAnalysis(seq).gravy()


def compute_behavioral_descriptors(seq: str) -> tuple[float, float, int]:
    """Return (charge, hydrophobicity, length) for a sequence."""
    return compute_charge(seq), compute_hydrophobicity(seq), len(seq)


# ---------------------------------------------------------------------------
# FASTA I/O
# ---------------------------------------------------------------------------

def read_fasta_sequences(path: Path) -> list[str]:
    """Read a FASTA file and return a list of uppercase sequences."""
    sequences: list[str] = []
    parts: list[str] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if parts:
                sequences.append("".join(parts))
                parts = []
        else:
            parts.append(line.upper())
    if parts:
        sequences.append("".join(parts))
    return sequences


# ---------------------------------------------------------------------------
# Combined hard-filter pass
# ---------------------------------------------------------------------------

def apply_hard_filters(sequences: list[str], references: set[str]) -> list[str]:
    """Apply sequence-validity and novelty checks, printing progress.

    Returns the list of sequences that pass both filters.
    """
    total = len(sequences)
    print(f"Applying hard filters to {total} sequences...")

    filtered: list[str] = []
    for i, seq in enumerate(sequences):
        if (i + 1) % 5000 == 0 or (i + 1) == total:
            print(f"  processed {i + 1}/{total} — kept {len(filtered)} so far")
        if not is_valid_sequence(seq):
            continue
        if not passes_novelty_check(seq, references):
            continue
        filtered.append(seq)

    print(f"Hard filters complete: kept {len(filtered)}/{total} sequences.")
    return filtered
