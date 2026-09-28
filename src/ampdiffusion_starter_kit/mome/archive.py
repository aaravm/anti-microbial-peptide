"""Stage 5: MOME (Multi-Objective MAP-Elites) archive.

Maintains a local 2D Pareto front per CVT cell. Insertion adds non-dominated
candidates; overfull fronts are pruned by hypervolume contribution.
"""

from __future__ import annotations

import numpy as np

from ampdiffusion_starter_kit.mome.candidate import PeptideCandidate
from ampdiffusion_starter_kit.mome.cvt import CVTGrid
from ampdiffusion_starter_kit.mome.pareto import (
    dominates,
    hypervolume_contribution_2d,
)


class MOMEArchive:
    """MOME archive: one local 2D Pareto front per CVT cell.

    Usage::

        archive = MOMEArchive(grid, max_front_size=25)
        archive.add_batch(candidates)
        print(archive.summary())
    """

    def __init__(self, grid: CVTGrid, max_front_size: int = 25):
        self.grid = grid
        self.max_front_size = max_front_size
        self.cells: dict[int, list[PeptideCandidate]] = {}

    def add(self, candidate: PeptideCandidate) -> bool:
        """Try to insert a candidate into the archive.

        1. Assign it to a CVT cell.
        2. If the cell is empty, add it directly.
        3. If dominated by any resident, discard it.
        4. Remove any residents it dominates.
        5. Add it to the front.
        6. If the front exceeds max_front_size, prune the lowest-HVC member.

        Returns True if the candidate was added.
        """
        cell_idx = self.grid.assign(candidate.behavior_vector)
        new_obj = candidate.objective_vector

        if cell_idx not in self.cells:
            self.cells[cell_idx] = [candidate]
            return True

        front = self.cells[cell_idx]

        # Check if dominated by any existing resident
        for resident in front:
            if dominates(resident.objective_vector, new_obj):
                return False

        # Remove any residents dominated by the new candidate
        front = [r for r in front if not dominates(new_obj, r.objective_vector)]
        front.append(candidate)

        # Prune if over capacity: remove the member with the smallest HVC
        if len(front) > self.max_front_size:
            front = self._prune(front)

        self.cells[cell_idx] = front
        return True

    def _prune(self, front: list[PeptideCandidate]) -> list[PeptideCandidate]:
        """Remove the candidate with the smallest hypervolume contribution."""
        objectives = np.array([c.objective_vector for c in front])

        # Reference point: worst value in each objective minus epsilon
        ref_point = objectives.min(axis=0) - 1e-6

        # Find the candidate contributing the least hypervolume
        hvcs = [
            hypervolume_contribution_2d(objectives, i, ref_point)
            for i in range(len(front))
        ]
        worst_idx = int(np.argmin(hvcs))
        front.pop(worst_idx)
        return front

    def add_batch(self, candidates: list[PeptideCandidate]) -> int:
        """Add multiple candidates to the archive.

        Returns the number of candidates successfully added.
        """
        added = 0
        total = len(candidates)
        for i, c in enumerate(candidates):
            if self.add(c):
                added += 1
            if (i + 1) % 5000 == 0 or (i + 1) == total:
                print(f"  archive: processed {i + 1}/{total} — "
                      f"{self.filled_cells()} cells, {self.total_candidates()} candidates")
        return added

    def filled_cells(self) -> int:
        """Number of non-empty cells."""
        return len(self.cells)

    def total_candidates(self) -> int:
        """Total candidates across all cells."""
        return sum(len(front) for front in self.cells.values())

    def summary(self) -> str:
        """Human-readable summary of the archive."""
        front_sizes = [len(f) for f in self.cells.values()]
        if not front_sizes:
            return "MOME Archive: empty"
        return (
            f"MOME Archive: {self.filled_cells()}/{self.grid.n_cells} cells filled, "
            f"{self.total_candidates()} total candidates\n"
            f"  Front sizes — min: {min(front_sizes)}, max: {max(front_sizes)}, "
            f"mean: {np.mean(front_sizes):.1f}, median: {np.median(front_sizes):.0f}"
        )
