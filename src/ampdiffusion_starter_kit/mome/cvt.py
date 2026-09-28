"""Stage 3: Centroidal Voronoi Tessellation (CVT) grid.

Partitions the 3D behavioral space (Charge × Hydrophobicity × Length) into
evenly-sized niches using K-Means on uniformly distributed samples. This
ensures the grid tiles the space uniformly rather than clustering where the
data is dense.
"""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


class CVTGrid:
    """A CVT grid that maps behavioral vectors to cell indices.

    Usage::

        grid = CVTGrid(n_cells=175).fit(behavior_vectors)
        cell_idx = grid.assign(candidate.behavior_vector)
    """

    def __init__(
        self,
        n_cells: int = 175,
        n_uniform_samples: int = 100_000,
        random_state: int = 42,
    ):
        self.n_cells = n_cells
        self.n_uniform_samples = n_uniform_samples
        self.random_state = random_state
        self.centroids: np.ndarray | None = None   # shape (n_cells, 3), in scaled space
        self.scaler: StandardScaler | None = None
        self._bounds: tuple[np.ndarray, np.ndarray] | None = None  # (mins, maxs)

    def fit(self, behavior_vectors: np.ndarray) -> CVTGrid:
        """Compute CVT centroids from uniform samples spanning the data's bounds.

        Steps:
            1. Find per-dimension min/max from the real data.
            2. Generate ``n_uniform_samples`` uniform random points within those bounds.
            3. Fit a StandardScaler on the uniform samples.
            4. Run KMeans on the scaled uniform samples to find centroids.

        Args:
            behavior_vectors: shape (N, 3) — [charge, hydrophobicity, length] for
                each candidate that passed hard filters.

        Returns:
            self (for chaining).
        """
        mins = behavior_vectors.min(axis=0)
        maxs = behavior_vectors.max(axis=0)
        self._bounds = (mins, maxs)

        # Generate uniform samples within the bounding box
        rng = np.random.default_rng(self.random_state)
        uniform = rng.uniform(mins, maxs, size=(self.n_uniform_samples, 3))

        # Standardize so KMeans distances are balanced across dimensions
        self.scaler = StandardScaler().fit(uniform)
        uniform_scaled = self.scaler.transform(uniform)

        # Run KMeans to find evenly-spaced centroids
        km = KMeans(n_clusters=self.n_cells, random_state=self.random_state, n_init=10)
        km.fit(uniform_scaled)
        self.centroids = km.cluster_centers_  # shape (n_cells, 3) in scaled space

        print(f"CVT grid fitted: {self.n_cells} cells over bounds "
              f"[{mins} .. {maxs}]")
        return self

    def assign(self, behavior_vector: np.ndarray) -> int:
        """Assign a single behavior vector to its nearest cell.

        Args:
            behavior_vector: shape (3,) — [charge, hydrophobicity, length].

        Returns:
            Cell index in [0, n_cells - 1].
        """
        assert self.centroids is not None, "Call fit() first"
        scaled = self.scaler.transform(behavior_vector.reshape(1, -1))  # type: ignore[union-attr]
        dists = np.linalg.norm(self.centroids - scaled, axis=1)
        return int(np.argmin(dists))

    def assign_batch(self, behavior_vectors: np.ndarray) -> np.ndarray:
        """Assign multiple behavior vectors to their nearest cells.

        Args:
            behavior_vectors: shape (N, 3).

        Returns:
            Array of cell indices, shape (N,).
        """
        assert self.centroids is not None, "Call fit() first"
        scaled = self.scaler.transform(behavior_vectors)  # type: ignore[union-attr]
        # Compute pairwise distances: (N, n_cells)
        # Using broadcasting: (N, 1, 3) - (1, n_cells, 3) -> (N, n_cells, 3) -> norm
        diffs = scaled[:, np.newaxis, :] - self.centroids[np.newaxis, :, :]
        dists = np.linalg.norm(diffs, axis=2)
        return np.argmin(dists, axis=1)

    def cell_occupancy(self, assignments: np.ndarray) -> dict[int, int]:
        """Return a mapping of cell_index -> number of candidates assigned.

        Args:
            assignments: array of cell indices from assign_batch.

        Returns:
            Dict with only non-zero cells.
        """
        unique, counts = np.unique(assignments, return_counts=True)
        return {int(k): int(v) for k, v in zip(unique, counts)}
