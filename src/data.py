"""
Case-study dataset: node-potential / reachability on a dynamic LEO graph.

Each sample is a graph snapshot + a randomly chosen destination. The model sees a
single seeded signal at the destination and must propagate it to recover the
geodesic-distance field. This rewards long-range operators and is independent of
the traffic-engineering pipeline of the qwgnn paper (no BPR/UE/TTT here).
"""

import numpy as np

from .constellation import grid_isl_graph, Walker
from .graphs import bfs_hops, largest_cc, eccentricity
from .operators import laplacian, eig_sym


def _restrict_to_cc(A):
    cc = largest_cc(A)
    return A[np.ix_(cc, cc)]


class Sample:
    """One training/eval instance: graph + eig + features + target + mask."""

    def __init__(self, A, dest, norm="eccentricity"):
        self.A = A
        self.n = A.shape[0]
        self.dest = dest
        # geodesic hop field from the destination
        d = bfs_hops(A, dest)
        reachable = np.isfinite(d)
        d = np.where(reachable, d, 0.0)
        self.hops = d
        self.reachable = reachable
        ecc = max(eccentricity(A, dest), 1.0)
        self.ecc = ecc
        if norm == "diameter":
            # PITFALL normalization (Exp C): scale by a global graph constant
            from .graphs import diameter
            denom = max(diameter(A), 1.0)
        elif norm == "eccentricity":
            denom = ecc
        else:  # raw
            denom = 1.0
        self.y = (d / denom).astype(np.float32)
        # input features: [is_destination, normalized degree]
        deg = A.sum(1)
        deg = deg / max(deg.max(), 1.0)
        seed = np.zeros(self.n, dtype=np.float32)
        seed[dest] = 1.0
        self.X = np.stack([seed, deg.astype(np.float32)], axis=1)
        # eigendecomposition of the (unnormalized) Laplacian, reused by all ops
        w, U = eig_sym(laplacian(A, normalized=False))
        self.w = w.astype(np.float64)
        self.U = U.astype(np.float64)
        # variant eigenbases (lazily filled by ensure_variant_eigs for Exp B2)
        self.w_A = self.U_A = None      # adjacency A
        self.w_S = self.U_S = None      # symmetric-normalized Laplacian

    def ensure_variant_eigs(self):
        """Eigendecompose A and L_sym for the quantum-walk-variant probe."""
        if self.w_A is None:
            wa, Ua = eig_sym(self.A.astype(float))
            self.w_A, self.U_A = wa.astype(np.float64), Ua.astype(np.float64)
        if self.w_S is None:
            ws, Us = eig_sym(laplacian(self.A, normalized=True))
            self.w_S, self.U_S = ws.astype(np.float64), Us.astype(np.float64)


def make_leo_samples(walker, n_samples, slot_s=30.0, seam=False,
                     norm="eccentricity", seed=0):
    """Build n_samples instances from distinct snapshots/destinations."""
    rng = np.random.default_rng(seed)
    samples = []
    period = walker.period_s
    for _ in range(n_samples):
        t = float(rng.uniform(0, period))
        A, _ = grid_isl_graph(walker, t, seam=seam)
        A = _restrict_to_cc(A)
        dest = int(rng.integers(0, A.shape[0]))
        samples.append(Sample(A, dest, norm=norm))
    return samples
