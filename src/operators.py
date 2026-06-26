"""
Graph propagation operators: the unifying view of the tutorial.

GCN, heat diffusion, and continuous-time quantum walks are all functions Phi(L)
of the graph Laplacian. This module builds each operator from a single
eigendecomposition so they are directly comparable and param-matched.
"""

import numpy as np


def laplacian(A, normalized=False):
    """Combinatorial (D - A) or symmetric-normalized Laplacian of adjacency A."""
    A = np.asarray(A, dtype=float)
    n = A.shape[0]
    d = A.sum(axis=1)
    if normalized:
        dinv = 1.0 / np.sqrt(np.maximum(d, 1e-12))
        return np.eye(n) - (dinv[:, None] * A * dinv[None, :])
    return np.diag(d) - A


def eig_sym(L):
    """Eigendecomposition of a symmetric Laplacian: L = U diag(w) U^T."""
    w, U = np.linalg.eigh(L)
    return w, U


def gcn_propagator(A):
    """Symmetric-normalized adjacency with self-loops: hatA = D~^-1/2 A~ D~^-1/2."""
    A = np.asarray(A, dtype=float)
    n = A.shape[0]
    At = A + np.eye(n)
    d = At.sum(axis=1)
    dinv = 1.0 / np.sqrt(np.maximum(d, 1e-12))
    return dinv[:, None] * At * dinv[None, :]


def heat_propagator(w, U, t):
    """Diffusion kernel exp(-t L) from eigenpairs. Row-stochastic, nonneg."""
    return (U * np.exp(-t * w)) @ U.T


def qwalk_unitary(w, U, t):
    """Continuous-time quantum walk unitary exp(-i t L) from eigenpairs."""
    return (U * np.exp(-1j * t * w)) @ U.conj().T


def qwalk_prob(w, U, t):
    """Quantum-walk transition probability Pi(t)_ij = |[exp(-i t L)]_ij|^2.

    Row-stochastic and real, so it plugs into the same GNN layer as the heat
    kernel and stays param-matched.
    """
    Uu = qwalk_unitary(w, U, t)
    return np.abs(Uu) ** 2


def spread_variance(prob_row, positions):
    """Second moment (spatial variance) of a propagated distribution.

    prob_row: a probability vector over nodes (a row of a propagator).
    positions: 1-D coordinate of each node (e.g. index on a path graph).
    Used by Exp A to contrast ballistic (Var ~ t^2) vs diffusive (Var ~ t).
    """
    p = np.asarray(prob_row, dtype=float)
    p = p / max(p.sum(), 1e-12)
    mean = np.sum(p * positions)
    return float(np.sum(p * (positions - mean) ** 2))
