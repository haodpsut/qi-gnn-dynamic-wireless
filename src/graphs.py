"""
Controlled synthetic graph families for the mechanism demo (Exp A) and the
evaluation-pitfall demo (Exp C). Kept tiny and deterministic so every figure is
reproducible from parameters alone.
"""

import numpy as np
from collections import deque


def path_graph(n):
    """1-D path of n nodes. Node coordinate = its index (for spread variance)."""
    A = np.zeros((n, n), dtype=float)
    for i in range(n - 1):
        A[i, i + 1] = A[i + 1, i] = 1.0
    pos = np.arange(n, dtype=float)
    return A, pos


def grid_graph(h, w):
    """2-D grid h x w with 4-neighbour connectivity. Returns A and (n,2) coords."""
    n = h * w
    A = np.zeros((n, n), dtype=float)

    def vid(r, c):
        return r * w + c

    coords = np.zeros((n, 2), dtype=float)
    for r in range(h):
        for c in range(w):
            coords[vid(r, c)] = (r, c)
            if c + 1 < w:
                A[vid(r, c), vid(r, c + 1)] = A[vid(r, c + 1), vid(r, c)] = 1.0
            if r + 1 < h:
                A[vid(r, c), vid(r + 1, c)] = A[vid(r + 1, c), vid(r, c)] = 1.0
    return A, coords


def bfs_hops(A, src):
    """Hop distance from src to every node (np.inf if unreachable)."""
    n = A.shape[0]
    dist = np.full(n, np.inf)
    dist[src] = 0
    q = deque([src])
    nbrs = [np.nonzero(A[i])[0] for i in range(n)]
    while q:
        u = q.popleft()
        for v in nbrs[u]:
            if dist[v] == np.inf:
                dist[v] = dist[u] + 1
                q.append(v)
    return dist


def largest_cc(A):
    """Return node indices of the largest connected component of A."""
    n = A.shape[0]
    seen = np.zeros(n, dtype=bool)
    best = []
    nbrs = [np.nonzero(A[i])[0] for i in range(n)]
    for s in range(n):
        if seen[s]:
            continue
        comp = []
        q = deque([s])
        seen[s] = True
        while q:
            u = q.popleft()
            comp.append(u)
            for v in nbrs[u]:
                if not seen[v]:
                    seen[v] = True
                    q.append(v)
        if len(comp) > len(best):
            best = comp
    return np.array(sorted(best))


def eccentricity(A, src):
    """Eccentricity of src within its component (max finite hop distance)."""
    d = bfs_hops(A, src)
    finite = d[np.isfinite(d)]
    return float(finite.max()) if finite.size else 0.0


def diameter(A):
    """Graph diameter over the largest component (max pairwise hop distance)."""
    cc = largest_cc(A)
    Asub = A[np.ix_(cc, cc)]
    dmax = 0.0
    for s in range(len(cc)):
        d = bfs_hops(Asub, s)
        finite = d[np.isfinite(d)]
        if finite.size:
            dmax = max(dmax, finite.max())
    return float(dmax)
