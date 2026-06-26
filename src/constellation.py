"""
Walker-delta LEO constellation -> time-varying +Grid ISL graph sequence.

Pure NumPy circular-orbit propagation (no skyfield / ns-3), so the whole pipeline
is reproducible from parameters alone. Adapted, for this tutorial, from the
author's qwgnn-leo-routing codebase; only the time-varying topology generator is
reused here. The tutorial task built on top (node-potential / reachability) is
new and independent of that paper's traffic-engineering pipeline.
"""

import numpy as np

RE_KM = 6371.0          # Earth radius
MU = 398600.4418        # km^3 / s^2, Earth gravitational parameter
C_KM_S = 299792.458     # speed of light


class Walker:
    """Walker-delta constellation: T sats, P planes, phasing F, inc, altitude."""

    def __init__(self, total, planes, phasing_f, inclination_deg, altitude_km):
        assert total % planes == 0, "total must be divisible by planes"
        self.T = total
        self.P = planes
        self.S = total // planes
        self.F = phasing_f
        self.inc = np.deg2rad(inclination_deg)
        self.r = RE_KM + altitude_km
        self.n_mean = np.sqrt(MU / self.r ** 3)
        self.period_s = 2 * np.pi / self.n_mean
        p_idx = np.repeat(np.arange(self.P), self.S)
        s_idx = np.tile(np.arange(self.S), self.P)
        self.raan = 2 * np.pi * p_idx / self.P
        self.u0 = (2 * np.pi * s_idx / self.S
                   + 2 * np.pi * self.F * p_idx / self.T)
        self.p_idx = p_idx
        self.s_idx = s_idx

    def positions(self, t_s):
        """ECI positions (T,3) km at time t seconds."""
        u = self.u0 + self.n_mean * t_s
        cu, su = np.cos(u), np.sin(u)
        ci, si = np.cos(self.inc), np.sin(self.inc)
        cO, sO = np.cos(self.raan), np.sin(self.raan)
        x = self.r * (cO * cu - sO * su * ci)
        y = self.r * (sO * cu + cO * su * ci)
        z = self.r * (su * si)
        return np.stack([x, y, z], axis=1)

    def latitudes(self, t_s):
        u = self.u0 + self.n_mean * t_s
        return np.arcsin(np.sin(u) * np.sin(self.inc))


def grid_isl_graph(walker, t_s, polar_cutoff_deg=70.0, seam=True):
    """Build the +Grid ISL adjacency A (T,T) and delay weights W (s) at time t."""
    T, P, S = walker.T, walker.P, walker.S
    pos = walker.positions(t_s)
    lat = np.abs(walker.latitudes(t_s))
    polar = lat > np.deg2rad(polar_cutoff_deg)

    A = np.zeros((T, T), dtype=np.int64)
    W = np.zeros((T, T), dtype=np.float64)

    def add(i, j):
        d = np.linalg.norm(pos[i] - pos[j]) / C_KM_S
        A[i, j] = A[j, i] = 1
        W[i, j] = W[j, i] = d

    idx = np.arange(T).reshape(P, S)

    for p in range(P):
        for s in range(S):
            add(idx[p, s], idx[p, (s + 1) % S])

    for p in range(P):
        q = (p + 1) % P
        if not seam and q == 0 and p == P - 1:
            continue
        for s in range(S):
            i = idx[p, s]
            if polar[i]:
                continue
            cand = idx[q]
            mask = ~polar[cand]
            if not mask.any():
                continue
            dists = np.linalg.norm(pos[cand] - pos[i], axis=1)
            dists[~mask] = np.inf
            j = cand[int(np.argmin(dists))]
            if A[i, j] == 0:
                add(i, j)
    return A, W


def graph_sequence(walker, horizon_s, slot_s, **kw):
    """Yield (t, A, W) snapshots over [0, horizon_s) at slot_s spacing."""
    n_slots = int(horizon_s // slot_s)
    for k in range(n_slots):
        t = k * slot_s
        A, W = grid_isl_graph(walker, t, **kw)
        yield t, A, W


# ---- preset shells -------------------------------------------------------
def iridium_like():
    # 66 sats, 6 planes x 11, ~86.4 deg, 780 km  (fast, for smoke)
    return Walker(66, 6, 1, 86.4, 780.0)


def starlink_mini():
    # 264 sats, 24 x 11, 53 deg, 550 km  (main case-study scale)
    return Walker(264, 24, 1, 53.0, 550.0)


def starlink_shell1():
    # Starlink shell 1: 1584 sats, 72 planes x 22, 53 deg, 550 km (scale stress test)
    return Walker(1584, 72, 1, 53.0, 550.0)
