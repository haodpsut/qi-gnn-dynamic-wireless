"""
Exp C -- evaluation pitfalls (the tutorial's signature section).

C1 Normalization-scale artifact. Across a graph family of growing diameter we
   train GCN and Heat and report the apparent global-vs-local gap under two target
   normalizations: by graph DIAMETER (the pitfall) vs by source ECCENTRICITY (the
   fix). Expected: diameter-normalization manufactures a gap that inflates with
   diameter; eccentricity-normalization removes the spurious trend.

C2 Single-seed vs multi-seed. On one LEO config we show that a single cherry-picked
   seed can flip the operator ranking, while mean +/- std over seeds is stable.

Outputs: results/exp_c1_normalization.csv, results/exp_c2_seeds.csv
"""

import _boot  # noqa: F401
import csv
import os
import numpy as np

from src.graphs import grid_graph, diameter
from src.data import Sample
from src.train import train_eval
from src.constellation import iridium_like
from src.data import make_leo_samples


def _grid_samples(h, w, norm, n, seed):
    rng = np.random.default_rng(seed)
    A, _ = grid_graph(h, w)
    samples = []
    for _ in range(n):
        dest = int(rng.integers(0, A.shape[0]))
        samples.append(Sample(A, dest, norm=norm))
    return samples, A


def run_c1(sizes=((3, 3), (4, 4), (5, 5), (6, 6), (7, 7)), epochs=150,
           norms=("raw", "diameter", "eccentricity"), out=None):
    """Apparent global-vs-local gap (GCN MAE - Heat MAE) vs diameter, under each
    target normalization. A normalization that couples the metric to graph scale
    manufactures a spurious diameter trend; a scale-free one stays flat."""
    rows = []
    for (h, w) in sizes:
        _, A = _grid_samples(h, w, "raw", 1, seed=0)
        diam = diameter(A)
        rec = {"diam": diam}
        for norm in norms:
            tr, _ = _grid_samples(h, w, norm, 6, seed=10)
            ev, _ = _grid_samples(h, w, norm, 6, seed=20)
            r_gcn = train_eval("gcn", tr, ev, n_layers=3, epochs=epochs, seed=0)
            r_heat = train_eval("heat", tr, ev, n_layers=3, epochs=epochs, seed=0)
            rec[f"gap_{norm}"] = r_gcn["mae"] - r_heat["mae"]
        rows.append(rec)
        msg = " ".join(f"gap_{n}={rec['gap_' + n]:+.4f}" for n in norms)
        print(f"  grid {h}x{w} diam={rec['diam']:.0f} {msg}")

    out = out or os.path.join(_boot.RESULTS, "exp_c1_normalization.csv")
    with open(out, "w", newline="") as f:
        wri = csv.writer(f)
        wri.writerow(["diam"] + [f"gap_{n}" for n in norms])
        for r in rows:
            wri.writerow([r["diam"]] + [r[f"gap_{n}"] for n in norms])
    d = np.array([r["diam"] for r in rows])
    print(f"[Exp C1] slope gap-vs-diam (spurious if magnitude grows):")
    for n in norms:
        g = np.array([r[f"gap_{n}"] for r in rows])
        print(f"    {n:12s} slope={np.polyfit(d, g, 1)[0]:+.4f}  "
              f"range=[{g.min():+.4f},{g.max():+.4f}]")
    print(f"  wrote {out}")
    return rows


def run_c2(seeds=range(6), epochs=200, out=None):
    walker = iridium_like()
    ops = ["gcn", "heat", "qw"]
    per = {op: [] for op in ops}
    for seed in seeds:
        tr = make_leo_samples(walker, 8, norm="eccentricity", seed=1000 + seed)
        ev = make_leo_samples(walker, 6, norm="eccentricity", seed=2000 + seed)
        for op in ops:
            r = train_eval(op, tr, ev, n_layers=3, epochs=epochs, seed=seed)
            per[op].append(r["mae"])

    out = out or os.path.join(_boot.RESULTS, "exp_c2_seeds.csv")
    with open(out, "w", newline="") as f:
        wri = csv.writer(f)
        wri.writerow(["seed"] + ops)
        for i, seed in enumerate(seeds):
            wri.writerow([seed] + [per[op][i] for op in ops])

    # demonstrate the flip: best-single-seed winner vs mean winner
    best_single = min(ops, key=lambda o: min(per[o]))
    best_mean = min(ops, key=lambda o: np.mean(per[o]))
    print(f"[Exp C2] best by cherry-picked seed = {best_single}; "
          f"best by mean over seeds = {best_mean}. wrote {out}")
    for op in ops:
        v = np.array(per[op])
        print(f"  {op:4s} mae mean={v.mean():.4f} std={v.std():.4f} "
              f"best={v.min():.4f} worst={v.max():.4f}")
    return per


if __name__ == "__main__":
    run_c1()
    run_c2()
