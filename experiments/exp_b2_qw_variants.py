"""
Exp B2 -- quantum-walk variant probe (the honest "did we try to rescue it" study).

The headline comparison (Exp B) found the quantum-inspired walk on the Laplacian
roughly ties the classical heat operator. Before concluding, we probe whether a
different quantum-walk design recovers a clear advantage:

  qw_lap   : |exp(-i t L)|^2        (baseline, walk on the Laplacian)
  qw_adj   : |exp(-i t A)|^2        (walk on the adjacency)
  qw_norm  : |exp(-i t L_sym)|^2    (walk on the normalized Laplacian)
  qw_multi : multi-time readout, concat of |exp(-i t_m L)|^2 over m=1..M
  heat     : exp(-t L)              (classical reference)

All but qw_multi are param-matched; qw_multi has more parameters (reported), so it
is a capacity-favoured upper bound for the quantum-walk family. Multi-seed.

Output: results/exp_b2_qw_variants.csv
"""

import _boot  # noqa: F401
import csv
import os
import numpy as np

from src.constellation import starlink_mini, iridium_like
from src.data import make_leo_samples
from src.train import train_eval


def run(shell="mini", seeds=(0, 1, 2, 3, 4), n_train=12, n_eval=8,
        hidden=32, n_layers=4, epochs=400, n_times=3, out=None):
    walker = starlink_mini() if shell == "mini" else iridium_like()
    # (label, op_kind, eig_key, n_times)
    variants = [
        ("heat", "heat", "L", 1),
        ("qw_lap", "qw", "L", 1),
        ("qw_adj", "qw", "A", 1),
        ("qw_norm", "qw", "S", 1),
        ("qw_multi", "qw", "L", n_times),
    ]
    agg = {v[0]: {"mae": [], "aurc": [], "params": None} for v in variants}

    for seed in seeds:
        tr = make_leo_samples(walker, n_train, norm="eccentricity", seed=1000 + seed)
        ev = make_leo_samples(walker, n_eval, norm="eccentricity", seed=2000 + seed)
        for (label, op, eig, nt) in variants:
            r = train_eval(op, tr, ev, hidden=hidden, n_layers=n_layers,
                           epochs=epochs, seed=seed, eig_key=eig, n_times=nt)
            agg[label]["mae"].append(r["mae"])
            agg[label]["aurc"].append(r["aurc"])
            agg[label]["params"] = r["params"]
            print(f"  seed{seed} {label:9s} mae={r['mae']:.4f} "
                  f"aurc={r['aurc']:.4f} params={r['params']}")

    out = out or os.path.join(_boot.RESULTS, "exp_b2_qw_variants.csv")
    with open(out, "w", newline="") as f:
        wri = csv.writer(f)
        wri.writerow(["variant", "mae_mean", "mae_std", "aurc_mean", "aurc_std", "params"])
        for (label, *_rest) in variants:
            mae = np.array(agg[label]["mae"])
            aurc = np.array(agg[label]["aurc"])
            wri.writerow([label, mae.mean(), mae.std(), aurc.mean(), aurc.std(),
                          agg[label]["params"]])
    print(f"[Exp B2] shell={shell} seeds={list(seeds)} wrote {out}")
    for (label, *_rest) in variants:
        mae = np.array(agg[label]["mae"])
        aurc = np.array(agg[label]["aurc"])
        print(f"  {label:9s} mae={mae.mean():.4f}+/-{mae.std():.4f} "
              f"aurc={aurc.mean():.4f}+/-{aurc.std():.4f} "
              f"params={agg[label]['params']}")
    return agg


if __name__ == "__main__":
    run()
