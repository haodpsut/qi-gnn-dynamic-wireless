"""
Exp B -- operator comparison on the dynamic LEO graph (the honest meat).

Three param-matched operators (GCN local, Heat global, QI quantum-inspired global)
on the node-potential task, over multiple seeds with mean +/- std. Expected,
honest outcome: the global operators (Heat, QI) beat local GCN on the long-range
field; QI is roughly on par with Heat, NOT a dramatic quantum win.

Output: results/exp_b_operators.csv
"""

import _boot  # noqa: F401
import csv
import os
import numpy as np

from src.constellation import iridium_like, starlink_mini
from src.data import make_leo_samples
from src.train import train_eval


def run(shell="iridium", seeds=(0, 1, 2), n_train=8, n_eval=6,
        hidden=16, n_layers=3, epochs=200, out=None):
    walker = iridium_like() if shell == "iridium" else starlink_mini()
    ops = ["gcn", "heat", "qw"]
    agg = {op: {"mae": [], "aurc": [], "params": None} for op in ops}
    perseed = []   # (seed, op, mae, aurc) rows, backs the per-seed table

    for seed in seeds:
        train = make_leo_samples(walker, n_train, norm="eccentricity",
                                 seed=1000 + seed)
        ev = make_leo_samples(walker, n_eval, norm="eccentricity",
                              seed=2000 + seed)
        for op in ops:
            r = train_eval(op, train, ev, hidden=hidden, n_layers=n_layers,
                           epochs=epochs, seed=seed)
            agg[op]["mae"].append(r["mae"])
            agg[op]["aurc"].append(r["aurc"])
            agg[op]["params"] = r["params"]
            perseed.append((seed, op, r["mae"], r["aurc"]))
            print(f"  seed{seed} {op:4s} mae={r['mae']:.4f} "
                  f"aurc={r['aurc']:.4f} params={r['params']}")

    out = out or os.path.join(_boot.RESULTS, "exp_b_operators.csv")
    with open(out, "w", newline="") as f:
        wri = csv.writer(f)
        wri.writerow(["op", "mae_mean", "mae_std", "aurc_mean", "aurc_std", "params"])
        for op in ops:
            mae = np.array(agg[op]["mae"])
            aurc = np.array(agg[op]["aurc"])
            wri.writerow([op, mae.mean(), mae.std(), aurc.mean(), aurc.std(),
                          agg[op]["params"]])
    # per-seed dump (single source of truth for the per-seed table)
    perseed_out = out.replace("exp_b_operators.csv", "exp_b_perseed.csv")
    with open(perseed_out, "w", newline="") as f:
        wri = csv.writer(f)
        wri.writerow(["seed", "op", "mae", "aurc"])
        wri.writerows(perseed)
    print(f"[Exp B] shell={shell} seeds={list(seeds)} wrote {out}")
    for op in ops:
        mae = np.array(agg[op]["mae"])
        print(f"  {op:4s} mae={mae.mean():.4f}+/-{mae.std():.4f} "
              f"params={agg[op]['params']}")
    return agg


if __name__ == "__main__":
    run()
