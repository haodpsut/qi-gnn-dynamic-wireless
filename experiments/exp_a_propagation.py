"""
Exp A -- propagation mechanism: ballistic (quantum walk) vs diffusive (heat).

On a path graph we release a unit impulse at the centre and measure how the
spread variance of the propagated distribution grows with propagation time t.
Theory: heat diffusion grows linearly (Var ~ t), the continuous-time quantum
walk grows quadratically (Var ~ t^2). This is the core intuition of the tutorial.

Output: results/exp_a_propagation.csv  with columns t, var_heat, var_qw.
"""

import _boot  # noqa: F401
import csv
import os
import numpy as np

from src.graphs import path_graph
from src.operators import laplacian, eig_sym, heat_propagator, qwalk_prob, spread_variance


def run(n=81, t_max=8.0, n_t=40, out=None):
    A, pos = path_graph(n)
    L = laplacian(A, normalized=False)
    w, U = eig_sym(L)
    src = n // 2
    ts = np.linspace(0.2, t_max, n_t)
    rows = []
    for t in ts:
        ph = heat_propagator(w, U, t)
        pq = qwalk_prob(w, U, t)
        v_heat = spread_variance(ph[src], pos)
        v_qw = spread_variance(pq[src], pos)
        rows.append((float(t), v_heat, v_qw))

    out = out or os.path.join(_boot.RESULTS, "exp_a_propagation.csv")
    with open(out, "w", newline="") as f:
        wri = csv.writer(f)
        wri.writerow(["t", "var_heat", "var_qw"])
        wri.writerows(rows)

    # quick slope check on a log-log fit: heat ~ t^1, qw ~ t^2
    lt = np.log(ts)
    sh = np.polyfit(lt, np.log([r[1] for r in rows]), 1)[0]
    sq = np.polyfit(lt, np.log([r[2] for r in rows]), 1)[0]
    print(f"[Exp A] N={n} wrote {out}")
    print(f"  log-log slope: heat ~ t^{sh:.2f} (expect ~1), "
          f"qw ~ t^{sq:.2f} (expect ~2)")
    return sh, sq


if __name__ == "__main__":
    run()
