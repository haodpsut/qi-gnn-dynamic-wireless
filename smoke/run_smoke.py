"""
Fast smoke test: runs all experiments at tiny scale to verify the pipeline end to
end (operators, dataset, training, metrics, CSV output). Not for paper numbers.

Run:  python smoke/run_smoke.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "experiments"))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import exp_a_propagation as A
import exp_b_operators as B
import exp_b2_qw_variants as B2
import exp_c_pitfalls as C

# Smoke outputs go to results/smoke/ so they NEVER overwrite the canonical
# results/*.csv that the paper and figures are built from.
SMOKE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "results", "smoke")
os.makedirs(SMOKE_DIR, exist_ok=True)
def _o(name):
    return os.path.join(SMOKE_DIR, name)


def main():
    print("=" * 60)
    print("SMOKE: Exp A (propagation mechanism)")
    sh, sq = A.run(n=41, t_max=6.0, n_t=20, out=_o("exp_a_propagation.csv"))
    assert sq > sh, "quantum walk should spread faster than heat"

    print("=" * 60)
    print("SMOKE: Exp B (operator comparison, tiny)")
    B.run(shell="iridium", seeds=(0, 1), n_train=4, n_eval=3,
          hidden=8, n_layers=2, epochs=40, out=_o("exp_b_operators.csv"))

    print("=" * 60)
    print("SMOKE: Exp B2 (quantum-walk variant probe, tiny)")
    B2.run(shell="iridium", seeds=(0, 1), n_train=4, n_eval=3,
           hidden=8, n_layers=2, epochs=40, n_times=3, out=_o("exp_b2_qw_variants.csv"))

    print("=" * 60)
    print("SMOKE: Exp C1 (normalization pitfall, tiny)")
    C.run_c1(sizes=((3, 3), (4, 4), (5, 5)), epochs=40, out=_o("exp_c1_normalization.csv"))

    print("=" * 60)
    print("SMOKE: Exp C2 (seed pitfall, tiny)")
    C.run_c2(seeds=range(3), epochs=40, out=_o("exp_c2_seeds.csv"))

    print("=" * 60)
    print("SMOKE OK (outputs in results/smoke/, canonical CSVs untouched)")


if __name__ == "__main__":
    main()
