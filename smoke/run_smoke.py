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
import exp_c_pitfalls as C


def main():
    print("=" * 60)
    print("SMOKE: Exp A (propagation mechanism)")
    sh, sq = A.run(n=41, t_max=6.0, n_t=20)
    assert sq > sh, "quantum walk should spread faster than heat"

    print("=" * 60)
    print("SMOKE: Exp B (operator comparison, tiny)")
    B.run(shell="iridium", seeds=(0, 1), n_train=4, n_eval=3,
          hidden=8, n_layers=2, epochs=40)

    print("=" * 60)
    print("SMOKE: Exp C1 (normalization pitfall, tiny)")
    C.run_c1(sizes=((3, 3), (4, 4), (5, 5)), epochs=40)

    print("=" * 60)
    print("SMOKE: Exp C2 (seed pitfall, tiny)")
    C.run_c2(seeds=range(3), epochs=40)

    print("=" * 60)
    print("SMOKE OK")


if __name__ == "__main__":
    main()
