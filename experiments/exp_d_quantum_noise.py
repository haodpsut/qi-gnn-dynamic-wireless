"""
Exp D -- from quantum-inspired to true quantum: how device noise erodes the
ballistic mechanism.

We run a continuous-time quantum walk on a small graph as an actual quantum circuit
(PennyLane, mixed-state simulator), then inject per-qubit depolarizing noise of
strength p and watch the coherent ballistic structure of Exp A collapse toward a
featureless, decohered distribution. This is the concrete near-term caveat behind the
quantum-inspired-versus-true-quantum discussion: the mechanism is real in theory but
fragile on NISQ hardware, which is why the runnable path today is the classical
surrogate used in the case study.

Output: results/exp_d_quantum_noise.csv  (noise_p, spread_var, l1_to_uniform).
"""

import _boot  # noqa: F401
import csv
import os
import numpy as np
from scipy.linalg import expm

from src.graphs import path_graph


def run(n_nodes=8, t=2.0, noise_levels=None, out=None):
    try:
        import pennylane as qml
    except ImportError:
        print("[Exp D] PennyLane not installed; skipping. "
              "`pip install pennylane` (it is in environment.yml).")
        return None

    assert (n_nodes & (n_nodes - 1)) == 0, "n_nodes must be a power of 2"
    n_qubits = int(np.log2(n_nodes))
    A, pos = path_graph(n_nodes)
    H = A.astype(complex)                       # CTQW Hamiltonian = adjacency
    U = expm(-1j * t * H)                       # noiseless walk unitary
    start = n_nodes // 2
    start_bits = [int(b) for b in format(start, f"0{n_qubits}b")]

    if noise_levels is None:
        noise_levels = [0.0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5]

    dev = qml.device("default.mixed", wires=n_qubits)

    @qml.qnode(dev)
    def circuit(p):
        qml.BasisState(np.array(start_bits), wires=range(n_qubits))
        qml.QubitUnitary(U, wires=range(n_qubits))
        for w in range(n_qubits):
            qml.DepolarizingChannel(p, wires=w)
        return qml.probs(wires=range(n_qubits))

    uniform = np.ones(n_nodes) / n_nodes
    rows = []
    for p in noise_levels:
        probs = np.real(circuit(p))
        probs = probs / probs.sum()
        mean = np.sum(probs * pos)
        var = float(np.sum(probs * (pos - mean) ** 2))
        l1 = float(np.sum(np.abs(probs - uniform)))   # distance from decohered state
        rows.append((float(p), var, l1))
        print(f"  p={p:.2f}  spread_var={var:.3f}  L1_to_uniform={l1:.3f}")

    out = out or os.path.join(_boot.RESULTS, "exp_d_quantum_noise.csv")
    with open(out, "w", newline="") as f:
        wri = csv.writer(f)
        wri.writerow(["noise_p", "spread_var", "l1_to_uniform"])
        wri.writerows(rows)
    v0, vlast = rows[0][1], rows[-1][1]
    print(f"[Exp D] N={n_nodes} t={t}: spread variance {v0:.2f} (p=0) "
          f"-> {vlast:.2f} (p={noise_levels[-1]}); coherent structure decays "
          f"toward the uniform mixed state. wrote {out}")
    return rows


if __name__ == "__main__":
    run()
