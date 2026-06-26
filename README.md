# Quantum-Inspired Graph Neural Networks for Dynamic Wireless Topologies — tutorial code

Reproducible companion code for the IEEE TNSE special-issue tutorial
*"Quantum and Quantum-Inspired Graph Neural Networks for Dynamic Wireless Network
Topologies."* Every figure in the paper is generated from the CSVs produced here.

The thesis in one line: **GCN, graph diffusion, and quantum walks are the same
object — a propagation operator `Phi(L)` on the graph Laplacian.** Pick the operator,
pick the spreading physics; the rest of the network is shared, so the comparison is
param-matched and honest.

## Layout
```
src/
  operators.py     # GCN / heat exp(-tL) / quantum-walk exp(-itL) propagators
  models.py        # one GNN, three pluggable operators (param-matched)
  data.py          # dynamic LEO graph -> node-potential / reachability task
  constellation.py # Walker-delta LEO ISL graph sequence (pure numpy)
  graphs.py        # synthetic graph families (path/grid) + BFS/diameter utils
  metrics.py       # MAE, far-node AURC, mean+/-std
  train.py         # shared seeded train/eval loop
experiments/
  exp_a_propagation.py  # ballistic (quantum walk) vs diffusive (heat) mechanism
  exp_b_operators.py    # operator comparison on LEO, multi-seed, mean+/-std
  exp_c_pitfalls.py     # evaluation pitfalls: normalization + single-seed
smoke/run_smoke.py      # fast end-to-end pipeline check (not paper numbers)
scripts/server_run.sh   # full-scale run under conda + tmux
results/                # CSV outputs (figures are built from these)
```

## Setup
```bash
conda env create -f environment.yml
conda activate qignn
```
(Only numpy / scipy / pytorch-cpu are required; matplotlib for previews,
pennylane only for the optional quantum-hardware demo.)

## Reproduce
```bash
python smoke/run_smoke.py          # ~1 min, verifies the pipeline
bash scripts/server_run.sh         # full multi-seed run -> results/*.csv
```

## What each experiment teaches
- **Exp A** — quantum-walk spreading is ballistic (`Var ~ t^2`) vs heat diffusion
  (`Var ~ t`). Mechanism, theory-backed.
- **Exp B** — on the dynamic LEO graph, global operators (heat, quantum-inspired)
  beat local GCN; reported honestly with mean +/- std over seeds.
- **Exp C** — how false advantages arise and die: scale-coupled target
  normalization manufactures a spurious diameter trend (use scale-free per-source
  normalization); single-seed results are untrustworthy (report mean +/- std).

## Honesty note
This code follows a strict empirical-verification protocol: param-matched
operators, multi-seed (never single-seed), scale-free metrics, and a dedicated
pitfall study. Where the quantum-inspired operator does **not** beat a classical
baseline, that is reported as-is — the tutorial's value includes teaching readers
how to avoid fooling themselves into claiming a quantum advantage.

The Walker LEO topology generator is adapted from the author's `qwgnn-leo-routing`
codebase; the node-potential task here is new and independent of that paper's
traffic-engineering pipeline.
