# Quantum and Quantum-Inspired Graph Neural Networks for Dynamic Wireless Topologies

Paper source and reproducible companion code for the manuscript

> **Quantum and Quantum-Inspired Graph Neural Networks for Dynamic Wireless Network
> Topologies: A Tutorial and Survey with a Reproducible LEO Case Study**
> Phuc Hao Do (Danang Architecture University)

**Submission target.** This manuscript is being submitted to **IEEE Transactions on
Network Science and Engineering (TNSE)**, Special Issue on *Tutorials and Surveys on AI
and Quantum-enabled Learning for Future Wireless Networks*. The LaTeX source and the
cover letter are in [`paper/`](paper/); every figure and number in the paper is
regenerated from seeded runs by the code here.

The thesis in one line: **GCN, graph diffusion, and quantum walks are the same object, a
propagation operator `Phi(L)` on the graph Laplacian.** Pick the operator, pick the
spreading physics; the rest of the network is shared, so the comparison is param-matched
and honest.

## Layout
```
paper/             # LaTeX source (main.tex, sections/, figs/, refs.bib) + cover_letter.md
src/
  operators.py     # GCN / heat exp(-tL) / quantum-walk exp(-itL) propagators
  models.py        # one GNN, three pluggable operators (param-matched)
  data.py          # dynamic LEO graph -> node-potential / reachability task
  constellation.py # Walker-delta LEO ISL graph sequence (pure numpy)
  graphs.py        # synthetic graph families (path/grid) + BFS/diameter utils
  metrics.py       # MAE, far-node AURC, mean+/-std
  train.py         # shared seeded train/eval loop
experiments/
  exp_a_propagation.py   # ballistic (quantum walk) vs diffusive (heat) mechanism
  exp_b_operators.py     # operator comparison on LEO, multi-seed, mean+/-std (+per-seed)
  exp_b2_qw_variants.py  # quantum-walk variant probe (adjacency / normalized / multi-time)
  exp_c_pitfalls.py      # evaluation pitfalls: normalization + single-seed
  exp_d_quantum_noise.py # NISQ depolarizing noise on a quantum walk (PennyLane)
smoke/run_smoke.py       # fast end-to-end pipeline check (writes results/smoke/, not paper numbers)
scripts/server_run.sh    # full-scale run (incl. Exp E, 1584-sat scale-up) under conda + tmux
results/                 # CSV outputs (figures are built from these)
```

## Setup
```bash
conda env create -f environment.yml
conda activate qignn
```
(numpy / scipy / pytorch-cpu are required; matplotlib for previews, pennylane for Exp D.)

## Reproduce
```bash
python smoke/run_smoke.py          # ~1 min, verifies the pipeline (outputs in results/smoke/)
bash scripts/server_run.sh         # full multi-seed run -> results/*.csv
```

## What each experiment shows
- **Exp A** — quantum-walk spreading is ballistic (`Var ~ t^2`) vs heat diffusion
  (`Var ~ t`). Mechanism, theory-backed and exact.
- **Exp B** — operator comparison on the dynamic LEO graph (264 sats): global operators
  beat local GCN ~2.3x; heat and the quantum-inspired walk tie within seed noise.
- **Exp B2** — quantum-walk variant probe: the adjacency walk is the best variant on the
  far-node metric at matched parameters.
- **Exp C** — evaluation pitfalls: scale-coupled normalization manufactures a spurious
  diameter trend (use scale-free per-source normalization); single-seed results are
  untrustworthy (report mean +/- std).
- **Exp D** — NISQ depolarizing noise decoheres the quantum walk toward the uniform mixed
  state, the near-term caveat behind preferring the classical quantum-inspired surrogate.
- **Exp E** — scale-up to a 1584-satellite Starlink shell: the quantum-inspired walk
  **overtakes** the classical kernel at large diameter, the headline scale-dependent
  finding (the ballistic edge grows with diameter).

## Building the paper
```bash
cd paper
pdflatex main; bibtex main; pdflatex main; pdflatex main   # -> main.pdf
```

## Honesty note
This work follows a strict empirical-verification protocol: param-matched operators,
multi-seed (never single-seed), scale-free metrics, a base-matrix sweep, and a dedicated
pitfall study. Advantages are reported only where the data supports them, and at the
scale and regime where they hold.

The Walker LEO topology generator is adapted from the author's `qwgnn-leo-routing`
codebase; the node-potential task here is new and independent of that paper's
traffic-engineering pipeline.
