#!/usr/bin/env bash
# Full-scale run on the GPU/CPU server under conda + tmux.
#   conda env create -f environment.yml   # first time
#   conda activate qignn
#   tmux new -s qignn
#   bash scripts/server_run.sh
#
# All experiments are CPU-bound (small graphs, eig-based operators); a GPU is not
# required. Results land in results/*.csv and a tee'd log.

set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p results
LOG="results/run_$(date +%Y%m%d_%H%M%S).log"

echo "== Exp A: propagation mechanism ==" | tee -a "$LOG"
python experiments/exp_a_propagation.py 2>&1 | tee -a "$LOG"

echo "== Exp B: operator comparison (multi-seed) ==" | tee -a "$LOG"
# main case-study scale: starlink_mini (264 sats), 5 seeds, full depth/epochs
python -c "import sys; sys.path[:0]=['experiments','.']; import exp_b_operators as B; \
B.run(shell='mini', seeds=(0,1,2,3,4), n_train=12, n_eval=8, hidden=32, n_layers=4, epochs=400)" \
  2>&1 | tee -a "$LOG"

echo "== Exp C1: normalization pitfall ==" | tee -a "$LOG"
python -c "import sys; sys.path[:0]=['experiments','.']; import exp_c_pitfalls as C; \
C.run_c1(sizes=((4,4),(5,5),(6,6),(7,7),(8,8),(9,9),(10,10)), epochs=300)" \
  2>&1 | tee -a "$LOG"

echo "== Exp C2: single-seed vs multi-seed pitfall ==" | tee -a "$LOG"
python -c "import sys; sys.path[:0]=['experiments','.']; import exp_c_pitfalls as C; \
C.run_c2(seeds=range(8), epochs=400)" \
  2>&1 | tee -a "$LOG"

echo "DONE -> $LOG"
