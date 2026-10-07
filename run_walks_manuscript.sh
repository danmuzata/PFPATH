#!/usr/bin/env bash
# Reproduce manuscript adaptive walks + population simulation.
# Uses PfPATH_current.py with BLOSUM, expanded oracle, and drug cycling.
set -euo pipefail

PYTHON=/home/danny/conda3/envs/bioatlas/bin/python
SCRIPT=PfPATH_current.py
OUTDIR=runs/pfpath_walks_manuscript_repro
LOGFILE=logs/walks_manuscript_repro.log

mkdir -p logs

nohup "$PYTHON" "$SCRIPT" \
  --wt         inputs/wt.fasta \
  --dms        inputs/epistasis_single_mutant_matrix.csv \
  --thermo     inputs/spired_thermo.csv \
  --mi         inputs/coevolution.csv \
  --dccm       inputs/dhfr_dccm_matrix.csv \
  --blosum     inputs/blosum_substitution_scores.csv \
  --oracle     inputs/dhfr_wetlab.csv \
  --oracle_mode combo \
  --drug_mode  cycle \
  --key_positions 7,41,42,50,99,155 \
  --beta       2.0 \
  --pop        10000 \
  --gens       20000 \
  --walk_reps  19 \
  --walk_steps 50 \
  --plot \
  --seed       1 \
  --out "$OUTDIR" \
> "$LOGFILE" 2>&1 &

echo "PID $! — log: $LOGFILE"
