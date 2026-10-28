#!/bin/bash
#SBATCH -J aimo-probe-fit
#SBATCH -p ice-cpu
#SBATCH -N 1 --ntasks-per-node=1 -c 16
#SBATCH --mem=32G
#SBATCH -t 01:00:00
#SBATCH -o logs/probe_fit_%j.log
export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16
cd ${AIMO_BASE}/distill
${AIMO_BASE}/runtime_venv/bin/python fit_probe.py ${AIMO_BASE}/distill/results ${AIMO_BASE}/data_trainv2/train-main-v2.parquet 2>&1 | grep -v Warning
