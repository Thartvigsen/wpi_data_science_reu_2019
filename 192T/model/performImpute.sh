#!/bin/sh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=3
#SBATCH -o performImpute.out
#SBATCH -t 05:00:00

python -u performImpute.py
