#!/bin/sh
#SBATCH -N 1
#SBATCH -n 3
#SBATCH -o peformImpute.out
#SBATCH -t 05:00:00

python -u performImpute.py
