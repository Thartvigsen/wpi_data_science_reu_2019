#!/bin/sh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -o runTest.out
#SBATCH -t 08:00:00

python -u runTest.py $1
