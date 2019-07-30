#!/bin/sh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -o runTestMed.out
#SBATCH -t 08:00:00

python -u runTestMed.py $1 $2
