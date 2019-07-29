#!/bin/sh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=6
#SBATCH -o mimic3_med.out
#SBATCH -t 08:00:00

python -u mimic3_med.py $1 $2 $3 $4
