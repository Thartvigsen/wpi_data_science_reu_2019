#!/bin/sh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=3
#SBATCH -o mimic3.out
#SBATCH -t 08:00:00

python -u mimic3.py $1 $2 $3 $4
