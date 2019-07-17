#!/bin/sh
#SBATCH -N 1
#SBATCH -n 3
#SBATCH -o mimic3_s.out
#SBATCH -t 08:00:00

python -u mimic3_s.py
