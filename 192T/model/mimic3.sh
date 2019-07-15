#!/bin/sh
#SBATCH -N 1
#SBATCH -n 3
#SBATCH -o mimic3.out
#SBATCH -t 08:00:00

python -u mimic3.py
