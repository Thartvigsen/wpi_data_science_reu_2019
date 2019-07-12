#!/bin/sh
#SBATCH -N 8
#SBATCH -n 16
#SBATCH -o mimic3.out
#SBATCH -t 05:00:00
#SBATCH --mem=55GB

python -u mimic3.py
