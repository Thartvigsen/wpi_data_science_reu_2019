#!/bin/sh
#SBATCH -N 1
#SBATCH -n 3
#SBATCH -o dataCleaner.out
#SBATCH -t 05:00:00
#SBATCH --mem=55GB 

python -u dataCleaner.py
