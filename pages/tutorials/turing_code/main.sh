#!/bin/bash
################################
# Script for running multiple experiments by submitting multiple slurm jobs
# How to use this script?
#  ./main.sh [taskid range]
# For example, if you want to run task id 1 to id 5 (5 tasks in parallel), you could type:
# ./main.sh 1-5
################################
# sbatch --array=$1 main.sbatch

#srun --array=$1 main.sbatch
# source ../env/bin/activate 
export PYTHONPATH=./env/bin/python
source ./env/bin/activate
sbatch --array=$1 main.sbatch
#srun --pty -t 24:00 --gres=gpu:0 --mem=64G python main.py --taskid=${1}

################################
# How to use more GPUs?
# change in file: main.sbatch
################################

