#!/bin/bash
#SBATCH -p short
#SBATCH -N 1                      # number of nodes
#SBATCH -n 3                      # number of cores
#SBATCH --mem=55GB               # memory pool for all cores
#SBATCH -t 0-24:00                # time (D-HH:MM)
#SBATCH --checkpoint=5
#SBATCH --checkpoint-dir=checkpoints
#SBATCH --gres=gpu:0              # number of GPU
#SBATCH --job-name=main
#SBATCH -o slurm-main-output%a    # STDOUT
#SBATCH -e slurm-main-error%a     # STDERR
##SBATCH --mail-type=END
##SBATCH --mail-user=twhartvigsen@wpi.edu

################################
# Run experiments in main.py (by submitting a slurm job) 
# How to use this script?
# in Cluster Head Node terminal, type: sbatch --array=1-5 main.sbatch 
# here the ids are the task ids to run 
################################

##echo "SLURMD_NODENAME"=$SLURMD_NODENAME
##export PYTHONPATH=./env/bin/python
##source ./env/bin/activate

python low_d_mnist.py --taskid=${SLURM_ARRAY_TASK_ID} #--num_gpu=1
#python main.py --taskid=${SLURM_ARRAY_TASK_ID} #--num_gpu=1




##python get_variables.py
##python edsc.py
##python splitting_data.py
##python imagine3.py

################################
# How to use more GPUs?
# change line 6 and line 19
################################
