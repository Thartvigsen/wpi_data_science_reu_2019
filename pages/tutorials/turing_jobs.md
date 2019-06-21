## Running jobs on WPI's turing cluster.

After this tutorial, you will know how to submit job arrays to SLURM, the workload manager used on Turing, by calling one script from the command line.

### Simplest and fastest approach:
In this approach, you will write a python file that runs your code, perhaps named *main.py*. You will create a *main.sbatch* file, telling SLURM the hardware requirements for the job, and a *main.sh* file, which you run to submit the job.

#### main.py
This file contains the actual code you want to run. For example:

```python
for i in range(10):
    print(i)
```

#### main.sbatch
This file contains the hardware specifics and the file to run (main.py). In general, the fewer resources you request, the faster a job will start running. The --taskid is not particulaly important but helps to name the output files so that you know to which jobs they correspond. Through experience this will become clearer. The first part of the file defines SLURM parameters, which can be found in the [SLURM DOCUMENTATION](https://slurm.schedmd.com/sbatch.html). As it is written now, I am using no GPUs -- if you want to use GPUs, you need to change your pytorch code to load your model and data onto the GPU prior to training.

```bash
#!/bin/bash
#SBATCH -p short
#SBATCH -N 1                      # number of nodes
#SBATCH -n 3                      # number of cores
#SBATCH --mem=55GB                # memory pool for all cores
#SBATCH -t 0-24:00                # time (D-HH:MM)
#SBATCH --gres=gpu:0              # number of GPU
#SBATCH --job-name=main
#SBATCH -o slurm-main-output%a    # STDOUT
#SBATCH -e slurm-main-error%a     # STDERR

python main.py --taskid=${SLURM_ARRAY_TASK_ID} # CALL YOUR PYTHON FILE
```
