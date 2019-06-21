## Running jobs on WPI's turing cluster.

After this tutorial, you will know how to submit job arrays to SLURM, the workload manager used on Turing, by calling one script from the command line.

### Simplest and fastest approach:
In this approach, you will write a python file that runs your code, perhaps named *main.py*. You will create a *main.sbatch* file, telling SLURM the hardware requirements for the job, and a *main.sh* file, which you run to submit the job.

Copy the following text into corresponding file names in your turing directories or upload the files in [this folder](/turing_code).
#### main.py
This file contains the actual code you want to run. For example:

```python
for i in range(10):
    print(i)
```

#### main.sbatch
This file contains the hardware specifics and the file to run (main.py). In general, the fewer resources you request, the faster a job will start running. The --taskid is not particulaly important but helps to name the output files so that you know to which jobs they correspond. Through experience this will become clearer. The first part of the file defines SLURM parameters, which can be found in the [SLURM documentation](https://slurm.schedmd.com/sbatch.html). As it is written now, I am using no GPUs -- if you want to use GPUs, you need to change your pytorch code to load your model and data onto the GPU prior to training.

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

#### main.sh
The number following the execution is the "job ID", which will be included in the names of the output and error logs.

The actual file's structure is very simple:
```bash
#!/bin/bash
################################
# Script for running multiple experiments by submitting multiple slurm jobs
# How to use this script?
#  ./main.sh [taskid range]
# For example, if you want to run task id 1 to id 5 (5 tasks in parallel), you could type:
# ./main.sh 1-5
################################
# sbatch --array=$1 main.sbatch

export PYTHONPATH=./env/bin/python
source ./env/bin/activate
sbatch --array=$1 main.sbatch
```
./env/bin/python is the path to the python version you are running in your virtual environment (in this example my virtual environment is called *env*. source ./env/bin/activate simply activates the virtual environment.

### Submitting a job
To run your file *main.py* on Turing, execute *main.sh* in the command line to run a job:
```bash
./main.sh 0
```

Remember, *main.sh* has to be executable, so when creating this file you need to modify the permissions of the file to add execution:
```bash
chmod +x main.sh
```
