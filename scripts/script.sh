#!/bin/bash

#SBATCH --job-name=wind                   # job name
#SBATCH --partition=hpc                   # partition or queue name
#SBATCH --nodes=1                         # num of nodes
#SBATCH --ntasks-per-node=1               # num of tasks per node
#SBATCH --cpus-per-task=50                # num of CPU cores per task
#SBATCH --mem-per-cpu=1G                  # memory per CPU requested
#SBATCH --time=00:30:00                   # max runtime
#SBATCH --output=job_output_%j.out        # job output
#SBATCH --error=job_output_%j.err         # job error output
#SBATCH --mail-type=START,END,FAIL        # send email at start and end of job
#SBATCH --mail-user=fld1@williams.edu     # email address for notifications

# load python interpreter
source ~/miniconda3/etc/profile.d/conda.sh
conda activate llj

# run scripts
python month_cross_sections.py
