#!/bin/bash

#SBATCH --job-name=wind
#SBATCH --partition=hpc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=256
#SBATCH --mem=250GB
#SBATCH --time=36:00:00
#SBATCH --output=job_output_%j.out
#SBATCH --error=job_output_%j.err
#SBATCH --mail-type=START,END,FAIL
#SBATCH --mail-user=fld1@williams.edu

# load python interpreter
source ~/miniconda3/etc/profile.d/conda.sh
conda activate llj

# run scripts
python SALLJ.py
