#!/bin/bash
#SBATCH --job-name=sallj
#SBATCH --output=/home/%u/SALLJ/logs/detect_%j.out
#SBATCH --error=/home/%u/SALLJ/logs/detect_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --partition=hpc*

# load python interpreter

source ~/miniconda3/etc/profile.d/conda.sh
conda activate llj

# scripts live in $HOME/SALLJ/scripts/MJO/
# pathlib in the Python scripts handles all other paths from there

SCRIPT_DIR="$HOME/SALLJ/scripts/MJO"

echo ""
echo "Job:         $SLURM_JOB_ID"
echo "Node:        $SLURMD_NODENAME"
echo "CPUs:        $SLURM_CPUS_PER_TASK"
echo "Script dir:  $SCRIPT_DIR"
echo "Started:     $(date)"

# Job 1: SALLJ Detection

echo ""
echo "[1/3] Running SALLJ detection"
python "$SCRIPT_DIR/SALLJ.py"

DETECT_EXIT=$?
if [ $DETECT_EXIT -ne 0 ]; then
    echo "ERROR: Detection script failed with exit code $DETECT_EXIT"
    exit $DETECT_EXIT
fi

echo ""
echo "Detection complete at $(date)"

# Job 2: Model MJO Calculation

echo ""
echo "[2/3] Running Model MJO Calculation"
python "$SCRIPT_DIR/calc_model_mjo.py"

MJO_CALC_EXIT=$?
if [ $MJO_CALC_EXIT -ne 0 ]; then
    echo "ERROR: Model MJO Calculation script failed with exit code $MJO_CALC_EXIT"
    exit $MJO_CALC_EXIT
fi

echo ""
echo "Model MJO Calculation complete at $(date)"


# Job 3: MJO Compositing

echo ""
echo "[3/3] Running MJO compositing"
python "$SCRIPT_DIR/MJO.py"

COMP_EXIT=$?
if [ $COMP_EXIT -ne 0 ]; then
    echo "ERROR: Compositing script failed with exit code $COMP_EXIT"
    exit $COMP_EXIT
fi

echo ""
echo "Compositing complete at $(date)"

# print summary

echo ""
echo "Pipeline complete at $(date)"
echo ""
echo "Final outputs in $HOME/SALLJ/figures/MJO/:"
ls -lh "$HOME/SALLJ/figures/MJO/"