#!/bin/bash
#SBATCH --job-name=sallj
#SBATCH --output=/home/%u/SALLJ/logs/detect_%j.out
#SBATCH --error=/home/%u/SALLJ/logs/detect_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --partition=hpc

# load python interpreter

source ~/miniconda3/etc/profile.d/conda.sh
conda activate llj

# scripts live in $HOME/sallj_project/scripts/
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
echo "[1/2] Running SALLJ detection"
python "$SCRIPT_DIR/SALLJ.py"

DETECT_EXIT=$?
if [ $DETECT_EXIT -ne 0 ]; then
    echo "ERROR: Detection script failed with exit code $DETECT_EXIT"
    exit $DETECT_EXIT
fi

echo ""
echo "Detection complete at $(date)"

# Job 2: MJO Compositing

echo ""
echo "[2/2] Running MJO compositing"
python "$SCRIPT_DIR/MJO.py"

COMP_EXIT=$?
if [ $COMP_EXIT -ne 0 ]; then
    echo "ERROR: Compositing script failed with exit code $COMP_EXIT"
    exit $COMP_EXIT
fi

echo ""
echo "Compositing complete at $(date)"

# Cleanup: delete intermediate CSVs written to local_data/MJO/
# Raw CMIP data in /scratch/fld1/cmip_cache is NOT deleted — keep it.
# Final outputs (PNGs + lag table CSVs in figures/MJO/) are NOT deleted.

DATA_DIR="$HOME/SALLJ/local_data/MJO"

echo ""
echo "Cleaning up intermediate files..."
rm -f "$DATA_DIR/sallj_index_full.csv" \
      "$DATA_DIR/sallj_monthly_clim.csv" \
      "$DATA_DIR/sallj_monthly_thresh.csv"
echo "Deleted intermediate CSVs from $DATA_DIR"


# print summary

echo ""
echo "Pipeline complete at $(date)"
echo ""
echo "Final outputs in $HOME/SALLJ/figures/MJO/:"
ls -lh "$HOME/SALLJ/figures/MJO/"
