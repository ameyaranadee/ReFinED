#!/bin/bash
#SBATCH --partition=gpu            # Partition to submit to
#SBATCH --time=3:00:00             # Maximum job duration
#SBATCH --cpus-per-task=2          # Number of CPU cores
#SBATCH --mem=40G                  # Memory in GB
#SBATCH --gpus=1                   # Number of GPUs
#SBATCH --constraint=vram40        # Extra Slurm constraint
#SBATCH --output=slurm_logs/slurm-%j.out      # Standard output and error log
#SBATCH --mail-type=BEGIN,END,FAIL # Notifications for job start, end, and failure
#SBATCH --mail-user=aranade@umass.edu

module load conda/latest
conda activate refined38

export PYTHONPATH="${PWD}/../src:${PYTHONPATH}"

echo "Running job on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

python3 refined_wns_training.py