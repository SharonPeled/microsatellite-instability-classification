#!/bin/bash
#SBATCH --job-name=pp10_10_3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=90
#SBATCH --mem-per-cpu=2G
#SBATCH --output=job_pp10_10_3_output.log
#SBATCH --error=job_pp10_10_3_error.log

echo "Current date and time: $(date)"
hostname
pwd

source /home/sharonpe/miniconda3/etc/profile.d/conda.sh
conda activate MSI

srun -p normal python -u /home/sharonpe/microsatellite-instability-classification/preprocess_parallel_batches.py --num_allowed_full_processes 10 --num_slides_per_process 10 --num_subprocesses_per_process 3


echo "Current date and time: $(date)"
echo "END"

