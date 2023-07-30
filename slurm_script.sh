#!/bin/bash

#SBATCH --job-name=LP_test
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err
#SBATCH --mem=0GB
#SBATCH --gres=gpu:1
#SBATCH --nodes=1            # This needs to match Trainer(num_nodes=...)
#SBATCH --ntasks-per-node=1   # This needs to match Trainer(devices=...)
#SBATCH --cpus-per-task=1

echo "Current date and time: $(date)"
hostname
pwd

source /home/sharonpe/miniconda3/etc/profile.d/conda.sh
conda activate MSI

nohup srun python /home/sharonpe/microsatellite-instability-classification/main.py --train-subtype-classification-tile

echo "Current date and time: $(date)"
echo "END"
