#!/bin/bash

#SBATCH --job-name=preprocess
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err
#SBATCH --mem=20GB


echo "Current date and time: $(date)"
hostname
pwd

source /home/sharonpe/miniconda3/etc/profile.d/conda.sh
conda activate MSI

srun python /home/sharonpe/microsatellite-instability-classification/main.py --preprocess --num-preprocess 10

echo "Current date and time: $(date)"
echo "END"
