#!/bin/bash

#SBATCH --job-name=LP_512
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err
#SBATCH --mem=25GB
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

hostname
pwd

source /tcmldrive/lib/miniconda3/etc/profile.d/conda.sh
conda activate MSI

nohup python /home/sharonpe/microsatellite-instability-classification/main.py --train-subtype-classification-tile

echo "END"
