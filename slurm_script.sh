#!/bin/bash

#SBATCH --job-name=L_D_FV_512
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err
#SBATCH --nodes=1            # This needs to match Trainer(num_nodes=...)
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1   # This needs to match Trainer(devices=...)
#SBATCH --mem=25GB

hostname
pwd

source /home/sharonpe/miniconda3/etc/profile.d/conda.sh
conda activate MSI

srun python /home/sharonpe/microsatellite-instability-classification/main.py --train-subtype-classification-tile

echo "END"
