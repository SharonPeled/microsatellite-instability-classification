from src.configs import Configs
import os
import subprocess


if __name__ == '__main__':
    with open("slurm_script.sh", "w") as file:
        file.write(f"""#!/bin/bash

#SBATCH --job-name={Configs.SC_RUN_NAME}
#SBATCH --output=out_%j_%x.out
#SBATCH --error=err_%j_%x.err
#SBATCH --mem=50GB
#SBATCH --gres=gpu:1
#SBATCH --nodes={Configs.SC_NUM_NODES}   # This needs to match Trainer(num_nodes=...)
#SBATCH --ntasks-per-node={len(Configs.SC_NUM_DEVICES)}   # This needs to match Trainer(devices=...)
#SBATCH --cpus-per-task=1

export CUDA_VISIBLE_DEVICES={Configs.SC_RUN_NAME}

echo "Current date and time: $(date)"
hostname
pwd

source /home/sharonpe/miniconda3/etc/profile.d/conda.sh
conda activate MSI

srun python /home/sharonpe/microsatellite-instability-classification/main.py --train-subtype-classification-tile

echo "Current date and time: $(date)"
echo "END"
""")
    proc = subprocess.Popen(['sbatch slurm_script.sh'], shell=True)
    proc.wait()
    print('Done.')
