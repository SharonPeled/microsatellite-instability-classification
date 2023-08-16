from src.configs import Configs
import os
import subprocess


if __name__ == '__main__':
    Configs.set_task_configs(['DN', 'SC'])
    with open("slurm_script.sh", "w") as file:
        file.write(f"""#!/bin/bash

#SBATCH --job-name={Configs.joined['RUN_NAME']}
#SBATCH --output=out_%j_%x.out
#SBATCH --error=err_%j_%x.err
#SBATCH --mem=25GB
#SBATCH --gres=gpu:{len(Configs.joined['NUM_DEVICES'])}
#SBATCH --nodes={Configs.joined['NUM_NODES']}   # This needs to match Trainer(num_nodes=...)
#SBATCH --ntasks-per-node={len(Configs.joined['NUM_DEVICES'])}   # This needs to match Trainer(devices=...)
#SBATCH --cpus-per-task=1

echo "Current date and time: $(date)"
hostname
pwd

source /home/sharonpe/miniconda3/etc/profile.d/conda.sh
conda activate MSI

srun -p debug python /home/sharonpe/microsatellite-instability-classification/main.py --pretrain-subtype-classification-tile

echo "Current date and time: $(date)"
echo "END"
""")
    proc = subprocess.Popen(['sbatch slurm_script.sh'], shell=True)
    proc.wait()
    print('Done.')
