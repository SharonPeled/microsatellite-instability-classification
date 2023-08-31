import subprocess
import os
from src.configs import Configs
from src.general_utils import get_time
import shutil
import pandas as pd
import numpy as np
from time import sleep
from dataclasses import dataclass


def generate_slide_paths_from_manifest(manifest_path, slides_dir):
    slide_paths = []
    df_m = pd.read_csv(manifest_path, sep='\t')
    for i, row in df_m.iterrows():
        slide_uuid = row['id']
        filename = row['filename']
        path = os.path.join(slides_dir, slide_uuid, filename)
        slide_paths.append(path)
    df_m['slide_path'] = slide_paths
    return slide_paths, df_m


def get_bash_str_preprocess(slide_ids, num_processes, full_batch_ind):
    slides_str = ' '.join(slide_ids)
    bash_str = f"""
    source /home/sharonpe/miniconda3/etc/profile.d/conda.sh
    conda activate MSI
    python preprocess_full.py --num-processes {num_processes} --slide_ids {slides_str} --slide_dir {Configs.SLIDES_DIR} > {full_batch_ind}_preprocess_full_{get_time()}.txt
    """
    return bash_str


if __name__ == '__main__':
    num_allowed_full_processes = 2
    num_slides_per_process = 2
    num_subprocesses_per_process = 5
    slide_paths, df_m = generate_slide_paths_from_manifest(manifest_path=Configs.MANIFEST_PATH,
                                                           slides_dir=Configs.SLIDES_DIR)
    manifest_fullprocess_batch = np.array_split(df_m, np.ceil(len(df_m) / num_slides_per_process))
    full_batch_ind = 0
    processes = []
    try:
        while True:
            num_running_full_processes = sum([proc.poll() is None for proc in processes])
            if full_batch_ind == len(manifest_fullprocess_batch) and num_running_full_processes == 0:
                # finished preprocessing everything
                break
            if num_running_full_processes < num_allowed_full_processes:
                df_batch = manifest_fullprocess_batch[full_batch_ind]
                slide_ids = list(df_batch['id'])
                full_batch_ind += 1
                bash_str = get_bash_str_preprocess(slide_ids, num_subprocesses_per_process, full_batch_ind)
                proc = subprocess.Popen([bash_str,],
                                      stdout=subprocess.PIPE,
                                      stderr=subprocess.PIPE,
                                      text=True, shell=True)
                processes.append(proc)
                print(bash_str)
                print(f'New Process created with slides: {slide_ids}')
                continue
            sleep(2)
        print('Finished All!!')
    except Exception as e:
        print(f"Main program received {e}")
        print('Stopping subprocesses...')
        for subproc in processes:
            subproc.terminate()
            subproc.wait()
        print("Subprocesses terminated.")