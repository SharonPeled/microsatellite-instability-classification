import subprocess
import os
from src.general_utils import get_time
import shutil
import pandas as pd
import numpy as np
from time import sleep
import argparse


def download_slides(slides_dir, slides_str):
    try:
        print('Start Downloading ..')
        bash_str = f"""
    . /home/sharonpe/miniconda3/etc/profile.d/conda.sh
    conda activate gdc
    cd {slides_dir}
    gdc-client download {slides_str} > "download_log_{get_time()}.txt"
                """
        proc = subprocess.Popen([bash_str], stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                                text=True,
                                shell=True)
        proc.wait()
        print(f"Finished Download {len(slides_str.split(' '))}.")
    except Exception as e:
        print(f"Main program received {e}")
        print('Stopping subprocesses...')
        proc.terminate()
        proc.wait()
        print("Subprocesses terminated.")


def delete_slides(slide_ids, slides_dir):
    print('Start Delete ..')
    for slide_id in slide_ids:
        dir_path = os.path.join(slides_dir, slide_id)
        shutil.rmtree(dir_path)
    print(f"Finished Delete {len(slide_ids)}.")


def get_bash_str_preprocess(tile_size, slide_ids, num_processes):
    slides_str = ' '.join(slide_ids)
    bash_str = f"""
    . /home/sharonpe/miniconda3/etc/profile.d/conda.sh
    conda activate MSI
    python main.py --preprocess --num-processes {num_processes} --config_filepath config_files/preprocess_{tile_size}.json --slide_ids {slides_str}
    """
    return bash_str


def main(args):
    slide_ids = [slide_id.strip("'") for slide_id in args.slide_ids]
    slides_dir = args.slide_dir
    num_processes = args.num_processes
    print(f'Starting processing slides: {slide_ids}')
    try:
        download_slides(slides_dir=slides_dir, slides_str=' '.join(slide_ids))

        bash_str = get_bash_str_preprocess(512, slide_ids, num_processes)
        proc1 = subprocess.Popen([bash_str, ], stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE,
                                 text=True, shell=True)
        print(bash_str)

        bash_str = get_bash_str_preprocess(1024, slide_ids, num_processes)
        print(bash_str)
        proc2 = subprocess.Popen([bash_str, ], stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE,
                                 text=True, shell=True)

        bash_str = get_bash_str_preprocess(224, slide_ids, num_processes)
        print(bash_str)
        proc3 = subprocess.Popen([bash_str, ], stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE,
                                 text=True, shell=True)
        print('Started 3 tiling processes.')

        proc1.wait()
        proc2.wait()
        proc3.wait()
        print(proc1.stderr.readlines())
        print(proc2.stderr.readlines())
        print(proc3.stderr.readlines())

        delete_slides(slide_ids, slides_dir)
    except Exception as e:
        print(f"Main program received {e}")
        print('Stopping subprocesses...')
        for proc in [proc1, proc2, proc3]:
            proc.terminate()
            proc.wait()
        print("Subprocesses terminated.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--slide_ids", nargs="+", type=str)
    parser.add_argument('--num-processes', type=int)
    parser.add_argument('--slide_dir', type=str)
    args = parser.parse_args()
    main(args)
