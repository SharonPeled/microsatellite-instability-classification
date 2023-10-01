import subprocess
import os
from src.configs import Configs
from src.general_utils import get_time
import shutil
import pandas as pd
import numpy as np
from time import sleep
from dataclasses import dataclass
import argparse


slides_to_use = ['b740c836-c964-4fa0-b82b-f061919bebb6', 'b3509368-20af-405d-a87f-3efc5899799b', 'ed3a04f5-677e-4e09-89d4-77179a3cc868', 'bdf6ff1b-e570-488b-a7c5-53a0fa5490ea', 'e3b37b06-0e9d-4308-8bc0-c3750f7d7e69', 'c8886af9-f9d0-4191-81d5-166d5a97f3fb', 'c986be01-be19-4ba1-9cc7-46eb1be902a0', 'e6814b0a-eeea-460d-bfe5-3a9f5afcf8db', '996b9676-92e5-4f6c-a133-f5ea67c40328', 'd65fd90a-14d9-40b5-ad9e-22a64cb0f7ca', '636016e1-c0ba-4dec-9f0c-31dbe18b42c9', '1bfd5b0e-52b9-463a-b00e-32fe330ffb19', '7f97f874-cc06-41bc-bfcb-cdabe0b9b7e6', '4737f507-5b7a-4eaf-a4a4-75332da7e951', '863fb6b6-99fc-45d7-b48c-ad25ae9707b7', '183b22ce-1357-4e6e-8246-269d645032dd', 'd00857da-45ac-4bd1-9983-c092ad062728', '1fefdb5c-6ba0-4040-9e0f-821a00314b5b', '9ff047b2-01fa-416d-b5c0-0402478ee824', '6a750c4d-1742-48fa-8710-3a1e319fe608', 'eaa0dab0-ce66-4e89-b65d-d300e98c718a', '08c57d33-6c74-49e9-b4d6-5d95baef6823', '358c8471-a32a-463f-9320-9d33e11d7300', '0b960c19-bb70-41ea-b05e-fddeb2eabc1a', '2fbe48e6-0022-4a40-9596-f9c1679142fd', 'cba350b1-43f4-4924-a015-10649ca5ef44', 'd29ad30d-54e6-4cef-9797-d61d6ac55487', 'd768c047-40d6-4ec8-bc21-465749e3a0fe', 'acad5023-32f0-48aa-b4dd-80df4b06fd4d', '58c45996-745e-4625-8e70-4b0b823b4f4c', 'ef351cab-c611-423e-83fd-fc7b568f4878', '3de2f425-8f0f-49ed-9529-16ac671fc76d', 'a60f03c0-fd1e-4349-8498-d41529d41302', 'c07075f0-a48e-4b3b-bc62-cbea82eabd9d', '849725a3-364e-44ba-b2cc-0b3491c799f2', '68087d5a-fe90-41ce-a0bc-0a13d9709b99', 'acda3bcf-1dcf-42f0-bbc9-5241b4250d5b', '2acf254a-9e59-458c-b454-4fef8917d245', 'e4e08050-868e-40a6-a197-19b274bcb7c1', '36694ad1-b51c-4db6-9a67-148a07d57a14', '18be524e-13bd-459f-aa2d-c0e793285ec7', '12bdfa01-6543-4eeb-a7bd-7c2b9d583950', '10fcd874-d174-443d-a6b2-ea5ecc666593', '8e487485-116a-492c-8a09-786cbc940b53', '2a9955fb-82aa-4da3-9ba2-af39e15bbbc2', 'c7ed6e59-c047-446f-9512-58502209511d', '275f49a4-0aea-4986-b1e1-735935c44614', '408475f9-f914-408b-92df-799780ee98c3', '4ef5985b-06d6-4cae-a8e7-8af653c66ba0', '92c0545b-868f-4115-bcb9-730269362d34', '2f705ffd-8974-4185-b5e7-71e94b1e00da', '2212c523-3058-4ee0-a4ed-87a88041c418', '7cf0e77f-ead2-422b-8948-9d2e2d1c1cfe', '6f1b67be-7cbd-4d0a-950a-4e6e0c3c8e4b', '4e5cd04c-ca91-4ad4-8957-94bff6d9706a', '626be827-791d-417a-bf3c-119e09ea14dd', '36934609-e0db-45ac-a5d0-214eb1f849d7', 'c8913491-389c-42b8-b219-51fb8a97c705', 'e4fa0e45-0c12-4898-b993-0131ee3df6eb', 'ec747c94-609e-4cca-8c2e-bea9e1f9ee33', '53507819-e29d-46fe-9fdc-fddc6d67a912', 'be93ab4e-ae4d-47bf-b62d-e4db65427ae7', '6cddfa2c-1591-4609-b85b-74569eb35fe7', 'e43b39e2-04a7-4037-8038-23ee68a2baba', '625e06ee-9cfc-4227-9db5-d262178774bd', '13090bb0-4e33-4093-9f50-c3a785d0a6d5', '4a9b0b9c-b1df-4439-9194-1db903cc6847', 'e9a2f2c8-886f-4361-ae6e-10d3b99518ad', '438cbed7-4fa2-4781-a86c-a3be78f32f6b', '78238dfe-69d4-4fb8-8657-1e1bb9121603', 'd08cd647-b66a-4e8b-8cb1-2dc3ed619890', '1130ace8-92b2-40d9-8d39-16ed58f7758f', '3c354c22-9092-47f3-b39d-6f23943f9c99', 'b5bed4a0-7467-4647-a750-d616e1e44b1a', 'ae9ea9fc-e043-4edd-800f-91292e0b88be', '30538bc8-2c80-4658-ab2b-0c18950782d4', 'cd789b66-2c1b-4511-b206-8b640378b015', 'a9e204ad-2c34-448a-af10-530c0e62ad23', 'a0bc504c-21c4-485b-b4f0-1db32b78a8e3', 'd6c923a1-c478-4bca-af4a-eda490db13b1', '01802d42-53ff-4674-b336-43c19a60b319', '21ed73f1-a6a5-4f0f-85d2-23ed027c4456', 'f8d3452a-497d-402f-b014-1d760f0d29a7', 'a0818139-ad5d-40b6-9901-842dcdab13fc', '35370123-f7da-477a-b0fd-48fecad5d748', 'f860621e-ef92-4acd-8036-92d9449a657b', 'ca8e8988-947d-4546-ba37-84c663c91986', '2db377cb-e9d9-417f-ac57-19fa6ed9c4de', '0603abd9-a7dc-4fd9-b7ff-7be32164cbf0', '5f69c2c2-45ba-4b2c-9708-b67a86154003', '9a02a024-5597-4e5c-aadd-a1edb9fad508', 'e6aa69ed-ce5a-4614-890c-77ca37e41dda', '24ca5f6c-58a2-4efc-bfa8-5a254068421f', 'ec79599e-ee10-4a4c-847c-49198fd14aa9', '8d1f0721-9810-48f6-9e57-373df93f31be', 'a21467ab-2cbc-49b5-9cb0-cc82ce7c728b', 'cca476ea-6161-45d8-874b-d0755aa5fc13', 'ab1a108d-8360-44a7-9021-f8b603d7f93b', 'f747939a-06cb-4d2f-8746-0016b5f8fd71', '1bcb63fe-4f14-49ae-8416-239680bd8ba3', '6a9dabd3-053d-44a0-a126-06c7f394ca66', '666b0125-15bf-4c50-b1e7-e98e2acc2b6f', 'cd16e760-05c8-47cf-a2ed-52953f4a2b41', 'c8833706-ff94-4196-ae78-9f8bb2a76fce', '9ba14f4a-5709-4e6a-a1ff-c95c67b26b46', 'd1dc64ae-1761-4b18-a484-6f15bb415d27', '0055d875-a4ad-4250-b7da-2152f5761d92', '4ecadacd-3ed0-4b61-a2b1-6ccba5d96a6e', '6c17329b-af08-47b2-930e-677f7f54a694', '6b0943e9-1b3d-4fa6-9f29-a51101709be7', '0bab616a-1a03-43cb-b2b2-023be4ab30b3', '9d11a092-fc5a-4c68-a39c-40f78d17ac7b', '4223bd71-e3b1-4b59-a235-4ff0cc614b62', '5c0685db-157f-4d01-81ab-71228b6bd84c', '7876a435-b9e4-4077-9b29-7a3b329404cd', '7db21b70-0487-4c4c-b8d8-863407fb9b3b', '767b9528-0051-47ee-a2bd-319478e73611', 'afac1e21-6a3b-445c-bd39-889195ac6ed4', '10304880-e029-42e5-85c0-e8cdef839dec', '4ecf8840-d45e-4be3-9d2a-459a8081d62c', 'e0a4cd26-126f-4b2d-9db5-2976a4599de1', '88fae228-c288-41a2-b832-57504b7ca97c', '18bcd6a4-2359-4b04-9e8b-7385149919cf', '9c65cd64-56be-46d3-83a1-0f759e6b510c', '88fcd2b1-5383-4a35-8a1e-ccbbf0f24f20', '7b481ea1-6ad0-4578-b47f-4c46518d79e9', '7c7731c4-d5d6-4329-af31-5b1fcf07fcd3', '584df19e-70ad-4a53-9563-8075f2217d6e', '796a94ef-3c33-4300-8ce1-720afcfa9d51', '7791b13d-925e-4106-8219-65a1d722809a']


def generate_slide_paths_from_manifest(manifest_path, slides_dir):
    slide_paths = []
    df_m = pd.read_csv(manifest_path, sep='\t')
    for i, row in df_m.iterrows():
        slide_uuid = row['id']
        filename = row['filename']
        path = os.path.join(slides_dir, slide_uuid, filename)
        slide_paths.append(path)
    df_m['slide_path'] = slide_paths
    return df_m


def get_bash_str_preprocess(slide_ids, num_processes, full_batch_ind):
    slides_str = ' '.join(slide_ids)
    bash_str = f"""
    bash
    conda activate MSI
    python -u preprocess_full.py --num-processes {num_processes} --slide_ids {slides_str} --slide_dir {Configs.SLIDES_DIR} --full_batch_ind {full_batch_ind} >> {full_batch_ind}_preprocess_full_{get_time()}.txt 2>&1
    """
    return bash_str


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_allowed_full_processes', type=int)
    parser.add_argument('--num_slides_per_process', type=int)
    parser.add_argument('--num_subprocesses_per_process', type=int)
    args = parser.parse_args()

    num_allowed_full_processes = args.num_allowed_full_processes
    num_slides_per_process = args.num_slides_per_process
    num_subprocesses_per_process = args.num_subprocesses_per_process
    df_m = generate_slide_paths_from_manifest(manifest_path=Configs.MANIFEST_PATH,
                                              slides_dir=Configs.SLIDES_DIR)
    df_m = df_m[df_m['id'].isin(slides_to_use)]
    print(f'len df_m: {len(df_m)}')
    manifest_fullprocess_batch = np.array_split(df_m, np.ceil(len(df_m) / num_slides_per_process))
    full_batch_ind = 0
    processes = []
    try:
        while True:
            num_running_full_processes = sum([proc.poll() is None for proc in processes])
            if full_batch_ind == len(manifest_fullprocess_batch) and num_running_full_processes == 0:
                # finished preprocessing everything
                break
            if num_running_full_processes < num_allowed_full_processes and \
                    full_batch_ind < len(manifest_fullprocess_batch):
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
            sleep(30)
        print('Finished All!!')
    except Exception as e:
        print(f"Main program received {e}")
        print('Stopping subprocesses...')
        for subproc in processes:
            subproc.terminate()
            subproc.wait()
        print("Subprocesses terminated.")