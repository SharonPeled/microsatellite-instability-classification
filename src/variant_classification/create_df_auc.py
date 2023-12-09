import numpy
import torch
from torch.nn.functional import softmax
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, classification_report
import os
from datetime import datetime
from tqdm import tqdm
from glob import glob


def create_df_auc(outputs):
    # tile based
    scores_tile = torch.concat([out["scores"].reshape(-1, 3, out["scores"].shape[-1] // 3)
                                for out in outputs])
    y_true_tile = torch.concat([out["y"] for out in outputs], dim=0).transpose(1, 0).numpy()
    logits_tile = softmax(scores_tile, dim=1).permute(2, 0, 1).numpy()
    # slide based
    slide_uuids = np.concatenate([out["slide_id"] for out in outputs])
    df_inds = pd.DataFrame({'slide_uuid': slide_uuids, 'ind': range(len(slide_uuids))})
    scores_slide_list = []
    y_true_slide_list = []
    for slide_uuid, df_s in df_inds.groupby('slide_uuid'):
        inds = df_s.ind.values
        scores_slide_list.append(scores_tile[inds].mean(dim=0))
        y_true_slide_list.append(torch.tensor(y_true_tile[:, inds[0]]))
    scores_slide = torch.stack(scores_slide_list)
    y_true_slide = torch.stack(y_true_slide_list).transpose(1, 0).numpy()
    logits_slide = softmax(scores_slide, dim=1).permute(2, 0, 1).numpy()

    auc_per_snp_tile = []
    auc_per_snp_slide = []
    for i in tqdm(range(y_true_tile.shape[0]), total=y_true_tile.shape[0]):
        auc_per_snp_tile.append(calc_safe_auc(y_true_tile[i, :], logits_tile[i, :], multi_class='ovr', average=None))
        auc_per_snp_slide.append(calc_safe_auc(y_true_slide[i, :], logits_slide[i, :], multi_class='ovr', average=None))

    df_auc_tile = pd.DataFrame(np.stack(auc_per_snp_tile), columns=['0', '1', '2']).astype(float)
    df_auc_slide = pd.DataFrame(np.stack(auc_per_snp_slide), columns=['0', '1', '2']).astype(float)
    return df_auc_tile, df_auc_slide


def calc_safe_auc(y_true_i, logits_i, **kwargs):
        available_classes = np.unique(y_true_i)
        if len(available_classes) == logits_i.shape[-1]:
            return roc_auc_score(y_true_i, logits_i, **kwargs)
        if len(available_classes) == 1:
            return [None for _ in range(3)]
        res = []
        for c in range(3):
            if c in available_classes:
                res.append(roc_auc_score(y_true_i, logits_i[:, c], **kwargs))
            else:
                res.append(None)
        return np.array(res)

# per fold
# per slide
# dir/fold/test/tensor

cohorts = ['COAD', 'STAD', 'UCEC']
snp_types = ['dna_repair', 'random']
artifact_dir = '/home/sharonpe/work/microsatellite-instability-classification/data/experiments_artifacts/VC_TILE_SSL_VIT_{cohort}_{snp_type}'
dicts = []
for cohort in cohorts:
    for snp_type in snp_types:
        dir_path = artifact_dir.format(cohort=cohort, snp_type=snp_type)
        for fold in range(3):
            print('\n\n\n')
            print(f'{cohort}, {snp_type}, {fold}')
            outputs_path = list(glob(f'{dir_path}/{fold}/test/*.tensor'))
            if len(outputs_path) != 1:
                print('-'*50 + f'{cohort}, {snp_type}, {fold} - SKIPPED!!')
                print(f'output_path: {outputs_path}')
                continue
            outputs_path = outputs_path[0]
            print(outputs_path)
            outputs = torch.load(outputs_path)
            df_auc_tile, df_auc_slide = create_df_auc(outputs)
            df_auc_tile_path = os.path.join(dir_path, str(fold), 'test', f'df_auc_tile_test_{fold}_{len(outputs)}.csv')
            df_auc_slide_path = os.path.join(dir_path, str(fold), 'test', f'df_auc_slide_test_{fold}_{len(outputs)}.csv')
            df_auc_tile.to_csv(df_auc_tile_path, index=False)
            df_auc_slide.to_csv(df_auc_slide_path, index=False)
            print(f'Tile saved in {df_auc_tile_path}')
            print(f'Slide saved in {df_auc_slide_path}')


