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
    scores = torch.concat([out["scores"].reshape(-1, 3, out["scores"].shape[-1] // 3)
                           for out in outputs])
    logits = softmax(scores, dim=1).permute(2, 0, 1).numpy()
    y_true = torch.concat([out["y"] for out in outputs], dim=0).transpose(1, 0).numpy()
    auc_per_snp = []
    for i in tqdm(range(y_true.shape[0]), total=y_true.shape[0]):
        auc_per_snp.append(calc_safe_auc(y_true[i, :], logits[i, :], multi_class='ovr', average=None))
    df_auc = pd.DataFrame(np.stack(auc_per_snp), columns=['0', '1', '2']).astype(float)
    df_auc['mean_auc'] = df_auc[['0', '1', '2']].mean(axis=1)
    return df_auc


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

cohorts = ['COAD', 'STAD']
snp_types = ['dna_repair', 'random']
artifact_dir = '/home/sharonpe/work/microsatellite-instability-classification/data/experiments_artifacts/VC_TILE_SSL_VIT_{cohort}_{snp_type}'
dicts = []
for cohort in cohorts:
    for snp_type in snp_types:
        dir_path = artifact_dir.format(cohort=cohort, snp_type=snp_type)
        tensor_files = glob(f'{dir_path}/**/*.tensor', recursive=True)
        if len(tensor_files) == 0:
            continue
        for fold in os.listdir(dir_path):
            fold_dir = os.path.join(dir_path, fold, 'test')
            if os.path.exists(fold_dir):
                assert len(os.listdir(fold_dir)) <= 3
                for tensor_filename in os.listdir(fold_dir):
                    # print(os.path.join(fold_dir, tensor_filename))
                    print(tensor_filename)
                    dicts.append({
                        'cohort': cohort,
                        'snp_type': snp_type,
                        'fold': fold,
                        'path':os.path.join(fold_dir, tensor_filename)
                    })
        print(dir_path)
        print()
df_t = pd.DataFrame(dicts)
print(df_t)
for (cohort, snp_type), df_g in df_t.groupby(['cohort', 'snp_type']):
    print(f'{datetime.now()}: Starting: {(cohort, snp_type)}')
    outputs = []
    for i,row in df_g.iterrows():
        print(f'{datetime.now()}: Loading Fold: {i}')
        fold_outputs = torch.load(row['path'])
        if fold_outputs is not None:
            outputs += fold_outputs
    df_auc = create_df_auc(outputs)
    df_auc_path = os.path.join(artifact_dir.format(cohort=cohort, snp_type=snp_type),
                               f'df_auc_test_{len(outputs)}.csv')
    df_auc.to_csv(df_auc_path, index=False)
    print(f'Saved in {df_auc_path}')





