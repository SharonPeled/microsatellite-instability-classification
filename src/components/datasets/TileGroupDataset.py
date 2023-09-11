from torch.utils.data import Dataset
import numpy as np
from PIL import Image
from src.components.objects.Logger import Logger
import torch


class TileGroupDataset(Dataset, Logger):
    def __init__(self, df_labels, cohort_to_index=None, transform=None, target_transform=None):
        self.df_labels = df_labels
        self.df_slide_list = [df_grouped.reset_index(drop=True) for _, df_grouped in
                              df_labels.groupby('slide_uuid', as_index=False)]
        self.cohort_to_index = cohort_to_index
        self.transform = transform
        self.target_transform = target_transform
        self.log(f"TileGroupDataset created with {len(self.df_slide_list)} slides, " +
                 f"and {len(df_labels)} tiles.",
                 log_importance=1)

    def join_metadata(self, df_pred, inds):
        df_pred.loc[:, 'slide_uuid'] = [self.df_slide_list[ind].iloc[0]['slide_uuid'] for ind in inds]
        return df_pred

    def __getitem__(self, index):
        return self.df_slide_list[index]

    def __len__(self):
        return len(self.df_slide_list)
