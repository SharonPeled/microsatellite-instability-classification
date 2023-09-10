from torch.utils.data import Dataset
import numpy as np
from PIL import Image
from src.components.objects.Logger import Logger
import torch


class TileGroupDataset(Dataset, Logger):
    def __init__(self, df_labels, cohort_to_index=None, transform=None, target_transform=None):
        self.df_slide_list = [df_grouped.reset_index(drop=True) for _, df_grouped in
                              df_labels.groupby('slide_uuid')]
        self.cohort_to_index = cohort_to_index
        self.transform = transform
        self.target_transform = target_transform
        self.log(f"TileGroupDataset created with {len(self.df_slide_list)} slides, " +
                 f"and {len(df_labels)} tiles.",
                 log_importance=1)

    # def join_metadata(self, df_pred, inds):
    #     if self.group_size > 1:
    #         return df_pred.merge(self.df_labels, how='inner', left_on='dataset_ind', right_on='slide_group_id')
    #     else:
    #         df_pred.loc[:, self.df_labels.columns] = self.df_labels.loc[inds].values
    #         return df_pred

    def __getitem__(self, index):
        return self.df_slide_list[index].values

    def __len__(self):
        return len(self.df_slide_list)
