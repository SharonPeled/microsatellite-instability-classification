from torch.utils.data import Dataset
import numpy as np
from PIL import Image
from src.components.objects.Logger import Logger
import torch


class ProcessedTileDataset(Dataset, Logger):
    def __init__(self, df_labels, transform=None, target_transform=None, group_size=-1):
        self.df_labels = df_labels.reset_index(drop=True)
        self.transform = transform
        self.target_transform = target_transform
        self.group_size = group_size
        if self.group_size > 1:
            # TODO: random state here
            self.df_labels = self.df_labels.sample(frac=1, random_state=None).reset_index(drop=True)
            self.df_labels['group_id'] = self.df_labels.groupby('slide_uuid').cumcount() // group_size
            self.df_labels = self.df_labels.groupby(['slide_uuid', 'group_id']).filter(lambda x: len(x) == group_size)
            self.df_labels['slide_group_id'] = self.df_labels.groupby(['slide_uuid', 'group_id']).ngroup()
            self.df_labels.set_index('slide_group_id', inplace=True)
        self.index_shift = 0
        self.dataset_full_length = self.df_labels.index.nunique()
        self.dataset_length = self.dataset_full_length
        self.log(f"ProcessedTileDataset created with {self.df_labels.slide_uuid.nunique()} slides, " +
                 f"{self.dataset_full_length} groups, and {len(self.df_labels)} tiles.",
                 log_importance=1)

    def deploy_dataset_limits(self, dataset_limits):
        """
        dataset_limits: (steps_to_fastforward, num_steps_to_execute)
        """
        self.index_shift = dataset_limits[0]
        if dataset_limits[1] == -1:
            self.dataset_length = self.dataset_full_length - self.index_shift
        else:
            self.dataset_length = dataset_limits[1]
        self.log(f"ProcessedTileDataset was limited to steps from {self.index_shift} to {self.index_shift+self.dataset_length} ", log_importance=1)

    def join_metadata(self, df_pred, inds):
        if self.group_size > 1:
            return df_pred.merge(self.df_labels, how='inner', left_on='dataset_ind', right_on='slide_group_id')
        else:
            df_pred.loc[:, self.df_labels.columns] = self.df_labels.loc[inds].values
            return df_pred

    def __getitem__(self, index):
        index += self.index_shift
        if self.group_size == -1:
            tile_path = self.df_labels.tile_path[index]
            y = self.df_labels.y[index]
            return self.load_single_tile(tile_path, y)
        else:
            return self.load_group_tiles(index)

    def load_group_tiles(self, index):
        group_rows = self.df_labels.loc[index]  # Get all rows of the group
        images = []
        labels = []
        for _, row in group_rows.iterrows():
            tile_path = row['tile_path']
            y = row['y']
            img, y = self.load_single_tile(tile_path, y)
            images.append(img)
            labels.append(y)
        return torch.stack(images), labels[-1]

    def load_single_tile(self, tile_path, y):
        img = Image.open(tile_path)
        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            y = self.target_transform(y)
        return img, y

    def __len__(self):
        return self.dataset_length
