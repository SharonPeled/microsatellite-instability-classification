from torch.utils.data import Dataset
from glob import glob
import os
import pandas as pd
from PIL import Image
from src.components.objects.Logger import Logger


class ProcessedTileDataset(Dataset, Logger):
    def __init__(self, df_labels, transform=None, target_transform=None):
        self.df_labels = df_labels.reset_index(drop=True)
        self.transform = transform
        self.target_transform = target_transform
        self.log(f"""ProcessedTileDataset created with {self.df_labels.slide_uuid.nunique()} slides and {len(self.df_labels)} tiles.""", log_importance=1)

    def join_metadata(self, df_pred, inds):
        df_pred.loc[:, self.df_labels.columns] = self.df_labels.loc[inds].values
        return df_pred

    def __getitem__(self, index):
        row = self.df_labels.iloc[index]
        path = row['tile_path']
        img = Image.open(path)
        y = row['y']
        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            y = self.target_transform(y)
        return img, y

    def __len__(self):
        return len(self.df_labels)
