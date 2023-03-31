from torch.utils.data import Dataset
from glob import glob
import os
import pandas as pd
from PIL import Image
from ..components.Logger import Logger


class TumorRegressionDataset(Dataset, Logger):
    def __init__(self, df, transform=None, target_transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.target_transform = target_transform
        self.log(f"""TumorRegressionDataset created with {len(self.df)} tiles.""", log_importance=1)

    def __getitem__(self, index):
        path = self.df['tile_path'][index]
        img = Image.open(path)
        y = self.df['dis_to_tum'][index]
        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            y = self.target_transform(y)
        return img, y

    def __len__(self):
        return len(self.df)
