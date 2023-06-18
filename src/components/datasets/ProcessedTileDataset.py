from torch.utils.data import Dataset
from glob import glob
import os
import pandas as pd
from PIL import Image
from src.components.objects.Logger import Logger


class ProcessedTileDataset(Dataset, Logger):
    def __init__(self, processed_tiles_dir, transform=None):
        self.processed_tiles_dir = processed_tiles_dir
        self.transform = transform
        tile_paths = glob(f"{processed_tiles_dir}/**/*Tissue*.jpg", recursive=True)
        self.df = pd.DataFrame(tile_paths, columns=['tile_path'])
        self.df['slide_uuid'] = self.df.tile_path.apply(lambda p: os.path.basename(os.path.dirname(p)))
        self.log(f"""ProcessedTileDataset created with {len(self.df)} tiles.""", log_importance=1)

    def join_metadata(self, df_pred, inds):
        df_pred.loc[:, self.df.columns] = self.df.loc[inds].values
        return df_pred

    def __getitem__(self, index):
        path = self.df['tile_path'][index]
        img = Image.open(path)
        if self.transform:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.df)
