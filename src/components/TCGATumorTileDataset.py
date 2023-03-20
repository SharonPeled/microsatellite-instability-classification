from torch.utils.data import Dataset
from glob import glob
from PIL import Image
from ..components.Logger import Logger
import pandas as pd


class TCGATumorTileDataset(Dataset, Logger):
    def __init__(self, dir, img_extension, y=None, transform=None):
        self.dir = dir
        self.transform = transform
        tile_paths = glob(f"{dir}/**/*.{img_extension}", recursive=True)
        self.df = pd.DataFrame(tile_paths, columns=['tile_path'])
        self.y = y  # same y for images, there are all tumors
        self.log(f"""ImageBagDataset created with {len(self.df)} images.""", log_importance=1)

    def join_metadata(self, df_pred, inds):
        df_pred.loc[:, self.df.columns] = self.df.loc[inds].values
        return df_pred

    def __getitem__(self, index):
        path = self.df['tile_path'][index]
        img = Image.open(path)
        if self.transform:
            img = self.transform(img)
        if self.y is not None:
            return img, self.y
        return img

    def __len__(self):
        return len(self.df)
