from torch.utils.data import Dataset
from glob import glob
from PIL import Image
from src.components.objects.Logger import Logger
import pandas as pd


class TumorTileDataset(Dataset, Logger):
    def __init__(self, dir, img_extension, y=None, transform=None, crop_and_agg=False):
        """
        crop_and_agg: the model trained on 224*224 tiles at 0.5MPP while this dataset is
        512*512 at 0.5MPP. To bridge this gap this option crop the big tile into 4 non-overlapping tiles
        and agg their predictions.
        """
        self.dir = dir
        self.transform = transform
        tile_paths = glob(f"{dir}/**/*.{img_extension}", recursive=True)
        self.df = pd.DataFrame(tile_paths, columns=['tile_path'])
        self.y = y  # same y for images, there are all tumors
        self.crop_and_agg = crop_and_agg
        if self.crop_and_agg:
            self.generate_cropped_df()
        self.log(f"""ImageBagDataset created with {len(self.df)} images from {self.dir}.""", log_importance=1)

    def join_metadata(self, df_pred, inds):
        df_pred.loc[:, self.df.columns] = self.df.loc[inds].values
        return df_pred

    def __getitem__(self, index):
        path = self.df['tile_path'][index]
        img = Image.open(path)
        if self.crop_and_agg:
            img = img.crop(self.df['crop_coords'][index])
        if self.transform:
            img = self.transform(img)
        if self.y is not None:
            return img, self.y
        return img

    def __len__(self):
        return len(self.df)

    def generate_cropped_df(self):
        self.df = pd.concat([self.df for _ in range(4)], ignore_index=True)
        crop_coords_list = [(0, 0, 256, 256),
                       (0, 256, 256, 512),
                       (256, 0, 512, 256),
                       (256, 256, 512, 512)]
        self.df['crop_coords'] = [crop_coords for crop_coords in crop_coords_list
                                  for _ in range(len(self.df)//4)]

