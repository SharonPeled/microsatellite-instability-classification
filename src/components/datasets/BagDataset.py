from torch.utils.data import Dataset
import numpy as np
from PIL import Image
from src.components.objects.Logger import Logger
import torch


class BagDataset(Dataset, Logger):
    def __init__(self, df_labels, slide_sample_size, cohort_to_index=None, transform=None, target_transform=None):
        self.slide_sample_size = slide_sample_size
        self.df_labels = df_labels.set_index(df_labels.slide_uuid.values)
        self.slide_uuids = list(self.df_labels.slide_uuid.unique())
        self.cohort_to_index = cohort_to_index
        self.transform = transform
        self.target_transform = target_transform
        self.log(f"BagDataset created with {len(self.slide_uuids)} slides, " +
                 f"and {len(df_labels)} tiles.",
                 log_importance=1)

    def __getitem__(self, index):
        slide_uuid = self.slide_uuids[index]
        df_s = self.df_labels.loc[slide_uuid]
        df_s = df_s.sample(min(len(df_s), self.slide_sample_size))
        tiles = [self.load_image_safe(path) for path in df_s.tile_path]
        y = df_s.y.iloc[0]
        c = df_s.cohort.apply(lambda c: self.cohort_to_index[c]).iloc[0]
        tile_path = df_s.tile_path.iloc[0]
        s = df_s.slide_uuid.iloc[0]
        p = df_s.patient_id.iloc[0]
        return torch.stack(tiles), c, y, s, p, tile_path

    def load_image_safe(self, path):
        try:
            # Try to open the image from the given path
            image = Image.open(path)
        except Exception as e:
            # If loading the image fails, create an empty (white) image
            image = Image.new('RGB', (224, 224), color='white')
            self.log('-'*25 + f"Invalid image: {path}, Error: {e}"+ '-'*25,
                     log_importance=1)
        if self.transform:
            image = self.transform(image)
        return image

    def __len__(self):
        return len(self.slide_uuids)


