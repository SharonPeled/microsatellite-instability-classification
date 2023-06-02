from torch.utils.data import Dataset
from glob import glob
import os
import pandas as pd
import pyvips
from ..components.Logger import Logger
from PIL import Image


class ThumbnailDataset(Dataset, Logger):
    def __init__(self, label_file, slides_dir, class_to_ind, transform=None, target_transform=None):
        self.df = self._load_df(label_file, slides_dir)
        self.df = self.df[self.df.subtype.isin(class_to_ind.keys())].reset_index(drop=True)
        self.df['label'] = self.df.subtype.apply(lambda s: class_to_ind[s])
        self.class_to_ind = class_to_ind
        self.transform = transform
        self.target_transform = target_transform
        self.log(f"""ThumbnailDataset created with {len(self.df)} slides.""", log_importance=1)

    def _load_df(self, label_file, slides_dir):
        df_labels = pd.read_csv(label_file)
        slides_path = glob(f"{slides_dir}/**/*.svs", recursive=True)
        df = pd.DataFrame({'slide_path', slides_path})
        df['patient_id'] = df.slide_path.apply(lambda s: s[:12])
        df = df.merge(df_labels, on='patient_id', how='inner').reset_index(drop=True)
        return df

    def __getitem__(self, index):
        row = self.df.iloc[index]
        path = row['slide_path']
        thumb = pyvips.Image.thumbnail(path, 224)
        img = Image.fromarray(thumb.numpy())
        y = row['label']
        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            y = self.target_transform(y)
        return img, y

    def __len__(self):
        return len(self.df)


# def generate_thumbnails_with_tissue_classification(df_pred, slides_dir, class_to_index, class_to_color,
#                                                    summary_df_filename, summary_df_pred_merged_filename,
#                                                    thumbnail_filename):
class ThumbnailSegDataset(ThumbnailDataset):
    def __init__(self, label_file, slides_dir, class_to_ind, pred_dir, ss_class_to_ind, summary_df_pred_merged_filename,
                 transform=None, target_transform=None):
        super(ThumbnailSegDataset, self).__init__(label_file, slides_dir, class_to_ind, transform=transform,
                                                  target_transform=target_transform)
        self.df['summary_df_path'] = self.df.slide_path.apply(lambda p: os.path.join(os.path.dirname(p),
                                                                                     summary_df_pred_merged_filename))



