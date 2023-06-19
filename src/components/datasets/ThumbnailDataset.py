from torch.utils.data import Dataset
from glob import glob
import os
import pandas as pd
import pyvips
from src.components.objects.Logger import Logger
from PIL import Image
import numpy as np
import torch


class ThumbnailDataset(Dataset, Logger):
    def __init__(self, df_labels, size, transform=None, target_transform=None):
        self.df = df_labels.reset_index(drop=True)
        self.size = size
        self.transform = transform
        self.target_transform = target_transform
        self.log(f"""ThumbnailDataset created with {len(self.df)} slides.""", log_importance=1)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        path = row['slide_path']
        thumb = pyvips.Image.thumbnail(path, self.size)
        # require to pad to desired size - not functional at the time
        img = Image.fromarray(thumb.numpy()[:, :, :3]) # removing alpha channel
        y = row['y']
        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            y = self.target_transform(y)
        return img, y

    def __len__(self):
        return len(self.df)

    def join_metadata(self, df_pred, inds):
        df_pred.loc[:, self.df.columns] = self.df.loc[inds].values
        return df_pred


class ThumbnailSegDataset(ThumbnailDataset):
    def __init__(self, labels_filepath, slides_dir, summary_df_pred_merged_filename, ss_class_to_ind,
                 class_to_ind=None, thumbnail_transform=None, ss_transform=None, target_transform=None):
        super(ThumbnailSegDataset, self).__init__(labels_filepath, slides_dir, class_to_ind,
                                                  transform=thumbnail_transform,
                                                  target_transform=target_transform)
        self.ss_transform = ss_transform
        self.prob_cols = [f'{col}_prob' for col in ss_class_to_ind.keys()]
        self.df['df_pred_path'] = self.df.slide_path.apply(lambda p: os.path.join(os.path.dirname(p),
                                                                                  summary_df_pred_merged_filename))

    def create_ss_tensor(self, df_path):
        df = pd.read_csv(df_path)
        max_row, max_col = df.row.max() + 1, df.col.max() + 1
        # Create an empty array with zeros
        img_np = np.zeros((max_row, max_col, len(self.prob_cols)))
        # Create a boolean mask for legal cells
        df_tissue = df[df['Tissue']]
        # Fill the matrix_array with the values from the DataFrame
        img_np[df_tissue['row'].values, df_tissue['col'].values] = df_tissue[self.prob_cols].values
        img_np_T = img_np.transpose((2, 0, 1))
        # Convert the transposed array to a PyTorch tensor
        tensor_img = torch.from_numpy(img_np_T)
        return tensor_img

    def __getitem__(self, index):
        row = self.df.iloc[index]
        thumb_tensor, y = super().__getitem__(index)
        ss_tensor = self.create_ss_tensor(row['df_pred_path'][index])
        if self.ss_transform:
            ss_tensor = self.ss_transform(ss_tensor)
        return torch.concat([thumb_tensor, ss_tensor], dim=0), y

    def __len__(self):
        return len(self.df)




