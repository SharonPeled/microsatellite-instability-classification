from torch.utils.data import Dataset
import numpy as np
from src.components.objects.Logger import Logger
import torch
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


class ProcessedTileDataset(Dataset, Logger):
    def __init__(self, df_labels, cohort_to_index=None, transform=None, target_transform=None, group_size=-1,
                 num_mini_epochs=0, pretraining=False, mini_epoch_shuffle_seed=None):
        self.mini_epoch_shuffle_seed = mini_epoch_shuffle_seed
        self.pretraining = pretraining
        self.df_labels = df_labels.reset_index(drop=True)
        self.cohort_to_index = cohort_to_index
        self.transform = transform
        self.target_transform = target_transform
        self.group_size = group_size
        if self.group_size > 1:
            # deprecated - see TileGroupDataset
            self.df_labels = self.df_labels.sample(frac=1, random_state=None).reset_index(drop=True)
            self.df_labels['group_id'] = self.df_labels.groupby('slide_uuid').cumcount() // group_size
            self.df_labels = self.df_labels.groupby(['slide_uuid', 'group_id']).filter(lambda x: len(x) == group_size)
            self.df_labels['slide_group_id'] = self.df_labels.groupby(['slide_uuid', 'group_id']).ngroup()
            self.df_labels.set_index('slide_group_id', inplace=True)
        self.dataset_full_length = self.df_labels.index.nunique()
        self.dataset_length = self.dataset_full_length
        self.num_mini_epochs = num_mini_epochs
        self.index_shift = 0
        self.init_mini_epochs()
        self.log(f"ProcessedTileDataset created with {self.df_labels.slide_uuid.nunique()} slides, " +
                 f"{self.dataset_full_length} groups, and {len(self.df_labels)} tiles.",
                 log_importance=1)

    def init_mini_epochs(self):
        if self.num_mini_epochs < 2:
            return
        self.dataset_length = self.dataset_full_length // self.num_mini_epochs
        self.index_shift = 0

    def set_mini_epoch(self, epoch):
        if self.num_mini_epochs < 2:
            return
        self.index_shift = self.dataset_length * (epoch % self.num_mini_epochs)
        if self.index_shift == 0 and self.pretraining:
            self.df_labels = self.df_labels.sample(frac=1, random_state=self.mini_epoch_shuffle_seed + epoch)
        Logger.log(f"Mini epoch number {epoch}, index_shift: {self.index_shift}.")

    def join_metadata(self, df_pred, inds):
        if self.group_size > 1:
            return df_pred.merge(self.df_labels, how='inner', left_on='dataset_ind', right_on='slide_group_id')
        else:
            df_pred.loc[:, self.df_labels.columns] = self.df_labels.loc[inds].values
            return df_pred

    def __getitem__(self, index):
        index += self.index_shift
        # TODO: index can be a slice
        if self.group_size == -1:
            row = self.df_labels.loc[index]
            return self.load_single_tile(row)
        else:
            imgs, cohort, y, slide_id, patient_id = self.load_group_tiles(index)
            if self.cohort_to_index is not None:
                return imgs, cohort, y, slide_id, patient_id
            return imgs, y, slide_id, patient_id

    def load_group_tiles(self, index):
        group_rows = self.df_labels.loc[index]  # Get all rows of the group
        images = []
        labels = []
        cohorts = []
        slide_ids = []
        patient_ids = []
        for _, row in group_rows.iterrows():
            img, cohort, y, slide_id, patient_id = self.load_single_tile(row)
            images.append(img)
            labels.append(y)
            cohorts.append(cohort)
            slide_ids.append(slide_id)
            patient_ids.append(patient_id)
        return torch.stack(images), cohorts[-1], labels[-1], slide_ids, patient_ids

    def load_single_tile(self, row):
        img = self.load_image_safe(row['tile_path'])
        if self.transform:
            img = self.transform(img)
        if self.pretraining:
            return img, self.cohort_to_index[row['cohort']]
        y = row['y']
        slide_id = row['slide_id']
        patient_id = row['patient_id']
        if self.target_transform:
            y = self.target_transform(y)
        if self.cohort_to_index is not None:
            return img, self.cohort_to_index[row['cohort']], y, slide_id, patient_id, row['tile_path']
        return img, None, y, slide_id, patient_id, row['tile_path']

    def load_image_safe(self, path):
        try:
            # Try to open the image from the given path
            image = Image.open(path)
        except Exception as e:
            # If loading the image fails, create an empty (white) image
            image = Image.new('RGB', (224, 224), color='white')
            self.log('-'*25 + f"Invalid image: {path}, Error: {e}"+ '-'*25,
                     log_importance=1)
        return image

    def apply_dataset_reduction(self, iter_args, scores):
        self.df_labels['epoch_score'] = scores
        if iter_args['schedule_type'] == 'step':
            self.df_labels = self.df_labels.groupby('slide_uuid', as_index=False).apply(
                lambda d: d.sort_values('epoch_score', ascending=False)[:int(len(d) * iter_args['reduction_factor'])])
            self.df_labels.reset_index(drop=True, inplace=True)
            self.dataset_length = len(self.df_labels)

    def __len__(self):
        return self.dataset_length
