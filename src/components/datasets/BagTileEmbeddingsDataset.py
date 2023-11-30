from torch.utils.data import Dataset
from src.components.objects.Logger import Logger
import torch


class BagTileEmbeddingsDataset(Dataset, Logger):
    def __init__(self, df_labels, cohort_to_index=None, transform=None, target_transform=None):
        self.df_labels = df_labels.reset_index(drop=True)
        self.cohort_to_index = cohort_to_index
        self.transform = transform
        self.target_transform = target_transform
        self.log(f"BagDataset created with {len(self.df_labels)} slides, ",
                 log_importance=1)

    def __getitem__(self, index):
        row = self.df_labels.iloc[index]
        tile_embeddings = torch.load(row['tile_embeddings_path'])
        y = row['y']
        c = self.cohort_to_index[row['cohort']]
        s = row['slide_uuid']
        p = row['patient_id']
        return tile_embeddings, c, y, s, p

    def __len__(self):
        return len(self.df_labels)


