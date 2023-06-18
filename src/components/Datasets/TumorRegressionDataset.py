from torch.utils.data import Dataset
from PIL import Image
from src.components.Objects.Logger import Logger


class TumorRegressionDataset(Dataset, Logger):
    def __init__(self, df, transform=None, target_transform=None, with_y=True):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.target_transform = target_transform
        self.with_y = with_y
        self.log(f"""TumorRegressionDataset created with {len(self.df)} tiles.""", log_importance=1)

    def __getitem__(self, index):
        path = self.df['tile_path'][index]
        img = Image.open(path)
        y = self.df['dis_to_tum'][index]
        if self.transform:
            img = self.transform(img)
        if not self.with_y:
            return img
        if self.target_transform:
            y = self.target_transform(y)
        return img, y

    def join_metadata(self, df_pred, inds):
        df_pred.loc[:, self.df.columns] = self.df.loc[inds].values
        return df_pred

    def __len__(self):
        return len(self.df)
