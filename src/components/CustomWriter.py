from pytorch_lightning.callbacks import BasePredictionWriter
import torch
import numpy as np
import pandas as pd
import os


class CustomWriter(BasePredictionWriter):
    def __init__(self, output_dir, write_interval, class_to_index, dataset):
        super().__init__(write_interval)
        self.output_dir = output_dir
        self.class_to_index = class_to_index
        self.score_names = list(class_to_index.keys())
        self.dataset = dataset

    def write_on_batch_end(self, trainer, pl_module, prediction, batch_indices, batch, batch_idx, dataloader_idx):
        pass

    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
        batch_indices = np.concatenate(batch_indices[0])
        predictions = torch.concat(predictions[0])
        df_pred = pd.DataFrame(data=predictions, columns=self.score_names)
        df_pred['dataset_ind'] = batch_indices
        df_pred = self.dataset.join_metadata(df_pred, batch_indices)
        df_pred.to_csv(os.path.join(self.output_dir, f"df_pred_{trainer.global_rank}.csv"), index=False)