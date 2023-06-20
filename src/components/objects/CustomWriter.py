from pytorch_lightning.callbacks import BasePredictionWriter
import torch
import numpy as np
import pandas as pd
import os
from datetime import datetime
from src.components.objects.Logger import Logger


class CustomWriter(BasePredictionWriter):
    def __init__(self, output_dir, write_interval, score_names, dataset):
        super().__init__(write_interval)
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.score_names = score_names
        self.dataset = dataset

    def write_on_batch_end(self, trainer, pl_module, prediction, batch_indices, batch, batch_idx, dataloader_idx):
        pass

    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
        batch_indices = np.concatenate(batch_indices[0])
        predictions = torch.concat(predictions[0])
        df_pred = pd.DataFrame(data=predictions, columns=self.score_names)
        df_pred['dataset_ind'] = batch_indices
        df_pred = self.dataset.join_metadata(df_pred, batch_indices)
        time_str = datetime.now().strftime('%d_%m_%Y_%H_%M')
        df_pred_path = os.path.join(self.output_dir, f"df_pred_{trainer.global_rank}_{time_str}.csv")
        df_pred.to_csv(df_pred_path, index=False)
        Logger.log(f"""CustomWriter df_pred saved: {df_pred_path}.""", log_importance=1)
