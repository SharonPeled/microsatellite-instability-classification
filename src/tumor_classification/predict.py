import numpy as np
import pandas as pd

from ..components.TileDataset import TileDataset
from torch.utils.data import DataLoader
from ..configs import Configs
from pytorch_lightning.callbacks import BasePredictionWriter
import torch
import os
import pytorch_lightning as pl
from .TumorClassifier import TumorClassifier
from torchvision import transforms


class CustomWriter(BasePredictionWriter):
    def __init__(self, output_dir, write_interval, class_to_index, dataset):
        super().__init__(write_interval)
        self.output_dir = output_dir
        self.class_to_index = class_to_index
        self.score_names = sorted(class_to_index.keys(), key=lambda k: class_to_index[k])
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


def predict():
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        # MacenkoNormalizerTransform(Configs.COLOR_NORM_REF_IMG),  # already norm
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    dataset = TileDataset(Configs.PROCESSED_TILES_DIR, transform=transform)
    dataloader = DataLoader(dataset, shuffle=False, batch_size=Configs.TUMOR_INFERENCE_BATCH_SIZE,
                            num_workers=Configs.TUMOR_INFERENCE_NUM_WORKERS)
    model = TumorClassifier.load_from_checkpoint(Configs.TUMOR_TRAINED_MODEL_PATH)
    pred_writer = CustomWriter(output_dir=Configs.TUMOR_PREDICT_OUTPUT_PATH,
                               write_interval="epoch", class_to_index=Configs.TUMOR_CLASS_TO_IND, dataset=dataset)
    trainer = pl.Trainer(accelerator=Configs.TUMOR_DEVICE, devices=Configs.TUMOR_NUM_DEVICES, callbacks=[pred_writer],
                         default_root_dir=Configs.TUMOR_PREDICT_OUTPUT_PATH)
    trainer.predict(model, dataloader, return_predictions=False)




