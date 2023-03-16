import numpy as np
import pandas as pd
from ..components.TileDataset import TileDataset
from torch.utils.data import DataLoader
from ..configs import Configs
from pytorch_lightning.callbacks import BasePredictionWriter
import torch
import os
import pytorch_lightning as pl
from ..components.TissueClassifier import TissueClassifier
from torchvision import transforms
from torch.utils.data.dataloader import default_collate


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


def my_collate(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return default_collate(batch)


def predict():
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(), # tiles are already macenko normalized
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    dataset = TileDataset(Configs.PROCESSED_TILES_DIR, transform=transform)
    dataloader = DataLoader(dataset, shuffle=False, batch_size=Configs.SS_INFERENCE_BATCH_SIZE,
                            num_workers=Configs.SS_INFERENCE_NUM_WORKERS)
                            # collate_fn=my_collate)
    model = TissueClassifier.load_from_checkpoint(Configs.SS_TRAINED_MODEL_PATH,
                                                  class_to_ind=Configs.SS_CLASS_TO_IND, learning_rate=None)
    pred_writer = CustomWriter(output_dir=Configs.SS_PREDICT_OUTPUT_PATH,
                               write_interval="epoch", class_to_index=Configs.SS_CLASS_TO_IND, dataset=dataset)
    trainer = pl.Trainer(accelerator=Configs.SS_DEVICE, devices=Configs.SS_NUM_DEVICES, callbacks=[pred_writer],
                         default_root_dir=Configs.SS_PREDICT_OUTPUT_PATH)
    trainer.predict(model, dataloader, return_predictions=False)




