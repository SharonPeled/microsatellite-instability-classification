import numpy as np
import pandas as pd
from ..components.MacenkoNormalizerTransform import MacenkoNormalizerTransform
from ..components.TileDataset import TileDataset
from torch.utils.data import DataLoader
from ..configs import Configs
from .ResnetEncoder import ResnetEncoder
from pytorch_lightning.callbacks import BasePredictionWriter
import os
import pytorch_lightning as pl
from torchvision import transforms
import mlflow


class CustomWriter(BasePredictionWriter):
    def __init__(self, output_dir, write_interval, dataset):
        super().__init__(write_interval)
        self.output_dir = output_dir
        self.dataset = dataset

    # TODO: decide on a batch level or epoch level
    def write_on_batch_end(self, trainer, pl_module, prediction, batch_indices, batch, batch_idx, dataloader_idx):
        """predictions are the encoded features"""
        """
        batch - Tensor: (64,3,224,224)
        batch_ids - 0
        batch_indices - list of indices [1,2,3...]
        pl_module - ResnetEncoder
        prediction - Tensor (64, 2048)
        
        """
        out_dict = {batch_indices: prediction}
        out_dict.save(os.path.join(self.output_dir,
                                   f'batch_{batch_indices[0]}_{batch_indices[-1]}_rank_{trainer.global_rank}.dict'))


def encode():
    # experiment = mlflow.get_experiment_by_name(Configs.GC_EXPERIMENT_NAME)
    # experiment_id = experiment.experiment_id if experiment is not None else \
    #     mlflow.create_experiment(Configs.GC_EXPERIMENT_NAME)
    # # encoded_features_files = glob(f"{Configs.ENCODED_FEATURES_OUTPUT_DIR}/**/batch*", recursive=True)
    # # random.shuffle(encoded_features_files)
    # # batches = np.array_split(encoded_features_files, Configs.GC_BATCH_SIZE)
    # for n_clusters in Configs.GC_NUM_CLUSTERS:
    #     with mlflow.start_run(experiment_id=experiment_id, run_name=Configs.GC_RUN_NAME + f'_{n_clusters}') as run:
    # already normalized
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    dataset = TileDataset(Configs.PROCESSED_TILES_DIR, transform=transform)
    dataloader = DataLoader(dataset, shuffle=False, batch_size=Configs.GCE_BATCH_SIZE,
                            num_workers=Configs.GCE_NUM_WORKERS)
    model = ResnetEncoder()
    feature_writer = CustomWriter(output_dir=Configs.GCE_OUTPUT_DIR,
                                  write_interval="batch", dataset=dataset)
    trainer = pl.Trainer(accelerator=Configs.GCE_DEVICE, devices=Configs.GCE_NUM_DEVICES, callbacks=[feature_writer],
                         default_root_dir=Configs.GCE_OUTPUT_DIR)
    trainer.predict(model, dataloader, return_predictions=False)




