import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.datasets import ImageFolder
import pytorch_lightning as pl
from torchvision.models import resnet50
from sklearn.model_selection import StratifiedShuffleSplit
from dataclasses import dataclass
from torch.multiprocessing import Pool, set_start_method
from pytorch_lightning.loggers import TensorBoardLogger
from torch.nn.functional import softmax
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchstain
from PIL import Image
import matplotlib.pyplot as plt
from ..configs import Configs
from ..components.MacenkoNormalizerTransform import MacenkoNormalizerTransform
from .utils import get_train_test_valid_dataset
from ..components.Logger import Logger
from .TumorClassifier import TumorClassifier


def train():
    set_start_method("spawn")
    dataset = ImageFolder(Configs.TUMOR_LABELED_TILES_DIR)
    assert dataset.class_to_idx == Configs.TUMOR_CLASS_TO_IND

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),  # reverse 50% of images
        transforms.RandomVerticalFlip(),  # reverse 50% of images
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        MacenkoNormalizerTransform(Configs.COLOR_NORM_REF_IMG),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    valid_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        MacenkoNormalizerTransform(Configs.COLOR_NORM_REF_IMG),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    train_dataset, valid_dataset, test_dataset = get_train_test_valid_dataset(dataset, Configs.TUMOR_TEST_SIZE,
                                                                              Configs.TUMOR_VALID_SIZE,
                                                                              Configs.RANDOM_SEED,
                                                                              train_transform,
                                                                              valid_transform)
    Logger.log(f"""Created tumor classification datasets: {len(train_dataset)}, {len(valid_dataset)}, {len(test_dataset)}""")

    train_loader = DataLoader(train_dataset, batch_size=Configs.TUMOR_TRAINING_BATCH_SIZE, shuffle=True,
                              num_workers=Configs.TUMOR_TRAINING_NUM_WORKERS)
    valid_loader = DataLoader(valid_dataset, batch_size=Configs.TUMOR_TRAINING_BATCH_SIZE, shuffle=False,
                              num_workers=Configs.TUMOR_TRAINING_NUM_WORKERS)
    test_loader = DataLoader(test_dataset, batch_size=Configs.TUMOR_TRAINING_BATCH_SIZE, shuffle=False,
                             num_workers=Configs.TUMOR_TRAINING_NUM_WORKERS)

    model = TumorClassifier(Configs.TUMOR_NUM_CLASSES, Configs.TUMOR_IND, Configs.TUMOR_INIT_LR)
    logger = TensorBoardLogger(Configs.TUMOR_EXPERIMENT)
    trainer = pl.Trainer(devices=Configs.TUMOR_NUM_DEVICES, accelerator=Configs.TUMOR_DEVICE,
                         deterministic=True,
                         check_val_every_n_epoch=1,
                         default_root_dir=Configs.TUMOR_LOG_DIR,
                         enable_checkpointing=True,
                         logger=logger,
                         num_sanity_val_steps=2,
                         max_epochs=Configs.TUMOR_NUM_EPOCHS)
    trainer.fit(model, train_loader, valid_loader, ckpt_path=None)
    trainer.test(model, dataloaders=test_loader)
    trainer.save_checkpoint(Configs.TUMOR_TRAINED_MODEL_PATH)











