from torchvision import transforms
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from torch.multiprocessing import Pool, set_start_method, set_sharing_strategy
from pytorch_lightning.loggers import TensorBoardLogger, MLFlowLogger
from ..configs import Configs
from ..components.Logger import Logger
from ..components.TumorRegressor import TumorRegressor
from ..components.MacenkoNormalizerTransform import MacenkoNormalizerTransform
from ..components.TumorRegressionDataset import TumorRegressionDataset
import pandas as pd
from sklearn.model_selection import train_test_split
from ..components.CustomWriter import CustomWriter
from pytorch_lightning.callbacks import LearningRateMonitor
from ..utils import train_test_valid_split_patients_stratified
from collections import defaultdict
import numpy as np


def set_worker_sharing_strategy(worker_id: int) -> None:
    set_sharing_strategy('file_system')


def train():
    set_sharing_strategy('file_system')
    set_start_method("spawn")

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),  # reverse 50% of images
        transforms.RandomVerticalFlip(),  # reverse 50% of images

        transforms.RandomApply([transforms.ColorJitter(brightness=0.1, contrast=0.4, saturation=0.4, hue=0.1)], p=0.75),
        transforms.RandomChoice([transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.25, 1)),
                                 transforms.RandomAdjustSharpness(sharpness_factor=2)], p=[0.15, 0.15]),
        transforms.Resize(224),
        transforms.ToTensor(),
        # MacenkoNormalizerTransform(Configs.COLOR_NORM_REF_IMG),  # gets tensor and output PIL ...
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        # MacenkoNormalizerTransform(Configs.COLOR_NORM_REF_IMG),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    df_full = pd.read_csv(Configs.TR_LABEL_DF_PATH)
    df_full = df_full[(df_full.dis_to_tum >= Configs.TR_MIN_DIS_TO_TUM)&(df_full.group_size > Configs.TR_MIN_GROUP_SIZE)]
    df_full.dis_to_tum = np.log1p(df_full.dis_to_tum)
    df_train, df_valid, df_test = train_test_valid_split_patients_stratified(df_full, Configs.TR_TEST_SIZE,
                                                                             Configs.TR_VALID_SIZE, Configs.RANDOM_SEED)

    train_dataset = TumorRegressionDataset(df_train, transform=train_transform)
    valid_dataset = TumorRegressionDataset(df_valid, transform=test_transform)
    test_dataset = TumorRegressionDataset(df_test, transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=Configs.TR_TRAINING_BATCH_SIZE, shuffle=True,
                              persistent_workers=True, num_workers=Configs.TR_TRAINING_NUM_WORKERS,
                              worker_init_fn=set_worker_sharing_strategy)
    valid_loader = DataLoader(valid_dataset, batch_size=Configs.TR_TRAINING_BATCH_SIZE, shuffle=False,
                              persistent_workers=True, num_workers=Configs.TR_TRAINING_NUM_WORKERS,
                              worker_init_fn=set_worker_sharing_strategy)
    test_loader = DataLoader(test_dataset, batch_size=Configs.TR_TRAINING_BATCH_SIZE, shuffle=False,
                             persistent_workers=True, num_workers=Configs.TR_TRAINING_NUM_WORKERS,
                             worker_init_fn=set_worker_sharing_strategy)
    if Configs.TR_SAMPLE_WEIGHT:
        y_value_counts = df_train.dis_to_tum.round(1).value_counts()
        y_percentages_dict = (1 / y_value_counts).to_dict()
        class_weight_dict = defaultdict(lambda: 0.1)  # default value
        class_weight_dict.update(y_percentages_dict)
    else:
        class_weight_dict = None

    model = TumorRegressor(Configs.TR_INIT_LR, class_weight_dict, Configs.TR_DROPOUT_VALUE)
    mlflow_logger = MLFlowLogger(experiment_name=Configs.TR_EXPERIMENT_NAME, run_name=Configs.TR_RUN_NAME,
                                 save_dir=Configs.MLFLOW_SAVE_DIR,
                                 artifact_location=Configs.MLFLOW_SAVE_DIR,
                                 log_model='all',
                                 tags={"mlflow.note.content": Configs.TR_RUN_DESCRIPTION})
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    Logger.log("Starting Training.", log_importance=1)
    trainer = pl.Trainer(devices=Configs.TR_NUM_DEVICES, accelerator=Configs.TR_DEVICE,
                         deterministic=True,
                         check_val_every_n_epoch=1,
                         callbacks=[lr_monitor],
                         enable_checkpointing=True,
                         logger=mlflow_logger,
                         num_sanity_val_steps=2,
                         max_epochs=Configs.TR_NUM_EPOCHS)
    trainer.fit(model, train_loader, valid_loader, ckpt_path=None)
    trainer.save_checkpoint(Configs.TR_TRAINED_MODEL_PATH)
    Logger.log("Done Training.", log_importance=1)
    Logger.log("Starting Test.", log_importance=1)
    trainer.test(model, dataloaders=test_loader)
    Logger.log(f"Done Test.", log_importance=1)
    Logger.log(f"Saving test results.", log_importance=1)
    # not nice code but whatever
    test_dataset.with_y = False
    model = TumorRegressor.load_from_checkpoint(Configs.TR_TRAINED_MODEL_PATH, learning_rate=None,
                                                class_weight_dict=None, dropout_value=Configs.TR_DROPOUT_VALUE)
    pred_writer = CustomWriter(output_dir=Configs.TR_PREDICT_OUTPUT_PATH,
                               write_interval="epoch", score_names=['y_pred', ],
                               dataset=test_dataset)
    test_loader = DataLoader(test_dataset, batch_size=Configs.TR_TRAINING_BATCH_SIZE, shuffle=False,
                             persistent_workers=True, num_workers=Configs.TR_TRAINING_NUM_WORKERS,
                             worker_init_fn=set_worker_sharing_strategy)
    trainer = pl.Trainer(accelerator=Configs.TR_DEVICE, devices=Configs.TR_NUM_DEVICES, callbacks=[pred_writer],
                         default_root_dir=Configs.TR_PREDICT_OUTPUT_PATH)
    trainer.predict(model, test_loader, return_predictions=False)








