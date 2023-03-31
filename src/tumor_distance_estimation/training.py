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


def set_worker_sharing_strategy(worker_id: int) -> None:
    set_sharing_strategy('file_system')


def train():
    set_sharing_strategy('file_system')
    set_start_method("spawn")

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),  # reverse 50% of images
        transforms.RandomVerticalFlip(),  # reverse 50% of images
        transforms.Resize(224),
        transforms.ToTensor(),
        MacenkoNormalizerTransform(Configs.COLOR_NORM_REF_IMG),  # gets tensor and output PIL ...
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        MacenkoNormalizerTransform(Configs.COLOR_NORM_REF_IMG),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    df_full = pd.read_csv(Configs.TR_LABEL_DF_PATH)
    df_train, df_test = train_test_split(df_full, test_size=Configs.TR_TEST_SIZE, random_state=Configs.RANDOM_SEED)
    df_train, df_valid = train_test_split(df_train, test_size=Configs.TR_VALID_SIZE, random_state=Configs.RANDOM_SEED)

    train_dataset = TumorRegressionDataset(df_train, transform=train_transform)
    valid_dataset = TumorRegressionDataset(df_valid, transform=test_transform)
    test_dataset = TumorRegressionDataset(df_test, transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=Configs.SS_TRAINING_BATCH_SIZE, shuffle=True,
                              persistent_workers=True, num_workers=Configs.SS_TRAINING_NUM_WORKERS,
                              worker_init_fn=set_worker_sharing_strategy)
    valid_loader = DataLoader(valid_dataset, batch_size=Configs.SS_TRAINING_BATCH_SIZE, shuffle=False,
                              persistent_workers=True, num_workers=Configs.SS_TRAINING_NUM_WORKERS,
                              worker_init_fn=set_worker_sharing_strategy)
    test_loader = DataLoader(test_dataset, batch_size=Configs.SS_TRAINING_BATCH_SIZE, shuffle=False,
                             persistent_workers=True, num_workers=Configs.SS_TRAINING_NUM_WORKERS,
                             worker_init_fn=set_worker_sharing_strategy)

    model = TumorRegressor(learning_rate=Configs.TR_INIT_LR)
    mlflow_logger = MLFlowLogger(experiment_name=Configs.TR_EXPERIMENT_NAME, run_name=Configs.TR_RUN_NAME,
                                 save_dir=Configs.MLFLOW_SAVE_DIR,
                                 artifact_location=Configs.MLFLOW_SAVE_DIR,
                                 log_model='all',
                                 tags={"mlflow.note.content": Configs.TR_RUN_DESCRIPTION})
    Logger.log("Starting Training.", log_importance=1)
    trainer = pl.Trainer(devices=Configs.TR_NUM_DEVICES, accelerator=Configs.TR_DEVICE,
                         deterministic=True,
                         check_val_every_n_epoch=1,
                         enable_checkpointing=True,
                         logger=mlflow_logger,
                         num_sanity_val_steps=2,
                         max_epochs=Configs.TR_NUM_EPOCHS)
    trainer.fit(model, train_loader, valid_loader, ckpt_path=None)
    Logger.log("Done Training.", log_importance=1)
    Logger.log("Starting Test.", log_importance=1)
    trainer.test(model, dataloaders=test_loader)
    Logger.log(f"Done.", log_importance=1)







