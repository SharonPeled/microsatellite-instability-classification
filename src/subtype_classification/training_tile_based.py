from torchvision import transforms
from torch.utils.data import DataLoader
from src.components.datasets.ProcessedTileDataset import ProcessedTileDataset
import pytorch_lightning as pl
from torch.multiprocessing import set_start_method, set_sharing_strategy
from pytorch_lightning.loggers import MLFlowLogger
from ..configs import Configs
from src.components.objects.Logger import Logger
from src.components.models.TransferLearningClassifier import TransferLearningClassifier
from src.components.objects.CheckpointEveryNSteps import CheckpointEveryNSteps
import pandas as pd
from src.general_utils import train_test_valid_split_patients_stratified, save_pred_outputs
from pytorch_lightning.callbacks import LearningRateMonitor
from src.components.models.SubtypeClassifier import SubtypeClassifier
import os
from src.components.objects.RandStainNA.randstainna import RandStainNA
from src.subtype_classification.init_task import init_task, get_loader_and_datasets


def set_worker_sharing_strategy(worker_id: int) -> None:
    set_sharing_strategy('file_system')


def train_single_split(df_train, df_valid, df_test, train_transform, test_transform, logger, callbacks=()):
    train_dataset, valid_dataset, test_dataset, train_loader, valid_loader, test_loader = get_loader_and_datasets(df_train,
                                                                    df_valid, df_test, train_transform, test_transform)
    # after split and loaders
    if Configs.SC_TEST_ONLY is None:
        model = SubtypeClassifier(tile_encoder_name=Configs.SC_TILE_ENCODER, class_to_ind=Configs.SC_CLASS_TO_IND,
                                  learning_rate=Configs.SC_INIT_LR, frozen_backbone=Configs.SC_FROZEN_BACKBONE,
                                  class_to_weight=Configs.SC_CLASS_WEIGHT,
                                  num_iters_warmup_wo_backbone=Configs.SC_ITER_TRAINING_WARMUP_WO_BACKBONE,
                                  cohort_to_ind=Configs.SC_COHORT_TO_IND, cohort_weight=Configs.SC_COHORT_WEIGHT)
    else:
        model = SubtypeClassifier.load_from_checkpoint(Configs.SC_TEST_ONLY, tile_encoder_name=Configs.SC_TILE_ENCODER,
                                                       class_to_ind=Configs.SC_CLASS_TO_IND,
                                                       learning_rate=Configs.SC_INIT_LR,
                                                       frozen_backbone=Configs.SC_FROZEN_BACKBONE,
                                                       class_to_weight=Configs.SC_CLASS_WEIGHT,
                                                       num_iters_warmup_wo_backbone=Configs.SC_ITER_TRAINING_WARMUP_WO_BACKBONE,
                                                       cohort_to_ind=Configs.SC_COHORT_TO_IND,
                                                       cohort_weight=Configs.SC_COHORT_WEIGHT)
    Logger.log("Starting Training.", log_importance=1)
    trainer = pl.Trainer(devices=Configs.SC_NUM_DEVICES, accelerator=Configs.SC_DEVICE,
                         deterministic=True,
                         val_check_interval=Configs.SC_VAL_STEP_INTERVAL,
                         callbacks=callbacks,
                         enable_checkpointing=True,
                         logger=logger,
                         num_sanity_val_steps=2,
                         max_epochs=Configs.SC_NUM_EPOCHS)
    if Configs.SC_TEST_ONLY is None:
        if valid_loader is None:
            trainer.fit(model, train_loader, ckpt_path=None)
            trainer.save_checkpoint(Configs.SC_TRAINED_MODEL_PATH)
        else:
            trainer.fit(model, train_loader, valid_loader, ckpt_path=None)
            trainer.save_checkpoint(Configs.SC_TRAINED_MODEL_PATH)
    Logger.log("Done Training.", log_importance=1)
    Logger.log("Starting Test.", log_importance=1)
    trainer.test(model, test_loader)
    Logger.log(f"Done Test.", log_importance=1)
    save_results(model, test_dataset, valid_dataset)
    return model


def save_results(model, test_dataset, valid_dataset):
    Logger.log(f"Saving test results...", log_importance=1)
    # since shuffle=False in test we can infer the batch_indices from batch_inx
    _, df_pred_path = save_pred_outputs(model.test_outputs, test_dataset, Configs.SC_TEST_BATCH_SIZE,
                                        save_path=Configs.SC_TEST_PREDICT_OUTPUT_PATH,
                                        class_to_ind=Configs.SC_CLASS_TO_IND)
    Logger.log(f"""Saved Test df_pred: {df_pred_path}""", log_importance=1)
    if valid_dataset is not None and model.valid_outputs is not None:
        _, df_pred_path = save_pred_outputs(model.valid_outputs, valid_dataset, Configs.SC_TEST_BATCH_SIZE,
                                            save_path=Configs.SC_VALID_PREDICT_OUTPUT_PATH,
                                            class_to_ind=Configs.SC_CLASS_TO_IND, suffix='')
        Logger.log(f"""Saved valid df_pred: {df_pred_path}""", log_importance=1)


def train():
    # TODO: test mlflow loggings
    df, train_transform, test_transform = init_task()
    mlflow_logger = MLFlowLogger(experiment_name=Configs.SC_EXPERIMENT_NAME, run_name=Configs.SC_RUN_NAME,
                                 save_dir=Configs.MLFLOW_SAVE_DIR,
                                 artifact_location=Configs.MLFLOW_SAVE_DIR,
                                 log_model='all',
                                 tags={"mlflow.note.content": Configs.SC_RUN_DESCRIPTION})
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    checkpoint_callback = CheckpointEveryNSteps(prefix=Configs.SC_RUN_NAME,
                                                save_step_frequency=Configs.SC_SAVE_CHECKPOINT_STEP_INTERVAL)
    if Configs.SC_CROSS_VALIDATE:
        cross_validate(df, train_transform, test_transform, mlflow_logger, callbacks=[lr_monitor, checkpoint_callback])
    else:
        df_train, df_valid, df_test = train_test_valid_split_patients_stratified(df,
                                                                                 y_col='y_to_be_stratified',
                                                                                 test_size=Configs.SC_TEST_SIZE,
                                                                                 valid_size=Configs.SC_VALID_SIZE,
                                                                                 random_seed=Configs.RANDOM_SEED)
        train_single_split(df_train, df_valid, df_test, train_transform, test_transform, mlflow_logger,
                           callbacks=[lr_monitor, checkpoint_callback])
    Logger.log(f"Finished.", log_importance=1)


def cross_validate(df, train_transform, test_transform, mlflow_logger, callbacks):
    split_obj = train_test_valid_split_patients_stratified(df,
                                                           y_col='y_to_be_stratified',
                                                           test_size=Configs.SC_TEST_SIZE,
                                                           valid_size=0,
                                                           random_seed=Configs.RANDOM_SEED,
                                                           return_split_obj=True)
    for i, (train_inds, test_inds) in enumerate(split_obj):
        Logger.log(f"Fold {i}", log_importance=1)
        df_train = df.iloc[train_inds].reset_index(drop=True)
        df_test = df.iloc[test_inds].reset_index(drop=True)
        # get all the scores..
        train_single_split(df_train, None, df_test, train_transform, test_transform, mlflow_logger,
                           callbacks=callbacks)






