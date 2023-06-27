from torchvision import transforms
from torch.utils.data import DataLoader
from src.components.datasets.ProcessedTileDataset import ProcessedTileDataset
import pytorch_lightning as pl
from torch.multiprocessing import set_start_method, set_sharing_strategy
from pytorch_lightning.loggers import MLFlowLogger
from ..configs import Configs
from src.components.objects.Logger import Logger
from src.components.models.TransferLearningClassifier import TransferLearningClassifier
from src.components.models.MIL_VIT import MIL_VIT
from src.components.objects.CheckpointEveryNSteps import CheckpointEveryNSteps
import pandas as pd
from src.utils import train_test_valid_split_patients_stratified, save_pred_outputs, modify_model_for_transfer_learning
from pytorch_lightning.callbacks import LearningRateMonitor
import os


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
    Logger.log("Loading Datasets..", log_importance=1)
    df_labels = pd.read_csv(Configs.SC_LABEL_DF_PATH, sep='\t')
    df_labels = df_labels[df_labels[Configs.SC_LABEL_COL].isin(Configs.SC_CLASS_TO_IND.keys())]
    df_labels['slide_uuid'] = df_labels.slide_path.apply(lambda p: os.path.basename(os.path.dirname(p)))
    df_labels['y'] = df_labels[Configs.SC_LABEL_COL].apply(lambda s: Configs.SC_CLASS_TO_IND[s])
    df_labels['y_to_be_stratified'] = df_labels['y'].astype(str) + '_' + df_labels['cohort']
    # merging labels and tiles
    df_tiles = pd.read_csv(Configs.SC_DF_TILE_PATHS_PATH)
    df_labels_merged_tiles = df_labels.merge(df_tiles, how='inner', on='slide_uuid')
    # sampling from each slide to reduce computational costs
    df_labels_merged_tiles_sampled = df_labels_merged_tiles.groupby('slide_uuid').apply(
        lambda slide_df: slide_df.sample(n=Configs.SC_TILE_SAMPLE_LAMBDA_TRAIN(len(slide_df)),
                                         random_state=Configs.RANDOM_SEED)).reset_index(drop=True)
    # split to train, valid, test
    df_train, df_valid, df_test = train_test_valid_split_patients_stratified(df_labels_merged_tiles_sampled,
                                                                             y_col='y_to_be_stratified',
                                                                             test_size=Configs.SC_TEST_SIZE,
                                                                             valid_size=Configs.SC_VALID_SIZE,
                                                                             random_seed=Configs.RANDOM_SEED)

    # t = pd.read_csv('/home/sharonpe/microsatellite-instability-classification/data/subtype_classification/resnet_tile_CIS_GS_2/test/df_pred_21_06_2023_05_41.csv')
    # df_test.tile_path.isin(t.tile_path).all() and t.tile_path.isin(df_test.tile_path).all()  # no filtered should be applied. should be exactly the same.
    train_dataset = ProcessedTileDataset(df_labels=df_train, transform=train_transform,
                                         group_size=Configs.SC_MIL_GROUP_SIZE)
    valid_dataset = ProcessedTileDataset(df_labels=df_valid, transform=test_transform,
                                         group_size=Configs.SC_MIL_GROUP_SIZE)
    test_dataset = ProcessedTileDataset(df_labels=df_test, transform=test_transform,
                                        group_size=Configs.SC_MIL_GROUP_SIZE)

    valid_loader = DataLoader(valid_dataset, batch_size=Configs.SC_TEST_BATCH_SIZE, shuffle=False,
                              persistent_workers=True, num_workers=Configs.SC_NUM_WORKERS,
                              worker_init_fn=set_worker_sharing_strategy)
    test_loader = DataLoader(test_dataset, batch_size=Configs.SC_TEST_BATCH_SIZE, shuffle=False,
                             persistent_workers=True, num_workers=Configs.SC_NUM_WORKERS,
                             worker_init_fn=set_worker_sharing_strategy)
    tile_encoder = TransferLearningClassifier.load_from_checkpoint(Configs.SC_TILE_BASED_TRAINED_MODEL,
                                                                   class_to_ind=Configs.SC_CLASS_TO_IND,
                                                                   learning_rate=None)
    # tile_encoder = modify_model_for_transfer_learning(tile_encoder, num_classes=None, freezing_backbone=True)
    model = MIL_VIT(class_to_ind=Configs.SC_CLASS_TO_IND, learning_rate=Configs.SC_INIT_LR,
                    tile_encoder=tile_encoder,
                    vit_variant=Configs.SC_MIL_VIT_MODEL_VARIANT, pretrained=Configs.SC_MIL_VIT_MODEL_PRETRAINED)
    steps_done = 0
    for phase_config in Configs.SC_TRAINING_PHASES:
        num_steps = phase_config['num_steps']
        if num_steps == 0:
            continue
        train_dataset.deploy_dataset_limits(dataset_limits=(steps_done, num_steps))
        train_loader = DataLoader(train_dataset, batch_size=Configs.SC_TRAINING_BATCH_SIZE, shuffle=True,
                                  persistent_workers=True, num_workers=Configs.SC_NUM_WORKERS,
                                  worker_init_fn=set_worker_sharing_strategy)
        model.next_training_phase()
        model.learning_rate = phase_config['lr']
        mlflow_logger = MLFlowLogger(experiment_name=Configs.SC_EXPERIMENT_NAME, run_name=Configs.SC_RUN_NAME + phase_config['run_suffix'],
                                     save_dir=Configs.MLFLOW_SAVE_DIR,
                                     artifact_location=Configs.MLFLOW_SAVE_DIR,
                                     log_model='all',
                                     tags={"mlflow.note.content": Configs.SC_RUN_DESCRIPTION})
        lr_monitor = LearningRateMonitor(logging_interval='epoch')
        Logger.log("Starting Training.", log_importance=1)
        trainer = pl.Trainer(devices=Configs.SC_NUM_DEVICES, accelerator=Configs.SC_DEVICE,
                             deterministic=True,
                             val_check_interval=Configs.SC_VAL_STEP_INTERVAL,
                             callbacks=[lr_monitor, CheckpointEveryNSteps(
                                 prefix=Configs.SC_RUN_NAME + phase_config['run_suffix'],
                                 save_step_frequency=Configs.SC_SAVE_CHECKPOINT_STEP_INTERVAL)],
                             enable_checkpointing=True,
                             logger=mlflow_logger,
                             num_sanity_val_steps=2,
                             max_epochs=1)
        trainer.fit(model, train_loader, valid_loader, ckpt_path=None)
        steps_done += num_steps
    trainer.save_checkpoint(Configs.SC_TRAINED_MODEL_PATH)
    Logger.log("Done Training.", log_importance=1)
    Logger.log("Starting Test.", log_importance=1)
    trainer.test(model, test_loader)
    Logger.log(f"Done Test.", log_importance=1)
    Logger.log(f"Saving test results...", log_importance=1)
    # since shuffle=False in test we can infer the batch_indices from batch_inx
    _, df_pred_path = save_pred_outputs(model.test_outputs, test_dataset, Configs.SC_TEST_BATCH_SIZE,
                                        save_path=Configs.SC_TEST_PREDICT_OUTPUT_PATH,
                                        class_to_ind=Configs.SC_CLASS_TO_IND)
    Logger.log(f"""Saved Test df_pred: {df_pred_path}""", log_importance=1)
    trainer.validate(model, valid_loader)
    for i, outputs in enumerate(model.valid_outputs):
        _, df_pred_path = save_pred_outputs(outputs, valid_dataset, Configs.SC_TEST_BATCH_SIZE,
                                            save_path=Configs.SC_VALID_PREDICT_OUTPUT_PATH,
                                            class_to_ind=Configs.SC_CLASS_TO_IND, suffix=str(i))
        Logger.log(f"""Saved valid {i} df_pred: {df_pred_path}""", log_importance=1)
    Logger.log(f"""Saving Done, df_pred saved in: {df_pred_path}""", log_importance=1)
    Logger.log(f"Finished.", log_importance=1)







