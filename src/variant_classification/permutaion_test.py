from torchvision import transforms
from torch.utils.data import DataLoader
from src.components.datasets.ProcessedTileDataset import ProcessedTileDataset
import pytorch_lightning as pl
from torch.multiprocessing import set_start_method, set_sharing_strategy
from pytorch_lightning.loggers import MLFlowLogger
from ..configs import Configs
from src.components.objects.Logger import Logger
from src.components.models.VariantClassifier import VariantClassifier
from src.components.objects.CheckpointEveryNSteps import CheckpointEveryNSteps
import pandas as pd
from sklearn.model_selection import train_test_split
from src.utils import save_pred_outputs
from pytorch_lightning.callbacks import LearningRateMonitor
import torch


def set_worker_sharing_strategy(worker_id: int) -> None:
    set_sharing_strategy('file_system')


def training_step(df_labels, df_tiles_sampled, permutation_num, train_transform, test_transform):
    Logger.log(f"Starting permutation {permutation_num}.", log_importance=1)
    # when permutation_num=0 then it's the original labels
    if permutation_num > 0:
        df_labels.y = df_labels.y.sample(frac=1)

    # split to train, valid, test - sample_id and patient_id are 1-1
    train_samples, test_samples = train_test_split(df_labels.sample_id, test_size=Configs.VC_TEST_SIZE,
                                                   random_state=Configs.RANDOM_SEED)
    # train_samples, valid_samples = train_test_split(train_samples, test_size=Configs.VC_VALID_SIZE,
    #                                                 random_state=Configs.RANDOM_SEED)
    df_train = df_labels[df_labels.sample_id.isin(train_samples)]
    # df_valid = df_labels[df_labels.sample_id.isin(valid_samples)]
    df_test = df_labels[df_labels.sample_id.isin(test_samples)]

    # merging labels and tiles
    df_train = df_train.merge(df_tiles_sampled, how='inner', on='patient_id')
    # df_valid = df_valid.merge(df_tiles_sampled, how='inner', on='patient_id')
    df_test = df_test.merge(df_tiles_sampled, how='inner', on='patient_id')

    train_dataset = ProcessedTileDataset(df_labels=df_train, transform=train_transform)
    # valid_dataset = ProcessedTileDataset(df_labels=df_valid, transform=test_transform)
    test_dataset = ProcessedTileDataset(df_labels=df_test, transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=Configs.VC_TRAINING_BATCH_SIZE, shuffle=True,
                              persistent_workers=True, num_workers=Configs.VC_NUM_WORKERS,
                              worker_init_fn=set_worker_sharing_strategy)
    # valid_loader = DataLoader(valid_dataset, batch_size=Configs.VC_TEST_BATCH_SIZE, shuffle=False,
    #                           persistent_workers=True, num_workers=Configs.VC_NUM_WORKERS,
    #                           worker_init_fn=set_worker_sharing_strategy)
    test_loader = DataLoader(test_dataset, batch_size=Configs.VC_TEST_BATCH_SIZE, shuffle=False,
                             persistent_workers=True, num_workers=Configs.VC_NUM_WORKERS,
                             worker_init_fn=set_worker_sharing_strategy)

    num_snps = len(df_train.y.iloc[0])
    model = VariantClassifier(output_shape=(3, num_snps), learning_rate=Configs.VC_INIT_LR,
                              class_to_ind=Configs.VC_CLASS_TO_IND)
    mlflow_logger = MLFlowLogger(experiment_name=Configs.VC_EXPERIMENT_NAME, run_name=Configs.VC_RUN_NAME.format(
        permutation_num=permutation_num),
                                 save_dir=Configs.MLFLOW_SAVE_DIR,
                                 artifact_location=Configs.MLFLOW_SAVE_DIR,
                                 # log_model='all',
                                 tags={"mlflow.note.content": Configs.VC_RUN_DESCRIPTION})
    # lr_monitor = LearningRateMonitor(logging_interval='epoch')
    Logger.log("Starting Training.", log_importance=1)
    trainer = pl.Trainer(devices=Configs.VC_NUM_DEVICES, accelerator=Configs.VC_DEVICE,
                         # deterministic=True,
                         # val_check_interval=Configs.VC_VAL_STEP_INTERVAL,
                         # callbacks=[lr_monitor, CheckpointEveryNSteps(
                         #     prefix=Configs.VC_RUN_NAME,
                         #     save_step_frequency=Configs.VC_SAVE_CHECKPOINT_STEP_INTERVAL)],
                         enable_checkpointing=True,
                         logger=mlflow_logger,
                         num_sanity_val_steps=2,
                         max_epochs=Configs.VC_NUM_EPOCHS)
    # trainer.fit(model, train_loader, valid_loader, ckpt_path=None)
    trainer.fit(model, train_loader, ckpt_path=None)
    # trainer.save_checkpoint(Configs.VC_TRAINED_MODEL_PATH)
    Logger.log("Done Training.", log_importance=1)
    Logger.log("Starting Test.", log_importance=1)
    trainer.test(model, test_loader)
    Logger.log(f"Done Test.", log_importance=1)
    Logger.log(f"Saving test results...", log_importance=1)
    _, df_pred_path = save_pred_outputs(model.test_outputs, test_dataset, Configs.VC_TEST_BATCH_SIZE,
                                        save_path=Configs.VC_TEST_PREDICT_OUTPUT_PATH.format(permutation_num=permutation_num),
                                        class_to_ind=Configs.VC_CLASS_TO_IND, saving_raw=True)
    Logger.log(f"""Saved Test df_pred: {df_pred_path}""", log_importance=1)


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
    df_labels = pd.read_csv(Configs.VC_LABEL_DF_PATH)
    df_labels.rename(columns={'GT_array': 'y'}, inplace=True)
    df_labels.y = df_labels.y.apply(lambda a: torch.Tensor(eval(a))[[121, 274, 95, 159, 315]].long())
    # loading tile filepaths
    df_tiles = pd.read_csv(Configs.VC_DF_TILE_PATHS_PATH)
    # sampling from each slide to reduce computational costs
    df_tiles_sampled = df_tiles.groupby('slide_uuid').apply(lambda slide_df:
                                                            slide_df.sample(
                                                                n=Configs.VC_TILE_SAMPLE_LAMBDA_TRAIN(len(slide_df)),
                                                                random_state=Configs.RANDOM_SEED)).reset_index(
        drop=True)
    # permutation loop
    for i in range(Configs.VC_NUM_PERMUTATIONS+1):
        training_step(df_labels, df_tiles_sampled, permutation_num=i, train_transform=train_transform,
                      test_transform=test_transform)
    Logger.log(f"Finished.", log_importance=1)







