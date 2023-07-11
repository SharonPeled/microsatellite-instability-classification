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


def set_worker_sharing_strategy(worker_id: int) -> None:
    set_sharing_strategy('file_system')


def init_task():
    set_sharing_strategy('file_system')
    set_start_method("spawn")
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),  # reverse 50% of images
        transforms.RandomVerticalFlip(),  # reverse 50% of images

        transforms.RandomApply([transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.25, 1)),
                                transforms.RandomAdjustSharpness(sharpness_factor=2)], p=0.1),

        transforms.RandomApply([transforms.Grayscale(num_output_channels=3), ], p=0.2),  # grayscale 20% of the images
        transforms.RandomApply([transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)], p=0.8),

        transforms.RandomApply([
            RandStainNA(yaml_file=Configs.SSL_STATISTICS['HSV'], std_hyper=0.01, probability=1.0, distribution="normal",
                        is_train=True),
            RandStainNA(yaml_file=Configs.SSL_STATISTICS['HED'], std_hyper=0.01, probability=1.0, distribution="normal",
                        is_train=True),
            RandStainNA(yaml_file=Configs.SSL_STATISTICS['LAB'], std_hyper=0.01, probability=1.0, distribution="normal",
                        is_train=True)],
            p=0.8),

        transforms.Resize(224),
        transforms.ToTensor(),

        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
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
                                         random_state=Configs.RANDOM_SEED))
    return df_labels_merged_tiles_sampled, train_transform, test_transform


def get_loader_and_datasets(df_train, df_valid, df_test, train_transform, test_transform):
    train_dataset = ProcessedTileDataset(df_labels=df_train, transform=train_transform,
                                         cohort_to_index=Configs.SC_COHORT_TO_IND)
    test_dataset = ProcessedTileDataset(df_labels=df_test, transform=test_transform,
                                        cohort_to_index=Configs.SC_COHORT_TO_IND)

    train_loader = DataLoader(train_dataset, batch_size=Configs.SC_TRAINING_BATCH_SIZE,
                              shuffle=True,
                              persistent_workers=True, num_workers=Configs.SC_NUM_WORKERS,
                              worker_init_fn=set_worker_sharing_strategy)
    test_loader = DataLoader(test_dataset, batch_size=Configs.SC_TEST_BATCH_SIZE, shuffle=False,
                             persistent_workers=True, num_workers=Configs.SC_NUM_WORKERS,
                             worker_init_fn=set_worker_sharing_strategy)
    if df_valid is None:
        return train_dataset, None, test_dataset, train_loader, None, test_loader

    valid_dataset = ProcessedTileDataset(df_labels=df_valid, transform=test_transform,
                                         cohort_to_index=Configs.SC_COHORT_TO_IND)
    valid_loader = DataLoader(valid_dataset, batch_size=Configs.SC_TEST_BATCH_SIZE, shuffle=False,
                              persistent_workers=True, num_workers=Configs.SC_NUM_WORKERS,
                              worker_init_fn=set_worker_sharing_strategy)

    return train_dataset, valid_dataset, test_dataset, train_loader, valid_loader, test_loader

