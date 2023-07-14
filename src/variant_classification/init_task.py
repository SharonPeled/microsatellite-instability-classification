from src.configs import Configs
from src.components.objects.Logger import Logger
import pandas as pd
import os
from src.training_utils import init_training_transforms, init_training_callbacks
from src.components.models.VariantClassifier import VariantClassifier
import torch


def init_task():
    train_transform, test_transform = init_training_transforms()

    logger, callbacks = init_training_callbacks()

    Logger.log("Loading Datasets..", log_importance=1)
    df_labels = pd.read_csv(Configs.VC_LABEL_DF_PATH)
    df_labels.rename(columns={'GT_array': 'y'}, inplace=True)
    df_labels.y = df_labels.y.apply(lambda a: torch.Tensor(eval(a)).long())
    df_labels['cohort'] = 'CRC'
    # loading tile filepaths
    df_tiles = pd.read_csv(Configs.VC_DF_TILE_PATHS_PATH)
    df_labels_merged_tiles = df_labels.merge(df_tiles, how='inner', on='patient_id')
    df_labels_merged_tiles['slide_id'] = df_labels_merged_tiles.slide_uuid

    num_snps = len(df_labels.y.iloc[0])
    model = init_model(num_snps)

    return df_labels_merged_tiles, train_transform, test_transform, logger, callbacks, model


def init_model(num_snps):
    if Configs.VC_TEST_ONLY is None:
        model = VariantClassifier(output_shape=(3, num_snps), tile_encoder_name=Configs.VC_TILE_ENCODER, class_to_ind=Configs.VC_CLASS_TO_IND,
                                  learning_rate=Configs.VC_INIT_LR, frozen_backbone=Configs.VC_FROZEN_BACKBONE,
                                  num_iters_warmup_wo_backbone=Configs.VC_ITER_TRAINING_WARMUP_WO_BACKBONE)
    else:
        model = VariantClassifier.load_from_checkpoint(Configs.VC_TEST_ONLY, output_shape=(3, num_snps),
                                                       tile_encoder_name=Configs.VC_TILE_ENCODER,
                                                       class_to_ind=Configs.VC_CLASS_TO_IND,
                                                       learning_rate=Configs.VC_INIT_LR,
                                                       frozen_backbone=Configs.VC_FROZEN_BACKBONE,
                                                       num_iters_warmup_wo_backbone=Configs.VC_ITER_TRAINING_WARMUP_WO_BACKBONE)
    return model




