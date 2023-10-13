from src.configs import Configs
from src.components.objects.Logger import Logger
import pandas as pd
import os
from src.training_utils import init_training_transforms, init_training_callbacks
from torch.multiprocessing import set_start_method, set_sharing_strategy
from src.subtype_classification.init_task_generic import load_df_labels_merged_tiles
from src.components.models.MIL_Fusion_VIT import MIL_Fusion_VIT
from src.training_utils import train as train_general
from src.components.datasets.TileGroupDataset import TileGroupDataset


def train():
    df, train_transform, test_transform, logger, callbacks, model = init_task()
    train_general(df, train_transform, test_transform, logger, callbacks, model, dataset_fn=TileGroupDataset,
                  custom_collate_fn=custom_collate_fn)


def init_task():
    set_sharing_strategy('file_system')
    set_start_method("spawn")

    train_transform, test_transform = init_training_transforms()

    logger, callbacks = init_training_callbacks()

    Logger.log("Loading Datasets..", log_importance=1)
    df_labels, df_labels_merged_tiles = load_df_labels_merged_tiles()
    df_labels_merged_tiles['tile_row'] = df_labels_merged_tiles.tile_path.apply(lambda p:
                                                                                int(os.path.basename(p).split('_')[0]))
    df_labels_merged_tiles['tile_col'] = df_labels_merged_tiles.tile_path.apply(lambda p:
                                                                                int(os.path.basename(p).split('_')[1]))
    if 'cohort' not in df_labels_merged_tiles.columns and 'cohort_x' in df_labels_merged_tiles.columns:
        df_labels_merged_tiles['cohort'] = df_labels_merged_tiles['cohort_x']

    model = init_model(dataloader_df_cols=df_labels_merged_tiles.columns, train_transform=train_transform,
                       test_transform=test_transform)

    return df_labels_merged_tiles, None, None, logger, callbacks, model


def init_model(dataloader_df_cols, train_transform, test_transform):
    tile_encoder_inference_params = {'batch_size': Configs.SC_MIL_TILE_INFERENCE_BATCH_SIZE,
                                     'num_workers': Configs.SC_MIL_TILE_INFERENCE_NUM_WORKERS,
                                     'train_transform': train_transform,
                                     'test_transform': test_transform}
    model = MIL_Fusion_VIT(dataloader_df_cols=dataloader_df_cols,
                           class_to_ind=Configs.SC_CLASS_TO_IND,
                           cohort_to_ind=Configs.SC_COHORT_TO_IND,
                           mil_model_params=(Configs.SC_MIL_MODEL_NAME, Configs.SC_MIL_MODEL_CKPT),
                           tile_encoder_params=(Configs.SC_MIL_TILE_ENCODER_NAME, Configs.SC_MIL_TILE_ENCODER_CKPT),
                           learning_rate_params=Configs.SC_MIL_LR_DICT,
                           tile_encoder_inference_params=tile_encoder_inference_params,
                           mil_pooling_strategy=Configs.SC_MIL_POOLING_STRATEGY,
                           max_tiles_mil=Configs.SC_MIL_MAX_TILES,
                           num_iters_warmup_wo_backbone=None,
                           **Configs.SC_KW_ARGS)
    Logger.log(f"New Model successfully created!", log_importance=1)
    return model


def custom_collate_fn(batch):
    num_tiles_per_slide = [len(df_s) for df_s in batch]
    return pd.concat(batch, ignore_index=True), num_tiles_per_slide


