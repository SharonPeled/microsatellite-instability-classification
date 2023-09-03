from src.configs import Configs
from src.components.objects.Logger import Logger
import pandas as pd
import os
from src.training_utils import init_training_transforms, init_training_callbacks
from src.components.models.SubtypeClassifier import SubtypeClassifier
from torch.multiprocessing import set_start_method, set_sharing_strategy


def init_task():
    set_sharing_strategy('file_system')
    set_start_method("spawn")

    logger, callbacks = init_training_callbacks()
    if Configs.SC_KW_ARGS.get('config_filepath', None):
        logger.experiment.log_artifact(logger.run_id, Configs.SC_KW_ARGS['config_filepath'],
                                       artifact_path="configs")
        Logger.log(f"""Pretraining: config file logged.""",
                   log_importance=1)

    Logger.log("Loading Datasets..", log_importance=1)
    df_tiles = pd.read_csv(Configs.DN_DF_TILE_PATHS_PATH)
    num_slides = len(df_tiles.slide_uuid.unique())

    if Configs.DINO_DICT.get('FoVs_augs_amounts', None):
        tile_df_paths = [Configs.SC_DF_TILE_PATHS_PATH_224, Configs.SC_DF_TILE_PATHS_PATH_1024]
        df_list = []
        for i, path in enumerate(tile_df_paths):
            if Configs.SC_KW_ARGS.get('FoVs_augs_amounts')[i] is None or Configs.SC_KW_ARGS.get('FoVs_augs_amounts')[i] == 0:
                continue
            df_tiles_aug = pd.read_csv(path)
            num_tiles = Configs.DINO_DICT['FoVs_augs_amounts'][i] * len(df_tiles)
            num_tiles_per_slide = int(num_tiles / num_slides)
            df_tiles_aug_sampled = df_tiles_aug.groupby('slide_uuid', as_index=False).apply(
                lambda slide_df: slide_df.sample(min(num_tiles_per_slide, len(slide_df)),
                                                 random_state=Configs.RANDOM_SEED))
            df_list.append(df_tiles_aug_sampled)
            Logger.log(f"Number of aug tiles {i}: {len(df_tiles_aug_sampled)}, {num_tiles_per_slide} per slide.",
                       log_importance=1)
        df_list.append(df_tiles)
        df_tiles = pd.concat(df_list)

    return df_tiles, logger



