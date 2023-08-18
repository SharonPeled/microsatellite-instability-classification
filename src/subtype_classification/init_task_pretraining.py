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

    train_transform, test_transform = init_training_transforms()

    logger, callbacks = init_training_callbacks()

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

    model = init_model()

    return df_tiles, train_transform, test_transform, logger, callbacks, model


def init_model():
    if Configs.SC_TEST_ONLY is None:
        model = SubtypeClassifier(tile_encoder_name=Configs.SC_TILE_ENCODER, class_to_ind=Configs.SC_CLASS_TO_IND,
                                  learning_rate=Configs.SC_INIT_LR, frozen_backbone=Configs.SC_FROZEN_BACKBONE,
                                  class_to_weight=Configs.SC_CLASS_WEIGHT,
                                  num_iters_warmup_wo_backbone=Configs.SC_ITER_TRAINING_WARMUP_WO_BACKBONE,
                                  cohort_to_ind=Configs.SC_COHORT_TO_IND, cohort_weight=Configs.SC_COHORT_WEIGHT,
                                  **Configs.SC_KW_ARGS)
        Logger.log(f"New Model successfully created!", log_importance=1)
    else:
        model = SubtypeClassifier.load_from_checkpoint(Configs.SC_TEST_ONLY, tile_encoder_name=Configs.SC_TILE_ENCODER,
                                                       class_to_ind=Configs.SC_CLASS_TO_IND,
                                                       learning_rate=Configs.SC_INIT_LR,
                                                       frozen_backbone=Configs.SC_FROZEN_BACKBONE,
                                                       class_to_weight=Configs.SC_CLASS_WEIGHT,
                                                       num_iters_warmup_wo_backbone=Configs.SC_ITER_TRAINING_WARMUP_WO_BACKBONE,
                                                       cohort_to_ind=Configs.SC_COHORT_TO_IND,
                                                       cohort_weight=Configs.SC_COHORT_WEIGHT,
                                                       **Configs.SC_KW_ARGS)
        Logger.log(f"Model successfully loaded from checkpoint!", log_importance=1)
    return model




