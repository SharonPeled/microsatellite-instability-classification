from src.configs import Configs
from src.components.objects.Logger import Logger
import pandas as pd
import os
from src.training_utils import init_training_transforms, init_training_callbacks
from src.components.models.SubtypeClassifier import SubtypeClassifier
from torch.multiprocessing import set_start_method, set_sharing_strategy
from src.training_utils import train as train_general


def train():
    df, train_transform, test_transform, logger, callbacks, model = init_task()
    train_general(df, train_transform, test_transform, logger, callbacks, model)


def load_df_labels_merged_tiles():
    df_labels = pd.read_csv(Configs.SC_LABEL_DF_PATH, sep='\t')
    if 'MSS' in Configs.SC_CLASS_TO_IND.keys():
        df_labels.subtype = df_labels.subtype.apply(lambda s: 'MSI' if s=='MSI' else 'MSS')
    if 'POLE' in Configs.SC_CLASS_TO_IND.keys():
        df_labels.subtype = df_labels.subtype.apply(lambda s: 'POLE' if s=='POLE' else 'NOT_POLE')
    df_labels = df_labels[df_labels[Configs.SC_LABEL_COL].isin(Configs.SC_CLASS_TO_IND.keys())]
    df_labels['slide_uuid'] = df_labels.slide_path.apply(lambda p: os.path.basename(os.path.dirname(p)))
    df_labels['y'] = df_labels[Configs.SC_LABEL_COL].apply(lambda s: Configs.SC_CLASS_TO_IND[s])
    df_labels.cohort = df_labels.cohort.apply(lambda c: c if c not in ['COAD', 'READ'] else 'CRC')
    df_labels[Configs.joined['Y_TO_BE_STRATIFIED']] = df_labels['y'].astype(str) + '_' + df_labels['cohort']
    df_labels = df_labels[df_labels.cohort.isin(Configs.SC_COHORT_TO_IND.keys())]
    # merging labels and tiles
    df_tiles = pd.read_csv(Configs.SC_DF_TILE_PATHS_PATH)
    # df_tiles = df_tiles.groupby('slide_uuid', as_index=False).apply(lambda d: d.sample(min(len(d), 10)))
    # df_tiles = df_tiles[df_tiles.slide_uuid.isin(df_tiles.slide_uuid.unique()[:10])]
    df_labels_merged_tiles = df_labels.merge(df_tiles, how='inner', on='slide_uuid')
    return df_labels, df_labels_merged_tiles


def init_task():
    df_labels_merged_tiles = init_data()
    model = init_model()
    train_transform, test_transform = init_training_transforms()
    logger, callbacks = init_training_callbacks()
    return df_labels_merged_tiles, train_transform, test_transform, logger, callbacks, model


def init_data():
    set_sharing_strategy('file_system')
    set_start_method("spawn")

    Logger.log("Loading Datasets..", log_importance=1)
    df_labels, df_labels_merged_tiles = load_df_labels_merged_tiles()

    if Configs.SC_KW_ARGS.get('FoVs_augs_amounts', None) and\
            any(fovs_rate > 0 for fovs_rate in Configs.SC_KW_ARGS['FoVs_augs_amounts']):
        tile_df_paths = [path for path in [Configs.SC_DF_TILE_PATHS_PATH_256,
                                           Configs.SC_DF_TILE_PATHS_PATH_512,
                                           Configs.SC_DF_TILE_PATHS_PATH_1024]
                         if path != Configs.SC_DF_TILE_PATHS_PATH]
        df_list = []
        for i, path in enumerate(tile_df_paths):
            if Configs.SC_KW_ARGS.get('FoVs_augs_amounts')[i] is None or Configs.SC_KW_ARGS.get('FoVs_augs_amounts')[i] == 0:
                continue
            df_tiles_aug = pd.read_csv(path)
            num_tiles = Configs.SC_KW_ARGS.get('FoVs_augs_amounts')[i] * len(df_labels_merged_tiles)
            num_tiles_per_slide = int(num_tiles / len(df_labels))
            df_tiles_aug_sampled = df_tiles_aug.groupby('slide_uuid', as_index=False).apply(
                lambda slide_df: slide_df.sample(min(num_tiles_per_slide, len(slide_df)),
                                                 random_state=Configs.RANDOM_SEED))
            df_tiles_aug_sampled = df_tiles_aug_sampled[df_tiles_aug_sampled.slide_uuid.isin(df_labels.slide_uuid)]
            df_list.append(df_tiles_aug_sampled)
            Logger.log(f"Number of aug tiles {i}: {len(df_tiles_aug_sampled)}, {num_tiles_per_slide} per slide.",
                       log_importance=True)
        df_tiles_aug_sampled = pd.concat(df_list)
        df_labels_merged_aug_tiles = df_labels.merge(df_tiles_aug_sampled, how='inner', on='slide_uuid')
        df_labels_merged_tiles = pd.concat([df_labels_merged_tiles, df_labels_merged_aug_tiles])

    if 'cohort' not in df_labels_merged_tiles.columns and 'cohort_x' in df_labels_merged_tiles.columns:
        df_labels_merged_tiles['cohort'] = df_labels_merged_tiles['cohort_x']
    if 'patient_id' not in df_labels_merged_tiles.columns and 'patient_id_x' in df_labels_merged_tiles.columns:
        df_labels_merged_tiles['patient_id'] = df_labels_merged_tiles['patient_id_x']

    return df_labels_merged_tiles


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




