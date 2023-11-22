from src.configs import Configs
from src.components.objects.Logger import Logger
import pandas as pd
import os
from src.training_utils import init_training_transforms, init_training_callbacks
from src.components.models.SubtypeClassifier import SubtypeClassifier
from torch.multiprocessing import set_start_method, set_sharing_strategy
from src.training_utils import train as train_general
from src.training_utils import split_evenly, sample_from_df
from functools import partial


def get_paired_tiles_diff_slide_same_cohort(df_s, df_c_s_dict):
    cohort = df_s.cohort.iloc[0]
    slide_uuid = df_s.slide_uuid.iloc[0]
    slide_uuids_list, df_slide_list = df_c_s_dict[cohort]
    sample_per_s = split_evenly(len(df_s), len(df_slide_list) - 1)
    sampled_pair_dfs = []
    shift = 0
    for i in range(len(slide_uuids_list)):
        sample_i = i + shift
        if slide_uuid == slide_uuids_list[i]:
            shift = -1
            continue
        sampled_pair_dfs.append(sample_from_df(df_slide_list[i], sample_per_s[sample_i]))
    df_sampled_tiles = pd.concat(sampled_pair_dfs, ignore_index=True)
    df_s['paired_tile_0'] = df_sampled_tiles.tile_path.values
    df_s['cohort_0'] = df_sampled_tiles.cohort.values
    return df_s[['tile_path', 'paired_tile_0', 'cohort_0']]


def get_paired_tiles_same_slide(df_c):
    df_c['paired_tile_1'] = df_c.tile_path.sample(frac=1).values
    return df_c[['tile_path', 'paired_tile_1']]


def attached_df_paired_tiles(df):
    df_c_s_dict = {cohort: list(zip(*df_c.groupby('slide_uuid')))
                   for cohort, df_c in df[['slide_uuid', 'cohort', 'tile_path']].groupby('cohort')}
    df_pairs_0 = df.groupby('slide_uuid').apply(partial(get_paired_tiles_diff_slide_same_cohort,
                                                        df_c_s_dict=df_c_s_dict))
    df_pairs_1 = df.groupby('slide_uuid').apply(get_paired_tiles_same_slide)
    df.loc[df_pairs_0.index, 'paired_tile_0'] = df_pairs_0.paired_tile_0.values
    df.loc[df_pairs_0.index, 'cohort_0'] = df_pairs_0.cohort_0.values
    df.loc[df_pairs_1.index, 'paired_tile_1'] = df_pairs_1.paired_tile_1.values
    return df


def train():
    df, train_transform, test_transform, logger, callbacks, model = init_task()
    train_general(df, train_transform, test_transform, logger, callbacks, model,
                  load_pairs=Configs.SC_KW_ARGS['load_pairs'])


def load_df_labels_merged_tiles():
    df_labels = pd.read_csv(Configs.SC_LABEL_DF_PATH, sep='\t')
    df_labels.cohort = df_labels.cohort.apply(lambda c: c if c not in ['COAD', 'READ'] else 'CRC')
    df_labels = df_labels[df_labels['subtype'].isin(['CIN', 'GS'])]
    df_labels = df_labels[df_labels.cohort.isin(Configs.SC_COHORT_TO_IND.keys())]
    df_labels['slide_uuid'] = df_labels.slide_path.apply(lambda p: os.path.basename(os.path.dirname(p)))
    df_labels[Configs.joined['Y_TO_BE_STRATIFIED']] = df_labels['subtype'].astype(str) + '_' + df_labels['cohort']
    # merging labels and tiles
    df_tiles = pd.read_csv(Configs.SC_DF_TILE_PATHS_PATH)
    # df_tiles = df_tiles.groupby('slide_uuid').apply(lambda df: df.sample(min(len(df), 100), replace=False)).reset_index(drop=True)
    df_labels_merged_tiles = df_labels.merge(df_tiles, how='inner', on='slide_uuid', suffixes=('', '_x')).reset_index(
        drop=True)
    df_labels_merged_tiles = attached_df_paired_tiles(df_labels_merged_tiles)
    df_labels_merged_tiles['y'] = -1
    return df_labels, df_labels_merged_tiles


def init_task():
    model = init_model()
    train_transform, test_transform = init_training_transforms()
    logger, callbacks = init_training_callbacks()
    df_labels_merged_tiles = init_data()
    return df_labels_merged_tiles, train_transform, test_transform, logger, callbacks, model


def init_data():
    set_sharing_strategy('file_system')
    set_start_method("spawn")

    Logger.log("Loading Datasets..", log_importance=1)
    df_labels, df_labels_merged_tiles = load_df_labels_merged_tiles()


    if Configs.SC_KW_ARGS.get('calc_proportions_class_w', None):
        Configs.SC_CLASS_WEIGHT = df_labels.groupby(
            Configs.SC_LABEL_COL).slide_uuid.nunique().to_dict()

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




