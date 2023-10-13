import pandas as pd
from src.training_utils import init_training_transforms, init_training_callbacks
from torch.multiprocessing import set_start_method, set_sharing_strategy
from src.components.datasets.ProcessedTileDataset import ProcessedTileDataset
from src.configs import Configs
from functools import partial
from src.training_utils import SLL_vit_small_cohort_aware
from src.components.objects.DINO.main_dino import train_dino, get_args_parser
from src.components.objects.DINO.run_with_submitit import main, parse_args
import argparse
from pathlib import Path
from src.components.objects.Logger import Logger


def set_configs():
    df, logger = init_task()
    dataset = ProcessedTileDataset(df_labels=df, transform=None, cohort_to_index=Configs.joined['COHORT_TO_IND'],
                                   num_mini_epochs=Configs.DN_NUM_MINI_EPOCHS, pretraining=True,
                                   mini_epoch_shuffle_seed=Configs.RANDOM_SEED)
    Configs.DINO_DICT['dataset'] = dataset
    Configs.DINO_DICT['model_fn'] = partial(SLL_vit_small_cohort_aware, pretrained=True,
                                            progress=False, key='DINO_p16',
                                            cohort_aware_dict=Configs.SC_KW_ARGS['cohort_aware_dict'])
    Configs.DINO_DICT['logger'] = logger
    Logger.log(f"Dino CMD: {Configs.DINO_CMD_flags}")


def train():
    set_configs()
    if Configs.USE_SLURM:
        parser = argparse.ArgumentParser("Submitit for DINO", parents=[get_args_parser(), parse_args()])
        args = parser.parse_args(Configs.DINO_CMD_flags.split())
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        main(args)
    else:
        parser = argparse.ArgumentParser('DINO', parents=[get_args_parser()])
        args = parser.parse_args(Configs.DINO_CMD_flags.split())
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        train_dino(args, configs=Configs)


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
    df_tiles = pd.read_csv(Configs.SC_DF_TILE_PATHS_PATH_512)
    df_tiles.cohort = df_tiles.cohort.apply(lambda c: c if c not in ['COAD', 'READ'] else 'CRC')
    num_slides = len(df_tiles.slide_uuid.unique())

    if Configs.DINO_DICT.get('FoVs_augs_amounts', None):
        tile_df_paths = [Configs.SC_DF_TILE_PATHS_PATH_256, Configs.SC_DF_TILE_PATHS_PATH_1024]
        df_list = []
        for i, path in enumerate(tile_df_paths):
            if Configs.SC_KW_ARGS.get('FoVs_augs_amounts')[i] is None or Configs.SC_KW_ARGS.get('FoVs_augs_amounts')[i] == 0:
                continue
            df_tiles_aug = pd.read_csv(path)
            df_tiles_aug.cohort = df_tiles_aug.cohort.apply(lambda c: c if c not in ['COAD', 'READ'] else 'CRC')
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



