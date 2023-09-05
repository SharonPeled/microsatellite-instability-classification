from src.subtype_classification.init_task_pretraining import init_task
from src.general_utils import rm_tmp_files
from src.components.datasets.ProcessedTileDataset import ProcessedTileDataset
from src.configs import Configs
from functools import partial
from src.training_utils import SLL_vit_small_cohort_aware
from src.components.objects.DINO.main_dino import train_dino, get_args_parser
from src.components.objects.DINO.run_with_submitit import main, parse_args
import argparse
from pathlib import Path
from src.components.objects.Logger import Logger


def train():
    df, logger = init_task()
    rm_tmp_files()
    dataset = ProcessedTileDataset(df_labels=df, transform=None, cohort_to_index=Configs.joined['COHORT_TO_IND'],
                                   num_mini_epochs=Configs.DN_NUM_MINI_EPOCHS, pretraining=True)
    Configs.DINO_DICT['dataset'] = dataset
    Configs.DINO_DICT['model_fn'] = partial(SLL_vit_small_cohort_aware, pretrained=True,
                                            progress=False, key='DINO_p16',
                                            cohort_aware_dict=Configs.SC_KW_ARGS['cohort_aware_dict'])
    Configs.DINO_DICT['logger'] = logger
    Logger.log(f"Dino CMD: {Configs.DINO_CMD_flags}")
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




