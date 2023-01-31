from dataclasses import dataclass
import os
from pathlib import Path
from .preprocessing.pen_filter import get_pen_color_palette
from .utils import set_global_configs
import torch
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torchstain")


@dataclass
class GeneralConfigs:
    RANDOM_SEED = 123
    VERBOSE = 3  # 1 logs to LOG_FILE, 2 logs to console, 3 logs to both to file and console
    # ROOT = Path(__file__).parent.parent.resolve()
    ROOT = '/home/sharonpe/microsatellite-instability-classification'
    PROGRAM_LOG_FILE_ARGS = ['log.txt', 'a+']  # slide level log is in the slide dir. Use --bring-slide-logs to get all slide logs.
    LOG_IMPORTANCE = 1  # 0 (all), 1 or 2 (only high importance logs)
    LOG_FORMAT = {'format': '%(process)d  %(asctime)s  [%(name)s] - %(message)s', 'datefmt':'%d-%m-%y %H:%M:%S'}


@dataclass
class PreprocessingConfigs:
    SLIDE_LOG_FILE_ARGS = ['log.txt', 'w']  # slide level log
    TILE_PROGRESS_LOG_FREQ = 100  # report progress every process of x tiles (convenient for multiprocessing)
    LOAD_METADATA = True
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Assuming TCGA folder structure, where each slide is in a separate dir and the dir is named after the slide ID
    SLIDES_DIR = os.path.join(GeneralConfigs.ROOT, 'data', 'slides')
    PROCESSED_TILES_DIR = os.path.join(GeneralConfigs.ROOT, 'data', 'processed_tiles')
    TILE_SIZE = 512  # should be divisible by downsample of reduced image, the easiest way is to set to be a power of 2
    REDUCED_LEVEL_TO_MEMORY = [3, 2]  # attempting to load according to order
    TARGET_MAG_POWER = 20
    MAG_ATTR = 'openslide.objective-power'
    TILE_NON_TISSUE_THRESHOLD = 0.5  # tiles with more non tissue percent than threshold are filtered
    OTSU_FILTER = {'reduced_img_factor': 0.8, 'attr_name': 'Background',
                   'color_palette': {'s': 10, 'otsu_val_factor': 1.1}}
    BLACK_FILTER = {'attr_name': 'Black', 'color_palette': [{'v': 50}, {'v': 200, 's': 25}]}
    PEN_FILTER = {'attr_name': 'Pen', 'attr_name_not_filtered': 'Pen_not_filtered', 'min_pen_tiles': 0.025,
                  'color_palette': get_pen_color_palette()}
    TISSUE_ATTR = 'Tissue'
    COLOR_NORM_SUCC = 'Normalized'
    COLOR_NORM_FAIL = 'Normalizing_Fail'
    COLOR_NORM_REF_IMG = os.path.join(GeneralConfigs.ROOT, 'src', 'preprocessing', 'color_norm_reference_image.png')
    ATTRS_TO_COLOR_MAP = {TISSUE_ATTR: 'pink', OTSU_FILTER['attr_name']: 'white', BLACK_FILTER['attr_name']: 'grey',
                          PEN_FILTER['attr_name']: 'green', PEN_FILTER['attr_name_not_filtered']: 'blue',
                          COLOR_NORM_FAIL: 'yellow'}


@dataclass
class TumorClassificationConfigs:
    TUMOR_EXPERIMENT = 'tumor_classifier_V2'
    TUMOR_TRAINED_MODEL_PATH = os.path.join(GeneralConfigs.ROOT, 'models', 'tumor_classifier_V2.ckpt')
    TUMOR_LOG_DIR = os.path.join(GeneralConfigs.ROOT, 'models')
    TUMOR_LABELED_TILES_DIR = os.path.join(GeneralConfigs.ROOT, 'data', 'tumor_labeled_tiles')
    TUMOR_CLASS = 'TUMSTU'
    NON_TUMOR_CLASSES = ['STRMUS', 'ADIMUC']
    TUMOR_TEST_SIZE = 0.2
    TUMOR_VALID_SIZE = 0.1
    TUMOR_TRAINING_BATCH_SIZE = 16
    TUMOR_TRAINING_NUM_WORKERS = 16
    TUMOR_NUM_CLASSES = 3
    TUMOR_CLASS_TO_IND = {'ADIMUC': 0, 'STRMUS': 1, 'TUMSTU': 2}  # alphabetical order as in ImageFolder
    TUMOR_IND = TUMOR_CLASS_TO_IND['TUMSTU']
    TUMOR_INIT_LR = 1e-4
    TUMOR_NUM_EPOCHS = 10
    TUMOR_NUM_DEVICES = 2
    TUMOR_DEVICE = 'gpu'


@dataclass
class ConfigsClass(GeneralConfigs, PreprocessingConfigs, TumorClassificationConfigs):
    def __init__(self):
        set_global_configs(verbose=self.VERBOSE,
                           log_file_args=self.PROGRAM_LOG_FILE_ARGS,
                           log_importance=self.LOG_IMPORTANCE,
                           log_format=self.LOG_FORMAT,
                           random_seed=self.RANDOM_SEED,
                           tile_progress_log_freq=self.TILE_PROGRESS_LOG_FREQ)


Configs = ConfigsClass()


