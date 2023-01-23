from dataclasses import dataclass
import os
from pathlib import Path
from .preprocessing.pen_filter import get_pen_color_palette
from .utils import set_global_configs
import torch


@dataclass
class ConfigsClass:
    RANDOM_SEED = 123
    VERBOSE = 3  # 1 logs to LOG_FILE, 2 logs to console, 3 logs to both to file and console
    ROOT = Path(__file__).parent.parent.resolve()
    PROGRAM_LOG_FILE = 'log.txt'  # slide level log is in the slide dir. Use --bring-slide-logs to get all slide logs.
    SLIDE_LOG_FILE = "log.txt"  # slide level log
    TILE_PROGRESS_LOG_FREQ = 20  # report progress every process of x tiles (convenient for multiprocessing)
    LOG_IMPORTANCE = 1  # 0 (all), 1 or 2 (only high importance logs)
    LOAD_METADATA = False
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    LOG_FORMAT = {'format': '%(process)d  %(asctime)s  [%(name)s] - %(message)s', 'datefmt':'%d-%m-%y %H:%M:%S'}
    # Assuming TCGA folder structure, where each slide is in a separate dir and the dir is named after the slide ID
    SLIDES_DIR = os.path.join(ROOT, 'data', 'slides')
    PROCESSED_TILES_DIR = os.path.join(ROOT, 'data', 'processed_tiles')
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
    COLOR_NORM_REF_IMG = os.path.join(ROOT, 'src', 'preprocessing', 'color_norm_reference_image.png')
    ATTRS_TO_COLOR_MAP = {TISSUE_ATTR: 'pink', OTSU_FILTER['attr_name']: 'white', BLACK_FILTER['attr_name']: 'grey',
                          PEN_FILTER['attr_name']: 'green', PEN_FILTER['attr_name_not_filtered']: 'blue',
                          COLOR_NORM_FAIL: 'yellow'}

    def __init__(self):
        set_global_configs(verbose=self.VERBOSE,
                           log_file=self.PROGRAM_LOG_FILE,
                           log_importance=self.LOG_IMPORTANCE,
                           log_format=self.LOG_FORMAT,
                           random_seed=self.RANDOM_SEED,
                           tile_progress_log_freq=self.TILE_PROGRESS_LOG_FREQ)


Configs = ConfigsClass()


