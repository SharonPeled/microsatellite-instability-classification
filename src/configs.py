from dataclasses import dataclass
import os
from pathlib import Path
from .preprocessing.pen_filter import get_pen_color_palette
from .utils import set_random_seed
from .components.Logger import Logger
import torch


@dataclass
class ConfigsClass:
    RANDOM_SEED = 123
    VERBOSE = 3 # 1 logs to LOG_FILE, 2 logs to console, 3 logs to both to file and console
    ROOT = Path(__file__).parent.parent.resolve()
    LOG_FILE = 'log.txt'
    LOG_IMPORTANCE = 0
    LOAD_METADATA = False
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    LOG_FORMAT = {'format': '%(asctime)s  [%(name)s] - %(message)s', 'datefmt':'%d-%m-%y %H:%M:%S'}
    SLIDES_DIR = os.path.join(ROOT, 'data', 'test_slides')
    PROCESSED_TILES_DIR = os.path.join(ROOT, 'data', 'test_processed_tiles')
    TILE_SIZE = 512  # should be divisible by downsample of reduced image, the easiest way is to set to be a power of 2
    REDUCED_LEVEL_TO_MEMORY = [3, 2]  # attempting to load according to order
    TARGET_MAG_POWER = 20
    MAG_ATTR = 'openslide.objective-power'
    OTSU_FILTER = {'threshold': 0.3, 'attr_name': 'Background'}  # tile with less than threshold percent tissue is filtered
    BLACK_FILTER = {'threshold': 0.5, 'attr_name': 'Black',
                    'color_palette': {'r': 100, 'g':100, 'b':100}}  # tile with more than threshold percent black is filtered
    PEN_FILTER = {'threshold': 0.7, 'attr_name': 'Pen', 'attr_name_not_filtered': 'Pen_not_filtered', 'min_pen_tiles': 0.05,
                  'color_palette': get_pen_color_palette()}  # tile with more than threshold percent pen is filtered
    TISSUE_ATTR = 'Tissue'
    COLOR_NORM_SUCC = 'Normalized'
    COLOR_NORM_FAIL = 'Normalizing_Fail'
    COLOR_NORM_REF_IMG = os.path.join(ROOT, 'src', 'preprocessing', 'color_norm_reference_image.png')
    ATTRS_TO_COLOR_MAP = {TISSUE_ATTR: 'pink', OTSU_FILTER['attr_name']: 'white', BLACK_FILTER['attr_name']: 'grey',
                          PEN_FILTER['attr_name']: 'green', PEN_FILTER['attr_name_not_filtered']: 'blue',
                          COLOR_NORM_FAIL: 'yellow'}

    def __init__(self):
        Logger.set_default_logger(self)
        set_random_seed(self.RANDOM_SEED)


Configs = ConfigsClass()


