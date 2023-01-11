from dataclasses import dataclass
import os
from pathlib import Path
from .preprocessing.pen_filter import get_pen_color_palette
from .utils import set_random_seed
from .components.Logger import Logger


@dataclass
class ConfigsClass:
    RANDOM_SEED = 123
    VERBOSE = 3 # 1 logs to LOG_FILE, 2 logs to console, 3 logs to both to file and console
    ROOT = Path(__file__).parent.parent.resolve()
    LOG_FILE = 'log.txt'
    LOG_IMPORTANCE = 1
    LOAD_METADATA = False
    LOG_FORMAT = {'format': '%(asctime)s  [%(name)s] - %(message)s', 'datefmt':'%d-%m-%y %H:%M:%S'}
    SLIDES_DIR = os.path.join(ROOT, 'data', 'test_slides')
    PROCESSED_TILES_DIR = os.path.join(ROOT, 'data', 'test_processed_tiles')
    TILE_SIZE = 512
    TARGET_MPP = 0.5
    MPP_ATTRIBUTE = 'aperio.MPP'
    OTSU_FILTER = {'threshold': 0.3, 'suffix': 'Background'}  # tile with less than threshold percent tissue is filtered
    BLACK_FILTER = {'threshold': 0.5, 'suffix': 'Black', 'min_black_tiles': 0.05,
                    'color_palette': {'r': 100, 'g':100, 'b':100}}  # tile with more than threshold percent black is filtered
    PEN_FILTER = {'threshold': 0.7, 'suffix': 'Pen', 'min_pen_tiles': 0.05,
                  'color_palette': get_pen_color_palette()}  # tile with more than threshold percent pen is filtered
    SUPERPIXEL_SIZE = 2
    TISSUE_SUFFIX = 'Tissue'
    TILE_RECOVERY_SUFFIX = 'Recovered'
    COLOR_NORMED_SUFFIX = 'Normed'
    FAIL_COLOR_NORMED_SUFFIX = 'Normed_Fail'
    TILE_SUFFIXES = {'filters': [OTSU_FILTER['suffix'], BLACK_FILTER['suffix'], PEN_FILTER['suffix'], FAIL_COLOR_NORMED_SUFFIX],
                     'color_normed': COLOR_NORMED_SUFFIX,
                     'failed_color_normed': FAIL_COLOR_NORMED_SUFFIX,
                     'recovered': TILE_RECOVERY_SUFFIX,
                     'background': OTSU_FILTER['suffix'],
                     'tissue': TISSUE_SUFFIX}
    COLOR_NORM_REF_IMG = os.path.join(ROOT, 'src', 'preprocessing', 'color_norm_reference_image.png')
    SUFFIXES_TO_COLOR_MAP = {'tissue': 'pink', OTSU_FILTER['suffix']: 'white', BLACK_FILTER['suffix']: 'grey',
                             PEN_FILTER['suffix']: 'green', TILE_RECOVERY_SUFFIX:'blue', FAIL_COLOR_NORMED_SUFFIX: 'yellow'}

    def __init__(self):
        Logger.set_default_logger(self)
        set_random_seed(self.RANDOM_SEED)


Configs = ConfigsClass()


