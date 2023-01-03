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
    TILES_DIR = os.path.join(ROOT, 'data', 'test_full_tiles')
    PROCESSED_TILES_DIR = os.path.join(ROOT, 'data', 'test_full_processed_tiles')
    TILE_SIZE = 512
    TARGET_MPP = 0.5
    MPP_ATTRIBUTE = 'aperio.MPP'
    OTSU_FILTER = {'threshold': 0.3, 'suffix': 'BG'}  # tile with less than threshold percent tissue is filtered
    BLACK_FILTER = {'threshold': 0.5, 'suffix': 'BLK', 'min_black_tiles': 0.05,
                    'color_palette': {'r': 100, 'g':100, 'b':100}}  # tile with more than threshold percent black is filtered
    PEN_FILTER = {'threshold': 0.8, 'suffix': 'PEN', 'min_pen_tiles': 0.05,
                  'color_palette': get_pen_color_palette()}  # tile with more than threshold percent pen is filtered
    SUPERPIXEL_SIZE = 2
    TILE_RECOVERY_SUFFIX = 'R'
    COLOR_NORMED_SUFFIX = 'N' # TODO: validate in the next phase of the analysis that all the tile are normalized
    TILE_SUFFIXES = {'filters': [OTSU_FILTER['suffix'], BLACK_FILTER['suffix'], PEN_FILTER['suffix']], # order does matter
                     'color_normed': COLOR_NORMED_SUFFIX,
                     'recovered': TILE_RECOVERY_SUFFIX}
    COLOR_NORM_REF_IMG = os.path.join(ROOT, 'src', 'preprocessing', 'color_norm_reference_image.png')
    SUFFIXES_TO_COLOR_MAP = {'tissue': 'pink', OTSU_FILTER['suffix']: 'white', BLACK_FILTER['suffix']: 'grey',
                             PEN_FILTER['suffix']: 'red', TILE_RECOVERY_SUFFIX:'blue'}

    def __init__(self):
        Logger.set_default_logger(self)
        set_random_seed(self.RANDOM_SEED)


Configs = ConfigsClass()


