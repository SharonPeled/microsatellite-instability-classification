from dataclasses import dataclass
import os
from pathlib import Path
from preprocessing.pen_filter import get_pen_color_palette


@dataclass
class ConfigsClass():
    ROOT = Path(__file__).parent.resolve()
    SLIDE_MANIFEST_PATH = os.path.join(ROOT, 'data', 'slides', 'manifest_COAD_READ_MSIStatus_DX.txt')
    SLIDE_DIR = os.path.join(ROOT, 'data', 'slides')
    TILE_DIR = os.path.join(ROOT, 'data', 'tiles')
    PROCESSED_TILE_DIR = os.path.join(ROOT, 'data', 'processed_tiles')
    TILE_SIZE = 512
    TARGET_MPP = 0.5
    MPP_ATTRIBUTE = 'aperio.MPP'
    OTSU_FILTER = {'threshold': 0.3, 'suffix': 'BG'}  # tile with less than threshold percent tissue is filtered
    BLACK_FILTER = {'threshold': 0.5, 'suffix': 'BLK',
                    'color_palette':{'r': 100, 'g':100, 'b':100}}  # tile with more than threshold percent black is filtered
    PEN_FILTER = {'threshold': 0.8, 'suffix': 'PEN',
                  'color_palette':get_pen_color_palette()}  # tile with more than threshold percent pen is filtered


Configs = ConfigsClass()

# img:
#     resize
#     crop
#     tiled
#     tiles:
#         filter_otsu
#         filter_black
#         filter_pen
