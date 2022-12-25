from dataclasses import dataclass
import os


@dataclass
class ConfigsClass():
    TARGET_MPP = 0.5
    TILES_FOLDER = os.path.join([])


Configs = ConfigsClass()

img:
    resize
    crop
    tiled
    tiles:
        filter_otsu
        filter_black
        filter_pen
