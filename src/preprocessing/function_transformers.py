from ..configs import Configs
from pyvips import Image
from skimage import filters
import os
import numpy as np
from pen_filter import pen_percent


def resize(slide, target_mpp):
    slide_mpp = float(slide.get(Configs.MPP_ATTRIBUTE))
    scale = slide_mpp / target_mpp
    return slide.resize(scale)


def center_crop(slide, tile_size):
    width = slide.width
    height = slide.height
    x_tiles = width // tile_size
    y_tiles = height // tile_size
    x_margins = width - x_tiles * tile_size
    y_margins = height - y_tiles * tile_size
    return slide.crop(x_margins // 2, y_margins // 2, tile_size * x_tiles, tile_size * y_tiles)


def calc_otsu(slide):
    slide_bw = slide.colourspace("b-w")
    hist = slide_bw.hist_find().numpy()
    otsu_val = filters.threshold_otsu(image=None, hist=(hist[0][:, 0], range(256)))
    slide = slide.set('otsu_val', otsu_val)
    return slide


def tile(slide, tile_dir, tile_size):
    Image.dzsave(
        slide,
        tile_dir,
        basename=slide.get_UUID(),
        suffix='.jpg',
        tile_size=tile_size,
        overlap=0,
        depth='one',
        properites=False
    )
    slide.set_tile_dir(os.path.join(tile_dir, slide.get_UUID() + '_files', '0'))
    return slide


def filter_otsu(tile, otsu_val, threshold, suffix):
    filtered_pixels = int((tile.colourspace("b-w").numpy() < otsu_val)[:, :, 0].sum())
    r = filtered_pixels / (tile.width * tile.height)
    if r < threshold: # classified as background
        tile.add_filename_suffix(suffix)
    return tile


def filter_black(tile, color_palette, threshold, suffix):
    r, g, b, _ = np.rollaxis(tile, -1)
    mask = (r < color_palette['r']) & (g < color_palette['g']) & (b < color_palette['b'])
    if mask.mean() > threshold:
        tile.add_filename_suffix(suffix)
    return tile


def filter_pen(tile, color_palette, threshold, suffix):
    pen_colors = color_palette.keys()
    max_pen_percent = max([pen_percent(tile, color_palette, color) for color in pen_colors])
    if max_pen_percent > threshold:
        tile.add_filename_suffix(suffix)
    return tile
