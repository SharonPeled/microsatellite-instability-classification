from ..configs import Configs
from skimage import color
import os
import numpy as np
from .pen_filter import pen_percent
from ..utils import get_filtered_tiles_to_recover
from histomicstk.preprocessing.color_normalization.\
    deconvolution_based_normalization import deconvolution_based_normalization
from ..components.Tile import Tile

def load_slide(slide):
    slide.load()
    return slide


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
    # slide_bw = slide.colourspace("b-w")
    # hist = slide_bw.hist_find().numpy()
    # otsu_val = filters.threshold_otsu(image=None, hist=(hist[0][:, 0], range(256)))
    # print(otsu_val) # 194
    # slide.otsu_val = 192
    slide.set('otsu_val', 194)
    # slide.set('otsu_val', otsu_val)
    return slide


def save_tiles(slide, tiles_dir, tile_size):
    """
    saves tiles to RGB (3 channels), even if the original image is RGBA.
    :param slide:
    :param tiles_dir:
    :param tile_size:
    :return:
    """
    slide.set_tile_dir(tiles_dir)
    # slide.dzsave(
    #     slide.tile_dir,
    #     suffix='.jpg',
    #     tile_size=tile_size,
    #     overlap=0,
    #     depth='one'
    # )
    if os.path.exists(slide.tile_dir + '_files') and os.path.exists(slide.tile_dir):
        # dzsave adds _files extension to output dir
        os.rename(slide.tile_dir + '_files', slide.tile_dir)
    return slide


def load_tile(tile):
    tile.load()
    return tile


def save_processed_tile(tile, processed_tiles_dir):
    tile.save(processed_tiles_dir)
    return tile


def filter_otsu(tile, threshold, suffix, **kwargs):
    bw_img = (color.rgb2gray(tile.img)*255)
    filtered_pixels = (bw_img < tile.get('otsu_val')).sum()
    r = filtered_pixels / tile.size
    if r < threshold: # classified as background
        tile.set_filename_suffix(suffix)
    return tile


def filter_black(tile, color_palette, threshold, suffix, **kwargs):
    r, g, b = np.rollaxis(tile.img, -1)
    mask = (r < color_palette['r']) & (g < color_palette['g']) & (b < color_palette['b'])
    if mask.mean() > threshold:
        tile.set_filename_suffix(suffix)
    return tile


def filter_pen(tile, color_palette, threshold, suffix, **kwargs):
    pen_colors = color_palette.keys()
    max_pen_percent = max([pen_percent(tile, color_palette, color) for color in pen_colors])
    if max_pen_percent > threshold:
        tile.set_filename_suffix(suffix)
    return tile


def recover_missfiltered_tiles(slide, pen_filter, black_filter, superpixel_size, tile_recovery_suffix):
    df = slide.get_tile_summary_df()
    if tile_recovery_suffix in df.columns:
        df.drop(columns=tile_recovery_suffix, inplace=True)
    filters = list(df.columns[2:])
    num_unfiltered_tiles = (df[filters].sum(axis=1) == 0).sum() # zero in all filters
    tiles_to_recover = {f: set(get_filtered_tiles_to_recover(df, f, superpixel_size))
                           for f in filters}
    # very few pen/black tiles are probably not a real pen/black tiles
    # when tissu is very colorful it can be misinterpreted as pen
    # tissue dark spots can also be misinterpreted as black tiles
    # in this case, recover all tiles from that category
    # empirical observation only
    pen_suffix = pen_filter['suffix']
    black_suffix = black_filter['suffix']
    if df[pen_suffix].sum() / num_unfiltered_tiles < num_unfiltered_tiles * pen_filter['min_pen_tiles']:
        tiles_to_recover[pen_suffix].update(df[df[pen_suffix]].tile_path.values)
    if df[black_suffix].sum() / num_unfiltered_tiles < num_unfiltered_tiles * black_filter['min_pen_tiles']:
        tiles_to_recover[black_suffix].update(df[df[black_suffix]].tile_path.values)
    for tile in tiles_to_recover:
        tile.recover()


def macenko_color_norm(tile, ref_img_path):
    #TODO: probably because of background images..
    if 'BG' in tile.out_filename:
        return tile
    ref_tile = Tile(path=ref_img_path)
    ref_tile.load()
    print(tile.img.shape, ref_tile.img.shape)
    stain_unmixing_routine_params = {
        'stains': ['H&E'],
        'stain_unmixing_method': 'macenko_pca',
    }
    normed_img = deconvolution_based_normalization(im_src=tile.img, im_target=ref_tile.img,
                                             stain_unmixing_routine_params=stain_unmixing_routine_params)
    return Tile.from_img(tile, normed_img)
