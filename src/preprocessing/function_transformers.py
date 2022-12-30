from ..configs import Configs
from pyvips import Image
from skimage import filters
import os
import numpy as np
from .pen_filter import pen_percent
from ..utils import get_filtered_tiles_to_recover
from histomicstk.preprocessing.color_normalization.\
    deconvolution_based_normalization import deconvolution_based_normalization


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
    slide_bw = slide.colourspace("b-w")
    hist = slide_bw.hist_find().numpy()
    otsu_val = filters.threshold_otsu(image=None, hist=(hist[0][:, 0], range(256)))
    print(otsu_val) # 192
    slide.otsu_val = otsu_val
    return slide


def save_tiles(slide, tile_dir, tile_size):
    Image.dzsave(
        slide,
        tile_dir,
        basename=slide.uuid,
        suffix='.jpg',
        tile_size=tile_size,
        overlap=0,
        depth='one',
        properites=False
    )
    slide.set_tile_dir(os.path.join(tile_dir, slide.uuid + '_files', '0'))
    return slide


def load_tile(tile):
    tile.load()
    return tile


def save_processed_tile(tile, processed_tile_dir):
    tile.save(processed_tile_dir)
    return tile


def filter_otsu(tile, otsu_val, threshold, suffix):
    filtered_pixels = int((tile.colourspace("b-w").numpy() < otsu_val)[:, :, 0].sum())
    r = filtered_pixels / (tile.width * tile.height)
    if r < threshold: # classified as background
        tile.set_filename_suffix(suffix)
    return tile


def filter_black(tile, color_palette, threshold, suffix):
    r, g, b, _ = np.rollaxis(tile, -1)
    mask = (r < color_palette['r']) & (g < color_palette['g']) & (b < color_palette['b'])
    if mask.mean() > threshold:
        tile.set_filename_suffix(suffix)
    return tile


def filter_pen(tile, color_palette, threshold, suffix):
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


def macenko_color_norm(tile, ref_img):
    tile_rgb = tile[:,:,:3]
    ref_img_rgb = ref_img[:,:,:3]
    stain_unmixing_routine_params = {
        'stains': ['H&E'],
        'stain_unmixing_method': 'macenko_pca',
    }
    return deconvolution_based_normalization(im_src=tile_rgb, im_target=ref_img_rgb,
                                             stain_unmixing_routine_params=stain_unmixing_routine_params)


