from ..configs import Configs
from skimage import color, filters
import os
import pyvips
import numpy as np
from .pen_filter import get_pen_mask
from ..utils import generate_spatial_filter_mask, center_crop_from_shape, center_crop_from_tile_size, conv2d
from histomicstk.preprocessing.color_normalization.\
    deconvolution_based_normalization import deconvolution_based_normalization
from ..components.Tile import Tile
from ..components.Logger import Logger
import matplotlib.pyplot as plt
import matplotlib.colors as colors


def load_slide(slide, load_level=None):
    slide.load(load_level)
    return slide


def resize(slide, target_mag_power, mag_attr):
    slide_mag_power = int(slide.get(mag_attr))
    scale = target_mag_power / slide_mag_power
    if scale > 1:
        Logger.log(f"Slide magnification power ({slide_mag_power}) is less then target ({target_mag_power}).",
                   importace=2)
    return slide.resize(scale)


def center_crop_reduced_image(slide, tile_size, tissue_attr):
    downsample = slide.get_downsample()
    if not tile_size % downsample == 0:
        raise Exception(f"Tile size {tile_size} is not divisible by downsample {downsample}.")
    tile_size_r = tile_size // downsample
    img_r_height, img_r_width, _ = slide.img_r.shape
    y_margins, x_margins, cropped_width, cropped_height, y_tiles, x_tiles = center_crop_from_tile_size(img_r_height,
                                                                                                       img_r_width,
                                                                                                       tile_size_r)
    slide.img_r = slide.img_r[x_margins:cropped_height+x_margins, y_margins:cropped_width+y_margins, :]
    slide.set('tile_size', tile_size)
    slide.set('tile_size_r', tile_size_r)
    slide.set('num_x_tiles', x_tiles)
    slide.set('num_y_tiles', y_tiles)
    slide.set('num_tiles', x_tiles*y_tiles)
    slide.set('reduced_shape', (cropped_height, cropped_width))
    slide.init_tiles_summary_df(tissue_attr)
    return slide


def load_reduced_image_to_memory(slide):
    slide.load_level_to_memory()
    return slide


def unload_reduced_image(slide):
    slide.unload_level_from_memory()
    return slide


def center_crop(slide):
    cropped_height_r, cropped_width_r = slide.get('reduced_shape')
    cropped_height = cropped_height_r * slide.get_downsample()
    cropped_width = cropped_width_r * slide.get_downsample()
    y_margins, x_margins, cropped_width, cropped_height = center_crop_from_shape(slide.height, slide.width,
                                                                                 cropped_height, cropped_width)
    t = {k:slide.get(k) for k in slide.get_fields()}
    slide.set('shape', (cropped_height, cropped_height))
    return slide.crop(y_margins, x_margins, cropped_width, cropped_height)


def filter_otsu_reduced_image(slide, threshold, attr_name):
    imr_r_bw = color.rgb2gray(slide.img_r)
    otsu_val = filters.threshold_otsu(image=imr_r_bw)
    slide.set('otsu_val', otsu_val)
    tile_size_r = slide.get('tile_size_r')
    tile_background_fracs = conv2d(imr_r_bw < otsu_val, np.ones((tile_size_r,tile_size_r)), tile_size_r) /\
                            (tile_size_r**2)
    slide.add_attribute_summary_df(np.argwhere(tile_background_fracs < threshold), attr_name, True, False, is_tissue_filter=True)
    return slide


def filter_black_reduced_image(slide, color_palette, threshold, attr_name):
    r, g, b = np.rollaxis(slide.img_r, -1)
    mask = (r < color_palette['r']) & (g < color_palette['g']) & (b < color_palette['b'])
    tile_size_r = slide.get('tile_size_r')
    tile_black_fracs = conv2d(mask, np.ones((tile_size_r, tile_size_r)), tile_size_r) / (
                tile_size_r ** 2)
    slide.add_attribute_summary_df(np.argwhere(tile_black_fracs > threshold), attr_name, True, False, is_tissue_filter=True)
    return slide


def filter_pen_reduced_image(slide, color_palette, threshold, attr_name, min_pen_tiles, attr_name_not_filtered):
    r, g, b = np.rollaxis(slide.img_r, -1)
    pen_colors = color_palette.keys()
    mask = sum([get_pen_mask(r, g, b, color_palette, color) for color in pen_colors]) > 0
    tile_size_r = slide.get('tile_size_r')
    tile_pen_fracs = conv2d(mask, np.ones((tile_size_r, tile_size_r)), tile_size_r) / (
                tile_size_r ** 2)
    tile_pen_mask = tile_pen_fracs > threshold
    # Empirical observation: it is commonly found that pen tiles tend to appear in large
    # amounts (when pen circle the tissue).
    # Few individual pen tiles are unlikely to be present. In case a few pen tiles is found it may be
    # a colorful tissue misinterpreted as a pen tile.
    # therefore, if the number of pen tiles is too low, it may be best to not filter them at all.
    if tile_pen_mask.sum() / tile_pen_mask.size < min_pen_tiles:
        slide.add_attribute_summary_df(np.argwhere(tile_pen_mask), attr_name_not_filtered, True, False, is_tissue_filter=False)
    else:
        slide.add_attribute_summary_df(np.argwhere(tile_pen_mask), attr_name, True, False, is_tissue_filter=True)
    return slide


def save_tiles(slide, tiles_dir, tile_size):
    """
    saves tiles to RGB (3 channels), even if the original image is RGBA.
    :param slide:
    :param tiles_dir:
    :param tile_size:
    :return:
    """
    if slide.get('tiling_complete', soft=True):
        return slide
    slide.set_tile_dir(tiles_dir)
    slide.dzsave(
        slide.get('tile_dir'),
        suffix='.jpg',
        tile_size=tile_size,
        overlap=0,
        depth='one'
    )
    if os.path.exists(slide.get('tile_dir') + '_files') and not os.path.exists(slide.get('tile_dir')):
        # dzsave adds _files extension to output dir
        os.rename(slide.get('tile_dir') + '_files', slide.get('tile_dir'))
    slide.set('tiling_complete', True)
    return slide


def load_tile(tile):
    tile.load()
    return tile


def save_processed_tile(tile, processed_tiles_dir, fail_norm_attr):
    if tile.get('norm_result', soft=True) is None or tile.get('norm_result', soft=True) == fail_norm_attr:
        return tile
    tile.save(processed_tiles_dir)
    return tile


def macenko_color_norm(tile, ref_img_path, succ_norm_attr, fail_norm_attr):
    ref_tile = Tile(path=ref_img_path)
    ref_tile.load()
    stain_unmixing_routine_params = {
        'stains': ['H&E'],
        'stain_unmixing_method': 'macenko_pca',
    }
    try:
        normed_img = deconvolution_based_normalization(im_src=tile.img, im_target=ref_tile.img,
                                                       stain_unmixing_routine_params=stain_unmixing_routine_params)
        normed_tile = Tile.from_img(tile, normed_img)
        normed_tile.add_filename_suffix(succ_norm_attr)
        Logger.log(f"""Tile {normed_tile} successfully normed.""")
        tile.set('norm_result', (True, succ_norm_attr))
        return normed_tile
    except Exception as e:
        Logger.log(f"""Tile {tile} normalization fail with exception {e}.""", log_importance=2)
        tile.add_filename_suffix(fail_norm_attr)
        tile.set('norm_result', (False, fail_norm_attr))
        return tile


def generate_slide_color_grid(slide, attrs_to_colors_map):
    attrs, color_list = attrs_to_colors_map.keys(), attrs_to_colors_map.values()
    df = slide.summary_df.assign(**{a: False for a in attrs if a not in slide.summary_df.columns})  # adding missing attrs as false
    grid = np.ones((slide.get('num_x_tiles'), slide.get('num_y_tiles')))
    for i, attr in enumerate(attrs, start=1):
        mask = generate_spatial_filter_mask(df, grid.shape, attr)
        grid[mask == 1] = i + 1
    norm = colors.Normalize(vmin=1, vmax=len(color_list))
    cmap = plt.cm.colors.ListedColormap(color_list)
    color_grid = cmap(norm(grid))
    patches = [plt.plot([], [], marker="s", color=cmap(i / float(len(color_list))), ls="")[0]
               for i in range(len(color_list))]

    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(10, 10))

    for ax in [ax1, ax2]:
        ax.set_xticks([])
        ax.set_yticks([])

    ax1.imshow(color_grid)
    ax1.legend(patches, attrs, loc='lower right')

    thumb = pyvips.Image.thumbnail(slide.path, 512)
    ax2.imshow(thumb)

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.tight_layout()
    fig.savefig(os.path.join(os.path.dirname(slide.path), 'thumbnail.png'), bbox_inches='tight', pad_inches=0.5)
    Logger.log(f"""Thumbnail Saved.""", log_importance=1)
    return slide






