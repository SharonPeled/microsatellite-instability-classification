import numpy as np
import pandas as pd
import torch
import random
from .components.Logger import Logger
import datetime
from glob import glob
import shutil
import os
from torch.nn.functional import conv2d


def conv2d_to_device(img_np, kernel_size, stride, device):
    if len(img_np.shape) == 2:
        new_shape = (1, 1, *img_np.shape)
    elif len(img_np.shape) == 3:
        new_shape = (1, *img_np.shape)
    img_t = torch.from_numpy(img_np).to(device).reshape(new_shape)
    return conv2d(img_t.float(), torch.ones((1, 1, kernel_size, kernel_size), device=device).float(),
                  stride=kernel_size).squeeze().cpu().numpy()


def get_time():
    now = datetime.datetime.now()
    return now.strftime("%d-%m-%y %H:%M:%S")


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def center_crop_from_tile_size(height, width, tile_size):
    x_tiles = height // tile_size
    y_tiles = width // tile_size
    x_margins = height - x_tiles * tile_size
    y_margins = width - y_tiles * tile_size
    return y_margins // 2, x_margins // 2, tile_size * y_tiles, tile_size * x_tiles, y_tiles, x_tiles


def center_crop_from_shape(height, width, new_height, new_width):
    x_margins = height - new_height
    y_margins = width - new_width
    return y_margins // 2, x_margins // 2, new_width, new_height


def generate_spatial_filter_mask(df, shape, attr):
    """
    Generates a mask for <attr> tiles.
    The array has shape (squeezed) of the original thumbnail, such that each cell represent the tile in that position.
    :param df: Dataframe with row,col columns tile position in thumbnail
    :param shape: mask shape
    :param attr: attr column name in df
    :return: Array where each cell is 1 if a tile was is <attr> and 0 otherwise
    """
    mask = np.zeros(shape)
    tuple_inds = df[df[attr]==True].index
    rows_inds = [t[0] for t in tuple_inds]
    cols_inds = [t[1] for t in tuple_inds]
    mask[rows_inds, cols_inds] = 1
    return mask


def bring_files(folder_in, file_extension, folder_out):
    os.makedirs(folder_out)
    filepaths = glob(f"{folder_in}/**/*.{file_extension}", recursive=True)
    for i, filepath in enumerate(filepaths):
        basename = os.path.basename(filepath)
        parent_dir = os.path.basename(os.path.dirname(filepath))
        shutil.copyfile(filepath, os.path.join(folder_out, f"{i}_{parent_dir}_{basename}"))

