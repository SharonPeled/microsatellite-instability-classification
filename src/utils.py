import numpy as np
from .components.Tile import Tile
import torch
import random


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def generate_spatial_filter_mask(df, filters):
    """
    Generates a mask for filtered tiles.
    The array has shape (squeezed) of the original thumbnail, such that each cell represent the tile in that position.
    :param df: Dataframe with row,col columns tile position in thumbnail
    :param filters: Filter columns names in df
    :return: Array where each cell is 1 if a tile was filtered by one of the filters and 0 otherwise
    """
    num_rows, num_cols = df['row'].max(), df['col'].max()
    mask = np.zeros((num_rows+1, num_cols+1))
    filter_inds = df[(df[filters] == True).any(axis=1)][['row', 'col']]
    mask[filter_inds['row'].values, filter_inds['col'].values] = 1
    return mask


def get_filtered_tiles_paths_to_recover(df, filters, superpixel_size):
    """
    Identify tiles that their surroundings wasn't filtered (square with radius superpixel_size).
    Usually, insignificant tiles come in big groups (background, pen, black spots..), therefore if tile isn't
    part a big filtered section then returns it to valid state.
    """
    mask = generate_spatial_filter_mask(df, filters)
    tile_paths_to_recover = []
    height, width = mask.shape
    for (i, j) in np.argwhere(mask==1):
        row_min, row_max = max(0, i-superpixel_size), min(i+1+superpixel_size, height)
        col_min, col_max = max(0,j-superpixel_size), min(j+1+superpixel_size, width)
        if (mask[row_min:row_max, col_min:col_max]).sum() == 1:
            # filtered tile is surrounded with tissue - recovering
            tile_paths_to_recover.append(df.loc[(i,j)].tile_path)
    return tile_paths_to_recover
