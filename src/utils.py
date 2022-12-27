import numpy as np
from components.Tile import Tile


def generate_spatial_filter_array(df, f):
    """
    Generates an array where each cell represents whether a tile was filtered or not.
    The array has shape (squeezed) of the original thumbnail, such that each cell represent the tile in that position.
    :param df: Dataframe with row,col columns represnts til
    :param f: Filter column name in df
    :return: Array where each cell is 1 if a tile was filtered by f and 0 otherwise
    """
    num_rows, num_cols = df['row'].max(), df['col'].max()
    array = np.zeros((num_rows, num_cols))
    inds = df[df[f] == True][['row', 'col2']].values
    array[np.ix_(*inds.T)] = 1
    return array


def get_filtered_tiles_to_recover(df, f, superpixel_size):
    """
    Identify tiles that their surroundings wasn't filtered (square with radius superpixel_size).
    Usually, insignificant tiles come in big groups (background, pen, black spots..), therefore if tile isn't
    part a big filtered section then returns it to valid state.
    :param array:
    :param superpixel_size:
    :param mark:
    :return:
    """
    array = generate_spatial_filter_array(df, f)
    tile_to_recover = []
    height, width = array.shape
    for (i, j) in np.argwhere(array==1):
        row_min, row_max = max(0, i-superpixel_size), min(i+1+superpixel_size, height)
        col_min, col_max = max(0,j-superpixel_size), min(j+1+superpixel_size, width)
        if (array[row_min:row_max, col_min:col_max]).sum() == 1:
            tile_to_recover.append(Tile(path=df[(df.row == i)&(df.col == j)].tile_path.values[0]))
    return tile_to_recover
