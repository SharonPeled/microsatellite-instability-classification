import numpy as np
import pandas as pd
import torch
import random
from .components.Logger import Logger
import datetime
from glob import glob
import shutil
import os
from torch.nn.functional import conv2d, softmax
import matplotlib.pyplot as plt
import pyvips


def set_global_configs(verbose, log_file_args, log_importance, log_format, random_seed, tile_progress_log_freq):
    Logger.set_default_logger(verbose, log_file_args, log_importance, log_format, tile_progress_log_freq)
    set_random_seed(random_seed)


def conv2d_to_device(img_np, kernel_size, stride, device):
    if len(img_np.shape) == 2:
        new_shape = (1, 1, *img_np.shape)
    elif len(img_np.shape) == 3:
        new_shape = (1, *img_np.shape)
    else:
        raise Exception(f"conv2d: Invalid shape {img_np.shape}")
    img_t = torch.from_numpy(img_np).to(device).reshape(new_shape)
    return conv2d(img_t.float(), torch.ones((1, 1, kernel_size, kernel_size), device=device).float(),
                  stride=stride).squeeze().cpu().numpy()


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


def bring_files(folder_in, file_format, folder_out):
    if not os.path.exists(folder_out):
        os.makedirs(folder_out)
    filepaths = glob(f"{folder_in}/**/{file_format}", recursive=True)
    for i, filepath in enumerate(filepaths):
        basename = os.path.basename(filepath)
        parent_dir = os.path.basename(os.path.dirname(filepath))
        shutil.copyfile(filepath, os.path.join(folder_out, f"{i}_{parent_dir}_{basename}"))


def bring_joined_log_file(folder_in, file_format, filepath_out):
    if os.path.isdir(os.path.dirname(filepath_out)) and not os.path.exists(os.path.dirname(filepath_out)):
        os.makedirs(os.path.dirname(filepath_out))
    filepaths = glob(f"{folder_in}/**/{file_format}", recursive=True)
    sep_line = f"\n{'-'*100}\n"
    with open(filepath_out, 'w') as outfile:
        for filepath in filepaths:
            with open(filepath, 'r') as infile:
                outfile.write(infile.read() + sep_line)


def delete_all_artifacts(configs):
    remove_artifact(os.path.join(configs.ROOT, configs.PROGRAM_LOG_FILE))
    remove_artifact(configs.PROCESSED_TILES_DIR)
    slide_paths = sorted(glob(f"{configs.SLIDES_DIR}/**/*.svs", recursive=True))  # all .svs files
    for path in slide_paths:
        slide_dir = os.path.dirname(path)
        remove_artifact(os.path.join(slide_dir, configs.PROGRAM_LOG_FILE_ARGS[0]))
        remove_artifact(os.path.join(slide_dir, 'metadata.json'))
        remove_artifact(os.path.join(slide_dir, 'thumbnail.png'))
        remove_artifact(os.path.join(slide_dir, 'tumor_thumbnail.png'))


def remove_artifact(path):
    if os.path.exists(path):
        if os.path.isfile(path):
            os.remove(path)
        else:
            shutil.rmtree(path)


def load_df_tumor_pred(pred_dir, class_to_index, tumor_class_name):
    df_paths = glob(f"{pred_dir}/**/df_pred_*", recursive=True)  # df pred from all devices
    df = pd.concat([pd.read_csv(path) for path in df_paths], ignore_index=True)
    score_names = sorted(class_to_index.keys(), key=lambda k: class_to_index[k])
    logits = softmax(torch.from_numpy(df[score_names].values), dim=-1)
    tumor_prob = logits[:, class_to_index[tumor_class_name]]
    df['tumor_prob'] = tumor_prob
    df['Tumor'] = torch.argmax(logits, dim=1) == class_to_index[tumor_class_name]
    return df


def generate_thumbnails_with_tumor_classification(df_tumor_pred, slides_dir):
    df_tumor_pred['row'] = df_tumor_pred.tile_path.apply(lambda p: int(os.path.basename(p).split('_')[0]))
    df_tumor_pred['col'] = df_tumor_pred.tile_path.apply(lambda p: int(os.path.basename(p).split('_')[1]))
    df_tumor_pred['slide_uuid'] = df_tumor_pred.tile_path.apply(lambda p: os.path.basename(os.path.dirname(p)))
    summary_df_paths = glob(f"{slides_dir}/**/summary_df.csv", recursive=True)
    df_list = []
    for path in summary_df_paths:
        summary_df = pd.read_csv(path, index_col=0)
        slide_uuid = os.path.basename(os.path.dirname(path))
        summary_df['slide_path'] = glob(f"{slides_dir}/{slide_uuid}/*.svs", recursive=True)[0]
        summary_df['slide_uuid'] = slide_uuid
        summary_df['row'] = summary_df.index.map(lambda a: eval(a)[0])
        summary_df['col'] = summary_df.index.map(lambda a: eval(a)[1])
        df_list.append(summary_df)
    summary_df_combined = pd.concat(df_list, ignore_index=True)
    df_merged = summary_df_combined.merge(df_tumor_pred, how='left', on=['slide_uuid', 'row', 'col'])
    df_merged.tumor_prob.fillna(0, inplace=True)  # all non-tumor tissue tiles has tumor_prob of 0
    df_merged.Tumor.fillna(False, inplace=True)
    df_merged.row = df_merged.row.astype(int)
    df_merged.col = df_merged.col.astype(int)
    for slide_uuid, slide_summary_df in df_merged.groupby('slide_uuid'):
        slide_path = slide_summary_df.slide_path.values[0]
        generate_tumor_thumbnail(slide_summary_df, slide_path)
        slide_summary_df.to_csv(os.path.join(os.path.dirname(slide_path), 'summary_df_pred_merged.csv'))


def add_cell_to_ax(row, col, ax, **kwargs):
    rect = plt.Rectangle((col - .5, row - .5), 1, 1, **kwargs)
    ax.add_patch(rect)
    return rect


def generate_tumor_thumbnail(slide_summary_df, slide_path):
    num_rows = slide_summary_df['row'].max() + 1
    num_cols = slide_summary_df['col'].max() + 1
    grid = np.zeros((num_rows, num_cols))

    tissue_inds = slide_summary_df[slide_summary_df.Tissue][['row', 'col']]
    grid[tissue_inds.row.values, tissue_inds.col.values] = 0.2  # pinkish color

    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(10, 10))
    for ax in [ax1, ax2]:
        ax.set_xticks([])
        ax.set_yticks([])

    ax1.imshow(grid, cmap=plt.get_cmap('Reds'), vmin=0.0, vmax=1.0)
    patches = [add_cell_to_ax(row, col, ax1, fill=True, color='tab:red')
               for row, col in slide_summary_df[slide_summary_df.Tumor][['row', 'col']].values]
    patches[0].set_label('Tumor') if len(patches) > 0 else None
    ax1.legend(loc='lower right')

    thumb = pyvips.Image.thumbnail(slide_path, 512)
    ax2.imshow(thumb)

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.tight_layout()
    fig.savefig(os.path.join(os.path.dirname(slide_path), 'tumor_thumbnail.png'), bbox_inches='tight', pad_inches=0.5)
    plt.close(fig)
    Logger.log(f"""Tumor Thumbnail Saved {os.path.dirname(slide_path)}.""", log_importance=1)



















