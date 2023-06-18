import numpy as np
import pandas as pd
import torch
import random
from src.components.Objects.Logger import Logger
import datetime
from glob import glob
import shutil
import os
from torch.nn.functional import conv2d, softmax
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import pyvips
from sklearn.model_selection import StratifiedShuffleSplit
from src.components.Datasets.SubDataset import SubDataset
from torch.utils.data import Subset
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold


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


def get_train_test_dataset(dataset, test_size, random_state):
    # train test split
    train_test_split = StratifiedShuffleSplit(n_splits=1, test_size=test_size,
                                              random_state=random_state)
    train_inds, test_inds = next(train_test_split.split(dataset, y=dataset.targets))
    train_dataset = SubDataset(Subset(dataset, train_inds))
    test_dataset = SubDataset(Subset(dataset, test_inds))
    return train_dataset, test_dataset


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
    tuple_inds = df[(df[attr]==True)|(df[attr]==1)].index
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
    remove_artifact(os.path.join(configs.ROOT, configs.PROGRAM_LOG_FILE_ARGS[0]))
    remove_artifact(configs.PROCESSED_TILES_DIR)
    slide_paths = sorted(glob(f"{configs.SLIDES_DIR}/**/*.svs", recursive=True))  # all .svs files
    for path in slide_paths:
        slide_dir = os.path.dirname(path)
        remove_artifact(os.path.join(slide_dir, configs.PROGRAM_LOG_FILE_ARGS[0]))
        remove_artifact(os.path.join(slide_dir, configs.METADATA_JSON_FILENAME))
        remove_artifact(os.path.join(slide_dir, configs.THUMBNAIL_FILENAME))
        remove_artifact(os.path.join(slide_dir, configs.TUMOR_THUMBNAIL_FILENAME))
        remove_artifact(os.path.join(slide_dir, configs.SS_THUMBNAIL_FILENAME))
        remove_artifact(os.path.join(slide_dir, configs.SUMMARY_DF_FILENAME))
        remove_artifact(os.path.join(slide_dir, configs.TUMOR_SUMMARY_DF_PRED_MERGED_FILENAME))
        remove_artifact(os.path.join(slide_dir, configs.SS_SUMMARY_DF_PRED_MERGED_FILENAME))


def remove_artifact(path):
    if os.path.exists(path):
        if os.path.isfile(path):
            os.remove(path)
        else:
            shutil.rmtree(path)


def load_df_pred(pred_dir, class_to_index):
    df_paths = glob(f"{pred_dir}/**/df_pred_*", recursive=True)  # df pred from all devices from all time
    Logger.log(f"Loaded {len(df_paths)} pred dataframes.", log_importance=1)
    df = pd.concat([pd.read_csv(path) for path in df_paths], ignore_index=True)
    classes = list(class_to_index.keys())
    df['y_pred'] = torch.argmax(torch.from_numpy(df[classes].values), dim=1).numpy()
    logits = softmax(torch.from_numpy(df[classes].values), dim=-1)
    df[[c + '_prob' for c in classes]] = logits
    class_indices = torch.argmax(logits, dim=1)  # get index of highest score for each sample
    labels = torch.zeros(logits.shape)  # initialize binary matrix
    labels[range(labels.shape[0]), class_indices] = 1
    df[classes] = labels
    return df


def train_test_valid_split_patients_stratified(df_full, test_size, valid_size, random_seed):
    splitter = StratifiedGroupKFold(n_splits=int(1/test_size), shuffle=True, random_state=random_seed)
    split = splitter.split(X=df_full, y=df_full['int_dis_to_tum'], groups=df_full['patient_id'])
    train_inds, test_inds = next(split)
    df_train = df_full.iloc[train_inds].reset_index(drop=True)
    df_test = df_full.iloc[test_inds].reset_index(drop=True)
    splitter = StratifiedGroupKFold(n_splits=int(1/valid_size), shuffle=True, random_state=random_seed)
    split = splitter.split(X=df_train, y=df_train['int_dis_to_tum'], groups=df_train['patient_id'])
    train_inds, valid_inds = next(split)
    df_valid = df_train.iloc[valid_inds].reset_index(drop=True)
    df_train = df_train.iloc[train_inds].reset_index(drop=True)
    return df_train, df_valid, df_test


def generate_thumbnails_with_tissue_classification(df_pred, slides_dir, class_to_index, class_to_color,
                                                   summary_df_filename, summary_df_pred_merged_filename,
                                                   thumbnail_filename):
    classes = list(class_to_index.keys())
    df_pred['row'] = df_pred.tile_path.apply(lambda p: int(os.path.basename(p).split('_')[0]))
    df_pred['col'] = df_pred.tile_path.apply(lambda p: int(os.path.basename(p).split('_')[1]))
    df_pred['slide_uuid'] = df_pred.tile_path.apply(lambda p: os.path.basename(os.path.dirname(p)))
    summary_df_paths = glob(f"{slides_dir}/**/{summary_df_filename}", recursive=True)
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
    df_merged = summary_df_combined.merge(df_pred, how='left', on=['slide_uuid', 'row', 'col'])
    df_merged[classes] = df_merged[classes].fillna(0)  # non-tissue tiles (BKG/PEN)
    df_merged.row = df_merged.row.astype(int)
    df_merged.col = df_merged.col.astype(int)
    for slide_uuid, slide_summary_df in df_merged.groupby('slide_uuid'):
        slide_path = slide_summary_df.slide_path.values[0]
        generate_classified_tissue_thumbnail(slide_summary_df, slide_path, class_to_color, thumbnail_filename)
        slide_summary_df.to_csv(os.path.join(os.path.dirname(slide_path), summary_df_pred_merged_filename))


def add_cell_to_ax(row, col, ax, **kwargs):
    rect = plt.Rectangle((col - .5, row - .5), 1, 1, **kwargs)
    ax.add_patch(rect)
    return rect


def generate_classified_tissue_thumbnail(slide_summary_df, slide_path, class_to_color, thumbnail_filename):
    num_rows = slide_summary_df['row'].max() + 1
    num_cols = slide_summary_df['col'].max() + 1
    slide_summary_df.index = [(x, y) for x in range(num_rows)
                              for y in range(num_cols)]
    cmap_colors = ['white']
    classes_str = ['NON-TISSUE']
    grid = np.zeros((num_rows, num_cols))
    for i, (class_str, color) in enumerate(class_to_color.items()):
        mask = generate_spatial_filter_mask(slide_summary_df, grid.shape, class_str)
        grid[mask == 1] = i+1
        cmap_colors.append(color)
        classes_str.append(class_str)
    cmap = ListedColormap(cmap_colors)  # create custom colormap
    norm = BoundaryNorm(np.arange(len(cmap_colors) + 1), len(cmap_colors))  # create custom normalization

    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(25, 25))
    for ax in [ax1, ax2]:
        ax.set_xticks([])
        ax.set_yticks([])

    ax1.imshow(grid, cmap=cmap, norm=norm)
    plt.legend(handles=[plt.Rectangle((0, 0), 1, 1, fc=cmap_colors[i], ec="k") for i in range(len(cmap_colors))],
               labels=classes_str,
               loc='center left', bbox_to_anchor=(1, 0.5))  # add legend with class names and corresponding colors

    thumb = pyvips.Image.thumbnail(slide_path, 512)
    ax2.imshow(thumb)

    out_path = os.path.join(os.path.dirname(slide_path), thumbnail_filename)
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches='tight', pad_inches=0.5)
    plt.close(fig)
    Logger.log(f"""Tissue Classified Thumbnail Saved {out_path}.""", log_importance=1)


def generate_confusion_matrix_figure(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred, normalize='pred')
    fig = plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, cmap=plt.cm.Blues, fmt=".2f", annot_kws={"fontsize": 7},
                xticklabels=list(classes), yticklabels=list(classes))
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    return fig

















