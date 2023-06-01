from dataclasses import dataclass
import os
from pathlib import Path
from .preprocessing.pen_filter import get_pen_color_palette
from .utils import set_global_configs
import torch
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torchstain")


@dataclass
class GeneralConfigs:
    RANDOM_SEED = 123
    VERBOSE = 3  # 1 logs to LOG_FILE, 2 logs to console, 3 logs to both to file and console
    ROOT = Path(__file__).parent.parent.resolve()
    PROGRAM_LOG_FILE_ARGS = ['log.txt', 'a+']  # slide level log is in the slide dir. Use --bring-slide-logs to get all slide logs.
    LOG_IMPORTANCE = 1  # 0 (all), 1 or 2 (only high importance logs)
    LOG_FORMAT = {'format': '%(process)d  %(asctime)s  [%(name)s] - %(message)s', 'datefmt':'%d-%m-%y %H:%M:%S'}
    MLFLOW_SAVE_DIR = os.path.join(ROOT, 'models', 'mlruns')


@dataclass
class PreprocessingConfigs:
    PREPROCESS_RUN_NAME = '224'
    METADATA_JSON_FILENAME = f'metadata_{PREPROCESS_RUN_NAME}.json'
    SUMMARY_DF_FILENAME = f'summary_df_{PREPROCESS_RUN_NAME}.csv'
    THUMBNAIL_FILENAME = f'thumbnail_{PREPROCESS_RUN_NAME}.png'
    SLIDE_LOG_FILE_ARGS = ['log.txt', 'w']  # slide level log
    TILE_PROGRESS_LOG_FREQ = 1000  # report progress every process of x tiles (convenient for multiprocessing)
    LOAD_METADATA = True
    TO_MACENKO_NORMALIZE = False
    PREPROCESSING_DEVICE = 'cpu'
    # Assuming TCGA folder structure, where each slide is in a separate dir and the dir is named after the slide ID
    SLIDES_DIR = os.path.join(GeneralConfigs.ROOT, 'data', 'slides')
    PROCESSED_TILES_DIR = os.path.join(GeneralConfigs.ROOT, 'data', 'processed_tiles_224')
    TILE_SIZE = 224  # should be divisible by downsample of reduced image, the easiest way is to set to be a power of 2
    REDUCED_LEVEL_TO_MEMORY = [3, 2]  # attempting to load according to order
    TARGET_MAG_POWER = 20
    MAG_ATTR = 'openslide.objective-power'
    TILE_NON_TISSUE_THRESHOLD = 0.5  # tiles with more non tissue percent than threshold are filtered
    OTSU_FILTER = {'reduced_img_factor': 0.8, 'attr_name': 'Background',
                   'color_palette': {'s': 10, 'otsu_val_factor': 1.1}}
    BLACK_FILTER = {'attr_name': 'Black', 'color_palette': [{'v': 50}, {'v': 200, 's': 25}]}
    PEN_FILTER = {'attr_name': 'Pen', 'attr_name_not_filtered': 'Pen_not_filtered', 'min_pen_tiles': 0.025,
                  'color_palette': get_pen_color_palette()}
    TISSUE_ATTR = 'Tissue'
    COLOR_NORM_SUCC = 'Normalized'
    COLOR_NORM_FAIL = 'Normalizing_Fail'
    COLOR_NORM_REF_IMG = os.path.join(GeneralConfigs.ROOT, 'src', 'preprocessing', 'color_norm_reference_image.png')
    ATTRS_TO_COLOR_MAP = {TISSUE_ATTR: 'pink', OTSU_FILTER['attr_name']: 'white', BLACK_FILTER['attr_name']: 'grey',
                          PEN_FILTER['attr_name']: 'green', PEN_FILTER['attr_name_not_filtered']: 'blue',
                          COLOR_NORM_FAIL: 'yellow'}


@dataclass
class TumorClassificationConfigs:
    TUMOR_EXPERIMENT_NAME = 'tumor_classifier'
    TUMOR_RUN_NAME = 'color_jitter'
    TUMOR_TRAINED_MODEL_PATH = os.path.join(GeneralConfigs.ROOT, 'models',
                                            f'{TUMOR_EXPERIMENT_NAME}_resnet50_10_epochs_{TUMOR_RUN_NAME}.ckpt')
    TUMOR_LABELED_TILES_DIR = os.path.join(GeneralConfigs.ROOT, 'data', 'tumor_labeled_tiles')
    TUMOR_SUMMARY_DF_PRED_MERGED_FILENAME = f'summary_df_pred_merged_{TUMOR_EXPERIMENT_NAME}_{TUMOR_RUN_NAME}.csv'
    TUMOR_THUMBNAIL_FILENAME = f'tumor_thumbnail_{TUMOR_RUN_NAME}.png'
    TUMOR_CLASS = 'TUMSTU'
    NON_TUMOR_CLASSES = ['STRMUS', 'ADIMUC']
    TUMOR_TEST_SIZE = 0.2
    TUMOR_VALID_SIZE = 0.1
    TUMOR_TRAINING_BATCH_SIZE = 16
    TUMOR_TRAINING_NUM_WORKERS = 16
    TUMOR_NUM_CLASSES = 3
    TUMOR_CLASS_TO_IND = {'ADIMUC': 0, 'STRMUS': 1, 'TUMSTU': 2}  # alphabetical order as in ImageFolder
    TUMOR_CLASS_TO_COLOR = {'ADIMUC': 'pink', 'STRMUS': 'pink', 'TUMSTU': 'red'}
    TUMOR_IND = TUMOR_CLASS_TO_IND['TUMSTU']
    TUMOR_INIT_LR = 1e-4
    TUMOR_NUM_EPOCHS = 10
    TUMOR_NUM_DEVICES = 2
    TUMOR_DEVICE = 'gpu'
    TUMOR_PREDICT_OUTPUT_PATH = os.path.join(GeneralConfigs.ROOT, 'data', 'tumor_classification',
                                             f'tumor_tiles_{TUMOR_RUN_NAME}_pred')
    TUMOR_INFERENCE_BATCH_SIZE = 64
    TUMOR_INFERENCE_NUM_WORKERS = 32


class SemanticSegConfigs:
    SS_EXPERIMENT_NAME = 'semantic_segmentation'
    SS_RUN_NAME = "sixth_try_wloss_aug_no_norm"
    SS_RUN_DESCRIPTION = """SS NCT, weight loss with 0.5 on BACK and 1.5 to NORM and TUM.
    Adding aug of random blurring and sharpening.
    Without normalization."""
    SS_RUN_OOD_NAME = f'OOD_IRCCS_validaiton_wloss'
    SS_OOD_RUN_DESCRIPTION = """third_try model with wloss validation on IRCSS dataset 150*150 tiles.
    With label missmatch."""
    SS_THUMBNAIL_FILENAME = f'ss_thumbnail_{SS_RUN_NAME}.png'
    SS_SUMMARY_DF_PRED_MERGED_FILENAME = f'summary_df_pred_merged_{SS_EXPERIMENT_NAME}_{SS_RUN_NAME}.csv'
    SS_LABELED_TILES_TRAIN_DIR = os.path.join(GeneralConfigs.ROOT, 'data', 'semantic_seg_tiles_NCT',
                                              'NCT-CRC-HE-100K')
    SS_LABELED_TILES_TEST_DIR = os.path.join(GeneralConfigs.ROOT, 'data', 'semantic_seg_tiles_NCT',
                                             'CRC-VAL-HE-7K')
    SS_PREDICT_OUTPUT_PATH = os.path.join(GeneralConfigs.ROOT, 'data', 'semantic_segmentation_results',
                                          f'ss_{SS_RUN_NAME}_processed_tiles_pred')
    SS_OOD_DATASET_DIR = os.path.join(GeneralConfigs.ROOT, 'data', 'semantic_seg_tiles_IRCCS', 'Result_Normalization')
    SS_OOD_DATASET_PREDICT_OUTPUT_PATH = os.path.join(GeneralConfigs.ROOT, 'data', 'semantic_segmentation_results',
                                                      f'ss_{SS_RUN_OOD_NAME}_pred')
    SS_VALID_SIZE = 0.05
    SS_TRAINING_BATCH_SIZE = 32
    SS_TRAINING_NUM_WORKERS = 16
    SS_INIT_LR = 1e-4
    SS_NUM_EPOCHS = 10
    SS_NUM_DEVICES = [0, ]
    SS_DEVICE = 'gpu'
    # alphabetical order as in ImageFolder (dicts preserve order in Python 3.7+)
    SS_CLASS_TO_IND = {'ADI': 0, 'BACK': 1, 'DEB': 2, 'LYM': 3, 'MUC': 4, 'MUS': 5, 'NORM': 6, 'STR': 7, 'TUM': 8}  # order does matter
    SS_CLASS_TO_WEIGHT = {'ADI': 1, 'BACK': 0.5, 'DEB': 1, 'LYM': 1, 'MUC': 1, 'MUS': 1, 'NORM': 1.5, 'STR': 1, 'TUM': 1.5}
    SS_OOD_CLASS_TRANSLATE = {'ADI': 'ADI', 'BACK': 'BACK', 'DEB': 'DEBRIS_MUCUS', 'LYM': 'LYM', 'MUC': 'DEBRIS_MUCUS',
                              'MUS': 'MUSC_STROMA', 'NORM': 'NORM', 'STR': 'MUSC_STROMA', 'TUM': 'TUM'}
    SS_CLASS_TO_COLOR = {'ADI': 'beige', 'BACK': 'silver', 'DEB': 'grey',
                         'LYM': 'yellow', 'MUC': 'chartreuse', 'MUS': 'orange',
                         'NORM': 'pink', 'STR': 'lavender', 'TUM': 'maroon'}
    SS_TUM_CLASS = 'TUM'
    SS_TRAINED_MODEL_PATH = os.path.join(GeneralConfigs.ROOT, 'models',
                                         f'{SS_EXPERIMENT_NAME}_resnet50_{SS_NUM_EPOCHS}_epochs_{SS_RUN_NAME}.ckpt')
    SS_INFERENCE_MODEL_PATH = os.path.join(GeneralConfigs.ROOT, 'models',
                                           f'{SS_EXPERIMENT_NAME}_resnet50_{SS_NUM_EPOCHS}_epochs_{SS_RUN_NAME}.ckpt')
    SS_INFERENCE_BATCH_SIZE = 256
    SS_INFERENCE_NUM_WORKERS = 5


class TumorRegressionConfigs:
    TR_EXPERIMENT_NAME = 'tumor_norm_distance_regression'
    TR_MIN_DIS_TO_TUM = 3
    TR_MIN_GROUP_SIZE = 5
    TR_RUN_NAME = f"2layers_loss_dropout50_dis{TR_MIN_DIS_TO_TUM}_g{TR_MIN_GROUP_SIZE}_aug_logy"
    TR_RUN_DESCRIPTION = f"""Simple resent50 backbone, 0.05 dropout for 10 epochs.
    Only connected components with size larger than {TR_MIN_GROUP_SIZE}. Tum_dis greater than {TR_MIN_GROUP_SIZE}.
    Separated patients (stratified) in train and test and in valid."""
    TR_LABEL_DF_PATH = os.path.join(GeneralConfigs.ROOT, 'data', 'tumor_regression_distances',
                                    'df_dis_g.csv')
    TR_PREDICT_OUTPUT_PATH = os.path.join(GeneralConfigs.ROOT, 'data', 'tumor_regression_results',
                                          f'tr_{TR_RUN_NAME}_test_pred')
    TR_TRAINED_MODEL_PATH = os.path.join(GeneralConfigs.ROOT, 'models', 'tumor_regression',
                                         f'tr_{TR_RUN_NAME}.ckpt')
    TR_SAMPLE_WEIGHT = False
    TR_DROPOUT_VALUE = 0.5
    TR_NUM_EPOCHS = 6
    TR_NUM_DEVICES = [0, ]
    TR_DEVICE = 'gpu'
    TR_TRAINING_BATCH_SIZE = 64
    TR_TRAINING_NUM_WORKERS = 8
    TR_TEST_SIZE = 0.2
    TR_VALID_SIZE = 0.05
    TR_INIT_LR = 1e-4


@dataclass
class ConfigsClass(GeneralConfigs, PreprocessingConfigs, TumorClassificationConfigs, SemanticSegConfigs, TumorRegressionConfigs):
    def __init__(self):
        set_global_configs(verbose=self.VERBOSE,
                           log_file_args=self.PROGRAM_LOG_FILE_ARGS,
                           log_importance=self.LOG_IMPORTANCE,
                           log_format=self.LOG_FORMAT,
                           random_seed=self.RANDOM_SEED,
                           tile_progress_log_freq=self.TILE_PROGRESS_LOG_FREQ)


Configs = ConfigsClass()


