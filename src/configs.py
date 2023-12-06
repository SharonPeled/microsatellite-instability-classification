from dataclasses import dataclass
import os
from pathlib import Path
from .preprocessing.pen_filter import get_pen_color_palette
from .general_utils import set_global_configs
import torch
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torchstain")
from datetime import datetime
from collections import defaultdict


@dataclass
class GeneralConfigs:
    RANDOM_SEED = 1234
    VERBOSE = 2  # 1 logs to LOG_FILE, 2 logs to console, 3 logs to both to file and console
    ROOT = Path(__file__).parent.parent.resolve()
    # DATA_FOLDER = '/mnt/data/users/sharonpe' if 'dev' not in str(ROOT) else os.path.join(ROOT, 'data')
    DATA_FOLDER = '/home/sharonpe/work/microsatellite-instability-classification/data'
    PROGRAM_LOG_FILE_ARGS = ['log.txt', 'a+']  # slide level log is in the slide dir. Use --bring-slide-logs to get all slide logs.
    LOG_IMPORTANCE = 1  # 0 (all), 1 or 2 (only high importance logs)
    LOG_FORMAT = {'format': '%(process)d  %(asctime)s  [%(name)s] - %(message)s', 'datefmt':'%d-%m-%y %H:%M:%S'}
    MLFLOW_SAVE_DIR = os.path.join(ROOT, 'models', 'mlruns')
    START_TIME = datetime.now().strftime('%d_%m_%Y_%H_%M')


@dataclass
class PreprocessingConfigs:
    TILE_SIZE = None  # should be divisible by downsample of reduced image, the easiest way is to set to be a power of 2
    PREPROCESS_RUN_NAME = f'{TILE_SIZE}'
    METADATA_JSON_FILENAME = f'metadata_{PREPROCESS_RUN_NAME}.json'
    SUMMARY_DF_FILENAME = f'summary_df_{PREPROCESS_RUN_NAME}.csv'
    THUMBNAIL_FILENAME = f'thumbnail_{PREPROCESS_RUN_NAME}.png'
    TILE_PROGRESS_LOG_FREQ = 1000  # report progress every process of x tiles (convenient for multiprocessing)
    LOAD_METADATA = True
    TO_MACENKO_NORMALIZE = False
    PREPROCESSING_DEVICE = 'cpu'
    # Assuming TCGA folder structure, where each slide is in a separate dir and the dir is named after the slide ID
    # SLIDES_DIR = os.path.join(GeneralConfigs.ROOT, 'data', 'slides')
    # PROCESSED_TILES_DIR = os.path.join(GeneralConfigs.ROOT, 'data', f'processed_tiles_{PREPROCESS_RUN_NAME}')
    SLIDES_DIR = os.path.join(GeneralConfigs.DATA_FOLDER, 'slides')
    PROCESSED_TILES_DIR = os.path.join(GeneralConfigs.DATA_FOLDER, f'processed_tiles_{PREPROCESS_RUN_NAME}')
    MANIFEST_PATH = os.path.join(GeneralConfigs.ROOT, 'data', 'slides', f'manifest_all_dx_slides_unlabeled_COAD_READ_UCEC_STAD_ESCA.tsv')
    SAMPLE_PROCESSED_TILES = {512: 1e10, 1024: 1000, 256: 1000}
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
class TileEmbeddingSavingConfigs:
    TS_EXPERIMENT_NAME = 'p100_balanced_both_loss_0_0_baseline'
    TS_ARTIFACT_DIR = os.path.join(GeneralConfigs.DATA_FOLDER, 'experiments_artifacts', TS_EXPERIMENT_NAME)
    TS_BATCH_SIZE = 512
    TS_NUM_WORKERS = 3

    TS_TILE_ENCODER_NAME = 'VIT_PRETRAINED_DINO'
    TS_PATH_DIR = os.path.join(GeneralConfigs.DATA_FOLDER, f'tile_embeddings_{TS_TILE_ENCODER_NAME}_p100')
    TS_DF_TILE_PATHS_PATH = os.path.join(GeneralConfigs.ROOT, 'data', 'subtype_classification',
                                         f'df_processed_tile_paths_512.csv')
    TS_COHORT_TO_IND = {'CRC': 0, 'STAD': 1, 'ESCA': 2, 'UCEC': 3}
    TS_COHORT_AWARE_DICT = {'num_cohorts': 4,
                         'num_heads_per_cohort': 6,
                         'num_blocks_per_cohort': 12,  # default is last blocks
                         'block_position': 'end',
                         'exclude_cohorts': [2, ],
                         # separate_query_per_block, separate_noisy_query, separate_query, 'one_hot_head',
                         # 'shared_query_separate_training'
                         'awareness_strategy': 'separate_attended_query_per_block',
                         'q_attention_type': '2_layered_tanh',  # linear, 2_layered_tanh
                         'q_attention_drop': 0.0,
                         'bias_matrices': None
                         }
    TS_KW_ARGS = {
                  'tile_encoder': TS_TILE_ENCODER_NAME,
                  'cohort_aware_dict': TS_COHORT_AWARE_DICT,
                  'pretrained_ckp_path': os.path.join(GeneralConfigs.ROOT, 'models', 'subtype_classification', 'dgx_SQ6B12_At2Lrelu_32k_4_dino_checkpoints', 'checkpoint.pth'),
                  # 'pretrained_ckp_path': "/home/sharonpe/microsatellite-instability-classification/data/subtype_classification/third_try_all_slides_16k_4_dino_checkpoints/checkpoint9.pth",
                  }


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
    SS_NUM_DEVICES = [1, ]
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
    SS_INFERENCE_NUM_WORKERS = 32


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


class SubtypeClassificationConfigs:
    SC_TILE_SIZE = 512
    SC_EXPERIMENT_NAME = 'SC_DISTILMIL_2'
    SC_FORMULATION = f'tile_backbone_sr_w16_5e4_400e_6b_3res'
    SC_RUN_NAME = f"{SC_FORMULATION}"
    SC_RUN_DESCRIPTION = f"""Labels are by bioportal.
    """
    SC_LABEL_DF_PATH = os.path.join(GeneralConfigs.ROOT, 'data', 'subtype_classification',
                                    'manifest_labeled_dx_molecular_subtype.tsv')
                                    # 'df_labels_msi_slides_and_other_msi_mss.tsv')
                                    # 'df_labels_cin_slides_and_other_cin_gs.tsv')
                                    # 'df_labels_cin_slides_and_other_cin_gs_2.tsv')
    SC_BACKBONE_ARTIFACT_DIR = os.path.join(GeneralConfigs.DATA_FOLDER, 'experiments_artifacts',
                                            'p100_balanced_both_loss_0_005_wm1_baseline')
    # f for fold to be replaced later
    SC_EXP_ARTIFACTS_DIR = os.path.join(GeneralConfigs.DATA_FOLDER, 'experiments_artifacts',
                                        f'{SC_EXPERIMENT_NAME}_{SC_RUN_NAME}', 'f')
    SC_USE_ARTIFACT_DIR = True
    SC_DF_TILE_EMBEDDINGS_PATH = os.path.join(GeneralConfigs.DATA_FOLDER,
                                              f'tile_embeddings', 'df_tile_embeddings.csv') # TODO: change this dummy variable
    SC_DF_TILE_PATHS_PATH = os.path.join(GeneralConfigs.ROOT, 'data', 'subtype_classification',
                                         f'df_processed_tile_paths_{SC_TILE_SIZE}.csv')
    SC_DF_TILE_PATHS_PATH_256 = os.path.join(GeneralConfigs.ROOT, 'data', 'subtype_classification',
                                             f'df_processed_tile_paths_256_reduced.csv')
    SC_DF_TILE_PATHS_PATH_512 = os.path.join(GeneralConfigs.ROOT, 'data', 'subtype_classification',
                                             f'df_processed_tile_paths_512.csv')
    SC_DF_TILE_PATHS_PATH_1024 = os.path.join(GeneralConfigs.ROOT, 'data', 'subtype_classification',
                                              f'df_processed_tile_paths_1024_reduced.csv')  # the labeled slides are full..
    SC_LABEL_COL = 'subtype'
    SC_TRAINED_MODEL_PATH = os.path.join(SC_EXP_ARTIFACTS_DIR, 'train', f'model_' + '{time}.ckpt')
    SC_TEST_PREDICT_OUTPUT_PATH = os.path.join(SC_EXP_ARTIFACTS_DIR, 'test')
    SC_TRAIN_PREDICT_OUTPUT_PATH = os.path.join(SC_EXP_ARTIFACTS_DIR, 'train')
    # SC_VALID_PREDICT_OUTPUT_PATH = os.path.join(GeneralConfigs.ROOT, 'data', 'subtype_classification',
    #                                             f'{SC_RUN_NAME}_pred', 'valid')
    SC_SSL_STATISTICS = {'HSV': os.path.join(GeneralConfigs.ROOT, 'data', 'subtype_classification',
                                             f'HSV_statistics_30_512.yaml'),
                         'HED': os.path.join(GeneralConfigs.ROOT, 'data', 'subtype_classification',
                                             f'HED_statistics_30_512.yaml'),
                         'LAB': os.path.join(GeneralConfigs.ROOT, 'data', 'subtype_classification',
                                             f'LAB_statistics_30_512.yaml')}
    SC_CROSS_VALIDATE = True  # num folds according to test size
    SC_CONTINUE_FROM_FOLD = 2  # 0 to 1/TEST_SIZE
    SC_SINGLE_FOLD = False  # 0 to 1/TEST_SIZE
    SC_Y_TO_BE_STRATIFIED = 'y_to_be_stratified'
    SC_CLASS_TO_IND = {'GS': 0, 'CIN': 1} # {'MSS': 0, 'MSI': 1} #
    SC_CLASS_WEIGHT = None #  {'GS': 770, 'CIN': 235}
    SC_COHORT_TO_IND = {'CRC': 0, 'STAD': 1, 'UCEC': 3}  # {'CRC': 0, 'STAD': 1, 'ESCA': 2, 'UCEC': 3}
    SC_EXCLUDE_COHORT_AWARENESS = {'ESCA': 2}
    SC_COHORT_WEIGHT = None # {('COAD', 'CIN'): 0.75, ('COAD', 'GS'): 2.25, ('ESCA', 'CIN'): 0.25, ('ESCA', 'GS'): 0.75, ('READ', 'CIN'): 0.75, ('READ', 'GS'): 2.25, ('STAD', 'CIN'): 0.25, ('STAD', 'GS'): 0.75, ('UCEC', 'CIN'): 0.25, ('UCEC', 'GS'): 0.75}
    # SC_COHORT_TUNE = None # ['COAD', 'READ']
    SC_TEST_ONLY = None
    SC_SAVE_TEST = True
    SC_SAVE_TRAIN = False
    SC_NUM_EPOCHS = 400
    SC_NUM_DEVICES = 1
    SC_NUM_NODES = 1
    SC_DEVICE = 'gpu'
    SC_TEST_BATCH_SIZE = 1
    SC_SAVE_CHECKPOINT_STEP_INTERVAL = None
    SC_VAL_STEP_INTERVAL = 1/2  # 2 times an epoch
    SC_TRAINING_BATCH_SIZE = 1  # accumulating gradients in MIL only
    SC_NUM_WORKERS = 3
    SC_TEST_SIZE = 0.3333
    SC_VALID_SIZE = 0  # not used if CV=True
    SC_INIT_LR = [1e-6,
                  5e-4]  # per part of the network, in order of the actual nn
    SC_TILE_SAMPLE_TRAIN = 1e10  # all tiles
    SC_TILE_SAMPLE_LAMBDA_TRAIN_TUNE = None
    SC_FROZEN_BACKBONE = False
    SC_ITER_TRAINING_WARMUP_WO_BACKBONE = 0
    SC_TILE_ENCODER = 'VIT_PRETRAINED_DINO'  # to load dino net for future classification - VIT_PRETRAINED_DINO
    COHORT_AWARE_DICT = {'num_cohorts': 4,
                         'num_heads_per_cohort': 6,
                         'num_blocks_per_cohort': 12,  # default is last blocks
                         'block_position': 'end',
                         'exclude_cohorts': list(SC_EXCLUDE_COHORT_AWARENESS.values()),
                         # separate_query_per_block, separate_noisy_query, separate_query, 'one_hot_head',
                         # 'shared_query_separate_training'
                         'awareness_strategy': 'separate_attended_query_per_block',
                         'q_attention_type': '2_layered_tanh', #  linear, 2_layered_tanh
                         'q_attention_drop': 0.0,
                         'bias_matrices': None
                         }
    # separate_head - each cohort allocated a head, head of other cohorts are zeroed
    # separate_query - each cohort allocated a query, query of other cohorts are used but not updates (no gradients)
    SC_KW_ARGS = {'one_hot_cohort_head': False, # single layer head, overrides n_nn_head
                  'calc_proportions_class_w': False,
                  'n_nn_head': {'num_layers': 1, 'dropout_value': 0.0},
                  'sep_cohort_w_loss': True,
                  'learnable_cohort_prior_type': None, # '*', # 0.1,  # initial prior value
                  'FoVs_augs_amounts': None,  # (0.15, 0.15),  # tuple of % from each FoVs to add
                  'tile_encoder': SC_TILE_ENCODER,
                  'cohort_aware_dict': COHORT_AWARE_DICT,
                  'pretrained_ckp_path': os.path.join(GeneralConfigs.ROOT, 'models', 'subtype_classification', 'dgx_SQ6B12_At2Lrelu_32k_4_dino_checkpoints', 'checkpoint.pth'),
                  # 'pretrained_ckp_path': "/home/sharonpe/microsatellite-instability-classification/data/subtype_classification/third_try_all_slides_16k_4_dino_checkpoints/checkpoint9.pth",
                  # 'pretrained_ckp_path': "/home/sharonpe/sandox/checkpoint_argus.pth",
                  'config_filepath': str(Path(__file__).resolve())
                  }
    # combined loss
    SC_COMBINED_LOSS_ARGS = {
        'cohort_loss_w': 0.0,
        'cohort_warmup': None,
        'slide_loss_w': 0.05,
        'slide_warmup': 1,
        'n_nn_cohort_head': {'num_layers': 3, 'dropout_value': 0.0},
        'n_nn_slide_head': {'num_layers': 3, 'dropout_value': 0.0}
    }
    SC_CT_MIL = {
        'combined_loss_args': SC_COMBINED_LOSS_ARGS,
        'vit_adapter_trainable_blocks': None,
        'attn_dim': 128,
        'num_bags': 6,
        'num_tiles_per_bag': None,
        'cumulative_batch_size': 16,
        'weight_decay': 1e-4,
        'grad_clip': 5,
        'num_res_blocks': 3
    }


class DINOConfigs:
    DN_TILE_SIZE = 512
    DN_EXPERIMENT_NAME = 'SC_fusion_dino'
    DN_FORMULATION = f'dgx_SQ6B12_At2Lrelu_32k'
    DN_RUN_NAME = f"{DN_FORMULATION}_4"
    DINO_DICT = {'FoVs_augs_amounts': (0.05, 0.15)}  # tuple of % from each FoVs to add
    USE_SLURM = True
    DN_OUT_DIM = 32768
    DN_BATCH_SIZE = 80
    DN_NUM_WORKERS = 10
    DN_NUM_MINI_EPOCHS = 10
    DN_NUM_EPOCHS = 10 * DN_NUM_MINI_EPOCHS
    CONTINUE_FROM_EPOCH = 0
    DN_NUM_GPUS_PER_NODE = 4  # num GPUS
    DN_NUM_NODES = 2
    DN_CPUS_PER_TASK = 15
    DN_MEM_PER_GPU = 60
    DN_TIMEOUT = 1440  # minutes
    DN_DEVICE = 'gpu'
    DN_PARTITION = 'work'
    DN_QOS = 'normal'
    DINO_BASIC_CMD_FLAGS = f'--arch fusion_cw --out_dim {DN_OUT_DIM} --momentum_teacher 0.999 ' + \
                           f'--batch_size_per_gpu {DN_BATCH_SIZE} ' + \
                           f'--epochs {DN_NUM_EPOCHS} --saveckp_freq 20 --num_workers {DN_NUM_WORKERS} ' + \
                           f'--seed {GeneralConfigs.RANDOM_SEED} ' + \
                           f'--output_dir {GeneralConfigs.ROOT}/models/subtype_classification/{DN_RUN_NAME}_dino_checkpoints ' + \
                           f'--warmup_teacher_temp_epochs 20 --teacher_temp 0.05 --warmup_teacher_temp 0.01  ' + \
                           f'--warmup_epochs 10 --min_lr  1e-6  --lr 0.0005  ' + \
                           f'--norm_last_layer False  ' + \
                           f'--local_crops_number 10  --freeze_last_layer 0 --use_fp16 False  ' + \
                           f'--global_crops_scale 0.25 1. --local_crops_scale 0.05 0.25 ' + \
                           f'--drop_path_rate 0.1   '
    DINO_SLURM_CMD_FLAGS = f'--num_gpus_per_node {DN_NUM_GPUS_PER_NODE} --nodes {DN_NUM_NODES} --cpus_per_task {DN_CPUS_PER_TASK} ' + \
                           f'--timeout {DN_TIMEOUT} --partition {DN_PARTITION} --qos {DN_QOS} --mem_per_gpu {DN_MEM_PER_GPU}'
    DINO_CMD_flags = DINO_BASIC_CMD_FLAGS if not USE_SLURM else DINO_BASIC_CMD_FLAGS + DINO_SLURM_CMD_FLAGS
    DN_RUN_DESCRIPTION = f"""DINO raw, single GPU, raw dataset and raw warmups.
    SC run name: {SubtypeClassificationConfigs.SC_RUN_NAME}
    Command:
    {DINO_CMD_flags}
            """


class VariantClassificationConfigs:
    VC_EXPERIMENT_NAME = 'VC_TILE_SSL_VIT'
    VC_FORMULATION = 'COAD_random'
    VC_RUN_NAME = f'{VC_FORMULATION}'
    VC_RUN_DESCRIPTION = f"""SSL_VIT - fill this
    """
    VC_TILE_SIZE = 512
    VC_COHORT = 'COAD'
    VC_LABEL_DF_PATH = os.path.join(GeneralConfigs.ROOT, 'data', 'variant_classification',
                                    # 'df_labels_dna.csv')
                                    'df_labels_rand.csv')
                                    # 'df_labels_gwas_COAD.csv')
                                    # 'df_labels_gwas_STAD.csv')
    VC_DF_TILE_PATHS_PATH = os.path.join(GeneralConfigs.ROOT, 'data', 'subtype_classification',
                                             f'df_processed_tile_paths_512.csv')
    VC_EXP_ARTIFACTS_DIR = SC_EXP_ARTIFACTS_DIR = os.path.join(GeneralConfigs.DATA_FOLDER, 'experiments_artifacts',
                                        f'{VC_EXPERIMENT_NAME}_{VC_RUN_NAME}', 'f')
    VC_TRAINED_MODEL_PATH = None
    VC_TEST_PREDICT_OUTPUT_PATH = os.path.join(VC_EXP_ARTIFACTS_DIR, 'test')
    VC_TRAIN_PREDICT_OUTPUT_PATH = None
    # VC_VALID_PREDICT_OUTPUT_PATH = os.path.join(GeneralConfigs.ROOT, 'data', 'variant_classification',
    #                                             f'{VC_RUN_NAME}_pred', 'valid')
    VC_SSL_STATISTICS = {'HSV': os.path.join(GeneralConfigs.ROOT, 'data', 'subtype_classification',
                                             f'HSV_statistics_30_512.yaml'),
                         'HED': os.path.join(GeneralConfigs.ROOT, 'data', 'subtype_classification',
                                             f'HED_statistics_30_512.yaml'),
                         'LAB': os.path.join(GeneralConfigs.ROOT, 'data', 'subtype_classification',
                                             f'LAB_statistics_30_512.yaml')}
    VC_CROSS_VALIDATE = True
    VC_CONTINUE_FROM_FOLD = 0
    VC_TEST_ONLY = None
    VC_SAVE_TEST = True
    VC_Y_TO_BE_STRATIFIED = None
    VC_CLASS_TO_IND = {'GT0': 0, 'GT1': 1, 'GT2': 2}
    VC_COHORT_TO_IND = {'COAD': 0, 'STAD': 1, 'UCEC': 3}  # {'CRC': 0, 'STAD': 1, 'ESCA': 2, 'UCEC': 3}
    VC_NUM_EPOCHS = 1
    VC_NUM_DEVICES = 1
    VC_NUM_NODES = 1
    VC_DEVICE = 'gpu'
    VC_TEST_BATCH_SIZE = 256
    VC_SAVE_CHECKPOINT_STEP_INTERVAL = None
    VC_VAL_STEP_INTERVAL = None  # 10 times an epoch
    VC_TRAINING_BATCH_SIZE = 256
    VC_NUM_WORKERS = 10
    VC_TEST_SIZE = 0.3333
    VC_VALID_SIZE = None
    VC_INIT_LR = [1e-6 * (VC_TRAINING_BATCH_SIZE/256),
                  1e-4 * (VC_TRAINING_BATCH_SIZE/256)]
    VC_ITER_TRAINING_WARMUP_WO_BACKBONE = 1000
    VC_TILE_SAMPLE_TRAIN = 1e10
    VC_FROZEN_BACKBONE = False
    VC_TILE_ENCODER = 'SSL_VIT_PRETRAINED'
    # sample stuff
    VC_SAMPLE_SNPS = None
    # permutation stuff
    VC_NUM_PERMUTATIONS = None
    VC_LAST_PERMUTATION = None


@dataclass
class ConfigsClass(GeneralConfigs, PreprocessingConfigs, TumorClassificationConfigs, SemanticSegConfigs,
                   TumorRegressionConfigs, SubtypeClassificationConfigs, VariantClassificationConfigs, DINOConfigs,
                   TileEmbeddingSavingConfigs):
    TASK_PREFIXES = ''

    def __init__(self):
        self.joined = {}

    def set_task_configs(self, task_prefix):
        if not isinstance(task_prefix, list):
            task_prefix = [task_prefix, ]
        self.TASK_PREFIXES = task_prefix
        common_configs = ['EXPERIMENT_NAME', 'RUN_NAME', 'RUN_DESCRIPTION', 'LABEL_DF_PATH', 'DF_TILE_PATHS_PATH',
                          'TRAINED_MODEL_PATH', 'CLASS_TO_IND', 'NUM_EPOCHS', 'NUM_DEVICES', 'DEVICE', 'TEST_BATCH_SIZE',
                          'SAVE_CHECKPOINT_STEP_INTERVAL', 'VAL_STEP_INTERVAL', 'TRAINING_BATCH_SIZE', 'NUM_WORKERS',
                          'TEST_SIZE', 'VALID_SIZE', 'INIT_LR', 'TILE_SAMPLE_TRAIN', 'SSL_STATISTICS',
                          'CROSS_VALIDATE', 'Y_TO_BE_STRATIFIED', 'TEST_ONLY', 'SAVE_TEST', 'SAVE_TRAIN',
                          'TEST_PREDICT_OUTPUT_PATH', 'TRAIN_PREDICT_OUTPUT_PATH',
                          'VALID_PREDICT_OUTPUT_PATH', 'COHORT_TO_IND', 'CONTINUE_FROM_FOLD', 'NUM_NODES',
                          'EXP_ARTIFACTS_DIR']
        for c in common_configs:
            for prefix in self.TASK_PREFIXES:
                task_c = f'{prefix}_{c}'
                if not hasattr(self, task_c):
                    continue
                self.joined[c] = getattr(self, task_c)
                break
        print(f"Starting experiment: {self.joined['EXPERIMENT_NAME']}.{self.joined['RUN_NAME']}")


Configs = ConfigsClass()


