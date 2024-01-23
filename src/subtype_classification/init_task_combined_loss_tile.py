from src.configs import Configs
from src.components.objects.Logger import Logger
from src.training_utils import init_training_transforms, init_training_callbacks
from src.components.models.CombinedLossSubtypeClassifier import CombinedLossSubtypeClassifier
from src.training_utils import train as train_general
from src.subtype_classification.init_task_generic import init_data


def train():
    df, train_transform, test_transform, logger, callbacks, model = init_task()
    train_general(df, train_transform, test_transform, logger, callbacks, model)


def init_task():
    df_labels_merged_tiles = init_data()
    model = init_model()
    train_transform, test_transform = init_training_transforms()
    logger, callbacks = init_training_callbacks()
    return df_labels_merged_tiles, train_transform, test_transform, logger, callbacks, model


def init_model():
    if Configs.SC_TEST_ONLY is None:
        model = CombinedLossSubtypeClassifier(combined_loss_args=Configs.SC_COMBINED_LOSS_ARGS,
                                              tile_encoder_name=Configs.SC_TILE_ENCODER, class_to_ind=Configs.SC_CLASS_TO_IND,
                                  learning_rate=Configs.SC_INIT_LR, frozen_backbone=Configs.SC_FROZEN_BACKBONE,
                                  class_to_weight=Configs.SC_CLASS_WEIGHT,
                                  num_iters_warmup_wo_backbone=Configs.SC_ITER_TRAINING_WARMUP_WO_BACKBONE,
                                  cohort_to_ind=Configs.SC_COHORT_TO_IND, cohort_weight=Configs.SC_COHORT_WEIGHT,
                                  **Configs.SC_KW_ARGS)
        Logger.log(f"New Model successfully created!", log_importance=1)
    else:
        model = CombinedLossSubtypeClassifier.load_from_checkpoint(Configs.SC_TEST_ONLY,
                                                                   combined_loss_args=Configs.SC_COMBINED_LOSS_ARGS, tile_encoder_name=Configs.SC_TILE_ENCODER,
                                                       class_to_ind=Configs.SC_CLASS_TO_IND,
                                                       learning_rate=Configs.SC_INIT_LR,
                                                       frozen_backbone=Configs.SC_FROZEN_BACKBONE,
                                                       class_to_weight=Configs.SC_CLASS_WEIGHT,
                                                       num_iters_warmup_wo_backbone=Configs.SC_ITER_TRAINING_WARMUP_WO_BACKBONE,
                                                       cohort_to_ind=Configs.SC_COHORT_TO_IND,
                                                       cohort_weight=Configs.SC_COHORT_WEIGHT,
                                                       **Configs.SC_KW_ARGS)
        Logger.log(f"Model successfully loaded from checkpoint!", log_importance=1)
    return model