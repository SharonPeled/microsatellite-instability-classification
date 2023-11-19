import numpy as np
import pandas as pd
from src.components.models.TransferLearningClassifier import TransferLearningClassifier
from torchvision.models import resnet50
from timm.models.vision_transformer import VisionTransformer
from torchvision.models.resnet import Bottleneck, ResNet
from torch import nn
import torch
from src.components.objects.Logger import Logger
from sklearn.metrics import roc_auc_score, classification_report
from src.components.objects.RandStainNA.randstainna import RandStainNA
from torchvision import transforms
from src.components.objects.CheckpointEveryNSteps import CheckpointEveryNSteps
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import MLFlowLogger
from src.configs import Configs
from torch.utils.data import DataLoader
from src.components.datasets.ProcessedTileDataset import ProcessedTileDataset
from torch.multiprocessing import set_start_method, set_sharing_strategy
import pytorch_lightning as pl
from src.general_utils import save_pred_outputs, rm_tmp_files
from src.general_utils import train_test_valid_split_patients_stratified
from pytorch_lightning.plugins.environments import SLURMEnvironment
import signal
from PIL import Image
from copy import deepcopy
from src.components.models.FusionClassifier import CohortAwareVisionTransformer, MIL_CohortAwareVisionTransformer
from datetime import datetime
from pytorch_lightning.strategies import DDPStrategy
import os


def train(df, train_transform, test_transform, logger, callbacks, model, **kwargs):
    if Configs.joined['CROSS_VALIDATE']:
        cross_validate(df, train_transform, test_transform, logger, model, callbacks=callbacks, **kwargs)
    else:
        df_train, df_valid, df_test = train_test_valid_split_patients_stratified(df,
                                                                                 y_col=Configs.joined['Y_TO_BE_STRATIFIED'],
                                                                                 test_size=Configs.joined['TEST_SIZE'],
                                                                                 valid_size=Configs.joined['VALID_SIZE'],
                                                                                 random_seed=Configs.RANDOM_SEED)
        train_single_split(df_train, df_valid, df_test, train_transform, test_transform, logger, model,
                           callbacks=callbacks, **kwargs)
    Logger.log(f"Finished.", log_importance=1)


def cross_validate(df, train_transform, test_transform, mlflow_logger, model, callbacks, **kwargs):
    split_obj = train_test_valid_split_patients_stratified(df,
                                                           y_col=Configs.joined['Y_TO_BE_STRATIFIED'],
                                                           test_size=Configs.joined['TEST_SIZE'],
                                                           valid_size=0,
                                                           random_seed=Configs.RANDOM_SEED,
                                                           return_split_obj=True)
    cv_metrics = []
    for i, (train_inds, test_inds) in enumerate(split_obj):
        if Configs.joined['CONTINUE_FROM_FOLD'] and Configs.joined['CONTINUE_FROM_FOLD'] > i:
            Logger.log(f"Skipped Fold {i}", log_importance=1)
            continue
        Logger.log(f"Fold {i}", log_importance=1)
        df_train = df.iloc[train_inds].reset_index(drop=True)
        df_test = df.iloc[test_inds].reset_index(drop=True)
        # model.fold = i
        # model.iter_args['save_path'] = os.path.join(model.iter_args['save_path'], str(i))
        fitted_model = train_single_split(df_train, None, df_test, train_transform, test_transform, mlflow_logger, deepcopy(model),
                                          callbacks=callbacks, **kwargs)
        cv_metrics.append(fitted_model.metrics)
    metrics_dict = pd.DataFrame(cv_metrics).mean().add_suffix('_cv').to_dict()
    for metric_str, metric_val in metrics_dict.items():
        mlflow_logger.experiment.log_metric(mlflow_logger.run_id, metric_str, metric_val)


def train_single_split(df_train, df_valid, df_test, train_transform, test_transform, logger, model, callbacks=(),
                       **kwargs):
    assert not model.is_fit
    rm_tmp_files()
    Logger.log(f'Single train split started with:', log_importance=1)
    Logger.log(f'Train slides - {df_train.slide_uuid.unique()}', log_importance=1)
    if df_valid is not None:
        Logger.log(f'Valid slides - {df_valid.slide_uuid.unique()}', log_importance=1)
    Logger.log(f'Test slides - {df_test.slide_uuid.unique()}', log_importance=1)
    df_train_sampled = df_train.groupby('slide_uuid').apply(
        lambda slide_df: slide_df.sample(n=min(Configs.joined['TILE_SAMPLE_TRAIN'], len(slide_df)),
                                         random_state=Configs.RANDOM_SEED)).reset_index(drop=True)

    if "is_aug" in df_train.columns:
        df_test = df_test[~df_test.is_aug]
        if df_valid is not None:
            df_valid = df_valid[~df_valid.is_aug]

    train_dataset, valid_dataset, test_dataset, train_loader, valid_loader, test_loader = get_loader_and_datasets(
        df_train_sampled,
        df_valid, df_test, train_transform, test_transform, **kwargs)
    if hasattr(model, 'init_tile_weight'):
        model.init_tile_weight(train_dataset)
    Logger.log("Starting Training.", log_importance=1)
    Logger.log(f"Training loader size: {len(train_loader)}.", log_importance=1)
    trainer = pl.Trainer(devices=Configs.joined['NUM_DEVICES'], accelerator=Configs.joined['DEVICE'],
                         num_nodes=Configs.joined['NUM_NODES'],
                         deterministic=False,
                         val_check_interval=Configs.joined['VAL_STEP_INTERVAL'],
                         callbacks=callbacks,
                         enable_checkpointing=False,
                         logger=logger,
                         num_sanity_val_steps=2,
                         max_epochs=Configs.joined['NUM_EPOCHS'],
                         strategy=DDPStrategy(find_unused_parameters=True),
                         reload_dataloaders_every_n_epochs=1
                         )
    model.train_loader = train_loader
    model.test_loader = test_loader
    if Configs.joined['TEST_ONLY'] is None:
        if valid_loader is None:
            trainer.fit(model, train_loader, ckpt_path=None)
        else:
            trainer.fit(model, train_loader, valid_loader, ckpt_path=None)
        time_str = datetime.now().strftime('%d_%m_%Y_%H_%M')
        trainer.save_checkpoint(Configs.joined['TRAINED_MODEL_PATH'].format(time=time_str))
    Logger.log("Done Training.", log_importance=1)
    Logger.log("Starting Test.", log_importance=1)
    trainer.test(model, test_loader)
    Logger.log(f"Done Test.", log_importance=1)
    if Configs.joined['SAVE_TEST']:
        save_results(model, test_dataset, valid_dataset)
    else:
        Logger.log('Saving results is suppressed.', log_importance=1)
    return model


def save_results(model, test_dataset, valid_dataset, saving_raw=True):
    Logger.log(f"Saving test results...", log_importance=1)
    # since shuffle=False in test we can infer the batch_indices from batch_inx
    _, df_pred_path = save_pred_outputs(model.test_outputs, test_dataset, Configs.joined['TEST_BATCH_SIZE'],
                                        save_path=Configs.joined['TEST_PREDICT_OUTPUT_PATH'],
                                        class_to_ind=Configs.joined['CLASS_TO_IND'], saving_raw=saving_raw)
    # Logger.log(f"""Saved Test df_pred: {df_pred_path}""", log_importance=1)
    if valid_dataset is not None and model.valid_outputs is not None:
        _, df_pred_path = save_pred_outputs(model.valid_outputs, valid_dataset, Configs.joined['TEST_BATCH_SIZE'],
                                            save_path=Configs.joined['VALID_PREDICT_OUTPUT_PATH'],
                                            class_to_ind=Configs.joined['CLASS_TO_IND'], suffix='', saving_raw=saving_raw)
        # Logger.log(f"""Saved valid df_pred: {df_pred_path}""", log_importance=1)


def init_training_callbacks():
    mlflow_logger = MLFlowLogger(experiment_name=Configs.joined['EXPERIMENT_NAME'], run_name=Configs.joined['RUN_NAME'],
                                 save_dir=Configs.MLFLOW_SAVE_DIR,
                                 artifact_location=Configs.MLFLOW_SAVE_DIR,
                                 log_model='all',
                                 tags={"mlflow.note.content": Configs.joined['RUN_DESCRIPTION']})
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    checkpoint_callback = CheckpointEveryNSteps(prefix=Configs.joined['RUN_NAME'],
                                                save_step_frequency=Configs.joined['SAVE_CHECKPOINT_STEP_INTERVAL'])
    return mlflow_logger, [lr_monitor, checkpoint_callback]


def init_training_transforms():
    train_transform = transforms.Compose([
        transforms.RandomApply([transforms.RandomResizedCrop(224, scale=(0.75, 1.0), interpolation=Image.BICUBIC), ],
                               p=0.1),

        transforms.Resize(224), # to lower on computational costs.

        transforms.RandomHorizontalFlip(),  # reverse 50% of images
        transforms.RandomVerticalFlip(),  # reverse 50% of images

        transforms.RandomApply([transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.25, 1)), ], p=0.1),

        # transforms.RandomApply([transforms.RandomAdjustSharpness(sharpness_factor=2), ], p=0.1),

        transforms.RandomApply([transforms.Grayscale(num_output_channels=3), ], p=0.2),  # grayscale 20% of the images
        transforms.RandomApply([transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)], p=0.8),

        transforms.RandomApply([
            RandStainNA(yaml_file=Configs.joined['SSL_STATISTICS']['HSV'], std_hyper=0.01, probability=1.0, distribution="normal",
                        is_train=True),
            RandStainNA(yaml_file=Configs.joined['SSL_STATISTICS']['HED'], std_hyper=0.01, probability=1.0, distribution="normal",
                        is_train=True),
            RandStainNA(yaml_file=Configs.joined['SSL_STATISTICS']['LAB'], std_hyper=0.01, probability=1.0, distribution="normal",
                        is_train=True)],
            p=0.8),

        transforms.Resize(224),
        transforms.ToTensor(),

        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    return train_transform, test_transform


def lr_scheduler_linspace_steps(lr_pairs, tot_iters):
    left_iters = tot_iters - sum([num_iters for _, num_iters in lr_pairs[:-1] if num_iters > 1])
    lr_array = []
    for (lr, iters), (next_lr, _) in zip(lr_pairs, lr_pairs[1:]):
        if iters == -1:
            lr_array.append(np.linspace(lr, next_lr, int(left_iters)))
        elif iters < 1:
            lr_array.append(np.linspace(lr, next_lr, int(left_iters*iters)))
            left_iters -= int(left_iters*iters)
        else:
            lr_array.append(np.linspace(lr, next_lr, iters))
    return np.concatenate(lr_array)


def set_worker_sharing_strategy(worker_id: int) -> None:
    set_sharing_strategy('file_system')


def get_loader_and_datasets(df_train, df_valid, df_test, train_transform, test_transform, **kwargs):
    if kwargs.get('dataset_fn', None):
        dataset_fn = kwargs['dataset_fn']
    else:
        dataset_fn = ProcessedTileDataset
    train_dataset = dataset_fn(df_labels=df_train, transform=train_transform,
                               cohort_to_index=Configs.joined['COHORT_TO_IND'])
    test_dataset = dataset_fn(df_labels=df_test, transform=test_transform,
                              cohort_to_index=Configs.joined['COHORT_TO_IND'])

    train_loader = DataLoader(train_dataset, batch_size=Configs.joined['TRAINING_BATCH_SIZE'],
                              shuffle=kwargs.get('shuffle_train', True),
                              persistent_workers=True, num_workers=Configs.joined['NUM_WORKERS'],
                              worker_init_fn=set_worker_sharing_strategy, 
                              collate_fn=kwargs.get('custom_collate_fn', None))
    test_loader = DataLoader(test_dataset, batch_size=Configs.joined['TEST_BATCH_SIZE'], shuffle=False,
                             persistent_workers=True, num_workers=Configs.joined['NUM_WORKERS'],
                             worker_init_fn=set_worker_sharing_strategy,
                             collate_fn=kwargs.get('custom_collate_fn', None))
    if df_valid is None:
        return train_dataset, None, test_dataset, train_loader, None, test_loader

    valid_dataset = ProcessedTileDataset(df_labels=df_valid, transform=test_transform,
                                         cohort_to_index=Configs.joined['COHORT_TO_IND'])
    valid_loader = DataLoader(valid_dataset, batch_size=Configs.joined['TEST_BATCH_SIZE'], shuffle=False,
                              persistent_workers=True, num_workers=Configs.joined['NUM_WORKERS'],
                              worker_init_fn=set_worker_sharing_strategy,
                              collate_fn=kwargs.get('custom_collate_fn', None))

    return train_dataset, valid_dataset, test_dataset, train_loader, valid_loader, test_loader


def calc_safe_auc(y_true, y_score):
    try:
        return roc_auc_score(y_true, y_score)
    except Exception as e:
        print(e)
        return np.nan


def get_pretrained_url(key):
    URL_PREFIX = "https://github.com/lunit-io/benchmark-ssl-pathology/releases/download/pretrained-weights"
    model_zoo_registry = {
        "DINO_p16": "dino_vit_small_patch16_ep200.torch",
        "DINO_p8": "dino_vit_small_patch8_ep200.torch",
        "BT": "bt_rn50_ep200.torch",
        "MoCoV2": "mocov2_rn50_ep200.torch",
        "SwAV": "swav_rn50_ep200.torch"
    }
    pretrained_url = f"{URL_PREFIX}/{model_zoo_registry.get(key)}"
    return pretrained_url


def SLL_vit_small(pretrained, progress, key, **kwargs):
    patch_size = kwargs.get("patch_size", 16)
    model = VisionTransformer(
        img_size=224, patch_size=patch_size, embed_dim=384, num_heads=6, num_classes=0
    )
    if pretrained:
        pretrained_url = get_pretrained_url(key)
        verbose = model.load_state_dict(
            torch.hub.load_state_dict_from_url(pretrained_url, progress=progress)
        )
        Logger.log(verbose, log_importance=1)
    return model


def SLL_vit_small_cohort_aware(pretrained, progress, key, cohort_aware_dict, **vit_kwargs):
    model = CohortAwareVisionTransformer(
        cohort_aware_dict=cohort_aware_dict,
        img_size=224, patch_size=16, embed_dim=384, num_heads=6, num_classes=0, depth=12, **vit_kwargs
    )
    if pretrained:
        pretrained_url = get_pretrained_url(key)
        model.load_pretrained_model(torch.hub.load_state_dict_from_url(pretrained_url, progress=progress))
    return model


def DINO_vit_small_cohort_aware(ckp_path, cohort_aware_dict, load_MIL_version=False, **vit_kwargs):
    state_dict = torch.load(ckp_path, map_location="cpu")
    state_dict = state_dict['teacher']
    # remove `module.` prefix
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    # remove `backbone.` prefix induced by multicrop wrapper
    state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
    if not load_MIL_version:
        model = CohortAwareVisionTransformer(
            cohort_aware_dict=cohort_aware_dict,
            img_size=224, patch_size=16, embed_dim=384, num_heads=6, num_classes=0, depth=12
        )
        model.fc = nn.Identity()
    else:
        model = MIL_CohortAwareVisionTransformer(cohort_aware_dict=cohort_aware_dict, tile_embed_size=384,
                                                 embed_dim=384, num_heads=6, num_classes=0,
                                                 depth=12, dropout=(0, 0), **vit_kwargs)
    msg = model.load_state_dict(state_dict, strict=False)
    Logger.log(msg)
    return model


class ResNetTrunk(ResNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_features = self.fc.in_features
        del self.fc  # remove FC layer

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x


def SSL_resnet50(pretrained, progress, key, **kwargs):
    model = ResNetTrunk(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        pretrained_url = get_pretrained_url(key)
        verbose = model.load_state_dict(
            torch.hub.load_state_dict_from_url(pretrained_url, progress=progress)
        )
        print(verbose)
    return model


def load_headless_tile_encoder(tile_encoder_name, path=None, **kwargs):
    if tile_encoder_name == 'pretrained_resent_tile_based':
        tile_encoder = TransferLearningClassifier.load_from_checkpoint(path,
                                                                       class_to_ind=None,
                                                                       learning_rate=None).model
        layers = list(tile_encoder.children())
        if len(layers) == 1:
            layers = layers[0]
        num_filters = layers[-1].in_features
        tile_encoder = nn.Sequential(*layers[:-1])
        return tile_encoder, num_filters
    elif tile_encoder_name == 'pretrained_resent_imagenet':
        backbone = resnet50(weights="IMAGENET1K_V2")
        num_filters = backbone.fc.in_features
        layers = list(backbone.children())[:-1]
        layers.append(nn.Flatten())
        tile_encoder = nn.Sequential(*layers)
        return tile_encoder, num_filters
    elif tile_encoder_name == 'SSL_VIT_PRETRAINED':
        model = SLL_vit_small(pretrained=True, progress=False, key="DINO_p16", patch_size=16)
        return model, model.num_features
    elif tile_encoder_name == 'SSL_RESNET_PRETRAINED':
        model = SSL_resnet50(pretrained=True, progress=False, key="MoCoV2")
        return model, model.num_features
    elif tile_encoder_name == 'SSL_VIT_PRETRAINED_COHORT_AWARE':
        model = SLL_vit_small_cohort_aware(pretrained=True, progress=False, key="DINO_p16",
                                           cohort_aware_dict=kwargs['cohort_aware_dict'])
        return model, model.num_features
    elif tile_encoder_name == 'VIT_PRETRAINED_DINO':
        model = DINO_vit_small_cohort_aware(ckp_path=kwargs['pretrained_ckp_path'],
                                            **kwargs)
        return model, model.num_features



