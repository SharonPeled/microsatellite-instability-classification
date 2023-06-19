from torchvision import transforms
from torch.utils.data import DataLoader
from src.components.datasets.ProcessedTileDataset import ProcessedTileDataset
import pytorch_lightning as pl
from torch.multiprocessing import set_start_method, set_sharing_strategy
from pytorch_lightning.loggers import MLFlowLogger
from ..configs import Configs
from src.components.objects.Logger import Logger
from src.components.models.TransferLearningClassifier import TransferLearningClassifier
import pandas as pd
from src.utils import train_test_valid_split_patients_stratified
from pytorch_lightning.callbacks import LearningRateMonitor
from src.components.objects.CustomWriter import CustomWriter
from glob import glob
import os


def set_worker_sharing_strategy(worker_id: int) -> None:
    set_sharing_strategy('file_system')


def train():
    set_sharing_strategy('file_system')
    set_start_method("spawn")
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),  # reverse 50% of images
        transforms.RandomVerticalFlip(),  # reverse 50% of images

        transforms.RandomApply([transforms.ColorJitter(brightness=0.1, contrast=0.4, saturation=0.4, hue=0.1)], p=0.75),
        transforms.RandomChoice([transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.25, 1)),
                                 transforms.RandomAdjustSharpness(sharpness_factor=2)], p=[0.15, 0.15]),
        transforms.Resize(224),
        transforms.ToTensor(),
        # MacenkoNormalizerTransform(Configs.COLOR_NORM_REF_IMG),  # gets tensor and output PIL ...
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        # MacenkoNormalizerTransform(Configs.COLOR_NORM_REF_IMG),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    df_labels = pd.read_csv(Configs.SC_LABEL_DF_PATH, sep='\t')
    df_labels = df_labels[df_labels[Configs.SC_LABEL_COL].isin(Configs.SC_CLASS_TO_IND.keys())]
    df_labels['slide_uuid'] = df_labels.slide_path.apply(lambda p: os.path.basename(os.path.dirname(p)))
    df_labels['y'] = df_labels[Configs.SC_LABEL_COL].apply(lambda s: Configs.SC_CLASS_TO_IND[s])
    df_labels['y_to_be_stratified'] = df_labels['y'].astype(str) + '_' + df_labels['cohort']
    df_train, df_valid, df_test = train_test_valid_split_patients_stratified(df_labels, y_col='y_to_be_stratified',
                                                                             test_size=Configs.SC_TEST_SIZE,
                                                                             valid_size=Configs.SC_VALID_SIZE,
                                                                             random_seed=Configs.RANDOM_SEED)

    tile_paths = glob(f"{Configs.PROCESSED_TILES_DIR}/**/*Tissue*.jpg", recursive=True)
    df_tiles = pd.DataFrame(tile_paths, columns=['tile_path'])
    df_tiles['slide_uuid'] = df_tiles.tile_path.apply(lambda p: os.path.basename(os.path.dirname(p)))
    df_train = df_train.merge(df_tiles, how='inner', on='slide_uuid')
    df_valid = df_valid.merge(df_tiles, how='inner', on='slide_uuid')
    df_test = df_test.merge(df_tiles, how='inner', on='slide_uuid')

    train_dataset = ProcessedTileDataset(df_labels=df_train, transform=train_transform)
    valid_dataset = ProcessedTileDataset(df_labels=df_valid, transform=test_transform)
    test_dataset = ProcessedTileDataset(df_labels=df_test, transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=Configs.SC_TRAINING_BATCH_SIZE, shuffle=True,
                              persistent_workers=True, num_workers=Configs.SC_TRAINING_NUM_WORKERS,
                              worker_init_fn=set_worker_sharing_strategy)
    valid_loader = DataLoader(valid_dataset, batch_size=Configs.SC_TRAINING_BATCH_SIZE, shuffle=False,
                              persistent_workers=True, num_workers=Configs.SC_TRAINING_NUM_WORKERS,
                              worker_init_fn=set_worker_sharing_strategy)
    test_loader = DataLoader(test_dataset, batch_size=Configs.SC_TRAINING_BATCH_SIZE, shuffle=False,
                             persistent_workers=True, num_workers=Configs.SC_TRAINING_NUM_WORKERS,
                             worker_init_fn=set_worker_sharing_strategy)

    model = TransferLearningClassifier(class_to_ind=Configs.SC_CLASS_TO_IND, learning_rate=Configs.SC_INIT_LR,
                                       class_to_weight=None)
    mlflow_logger = MLFlowLogger(experiment_name=Configs.SC_EXPERIMENT_NAME, run_name=Configs.SC_RUN_NAME,
                                 save_dir=Configs.MLFLOW_SAVE_DIR,
                                 artifact_location=Configs.MLFLOW_SAVE_DIR,
                                 log_model='all',
                                 tags={"mlflow.note.content": Configs.SC_RUN_DESCRIPTION})
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    Logger.log("Starting Training.", log_importance=1)
    trainer = pl.Trainer(devices=Configs.SC_NUM_DEVICES, accelerator=Configs.SC_DEVICE,
                         deterministic=True,
                         check_val_every_n_epoch=1,
                         callbacks=[lr_monitor],
                         enable_checkpointing=True,
                         logger=mlflow_logger,
                         num_sanity_val_steps=2,
                         max_epochs=Configs.SC_NUM_EPOCHS)
    trainer.fit(model, train_loader, valid_loader, ckpt_path=None)
    trainer.save_checkpoint(Configs.SC_TRAINED_MODEL_PATH)
    Logger.log("Done Training.", log_importance=1)
    Logger.log("Starting Test.", log_importance=1)
    trainer.test(model, dataloaders=test_loader)
    Logger.log(f"Done Test.", log_importance=1)
    Logger.log(f"Saving test results.", log_importance=1)
    # not nice code but whatever
    test_dataset.with_y = False
    model = TransferLearningClassifier.load_from_checkpoint(Configs.SC_TRAINED_MODEL_PATH,
                                                            class_to_ind=Configs.SC_CLASS_TO_IND, learning_rate=None,
                                                            class_to_weight=None)
    pred_writer = CustomWriter(output_dir=Configs.SC_PREDICT_OUTPUT_PATH,
                               write_interval="epoch", score_names=['y_pred', ],
                               dataset=test_dataset)
    test_loader = DataLoader(test_dataset, batch_size=Configs.SC_TRAINING_BATCH_SIZE, shuffle=False,
                             persistent_workers=True, num_workers=Configs.SC_TRAINING_NUM_WORKERS,
                             worker_init_fn=set_worker_sharing_strategy)
    trainer = pl.Trainer(accelerator=Configs.SC_DEVICE, devices=Configs.SC_NUM_DEVICES, callbacks=[pred_writer],
                         default_root_dir=Configs.SC_PREDICT_OUTPUT_PATH)
    trainer.predict(model, test_loader, return_predictions=False)







