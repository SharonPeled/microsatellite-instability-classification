from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import pytorch_lightning as pl
from torch.multiprocessing import Pool, set_start_method, set_sharing_strategy
from pytorch_lightning.loggers import TensorBoardLogger, MLFlowLogger
from ..configs import Configs
from src.utils import get_train_test_dataset
from ..components.Logger import Logger
from ..components.TissueClassifier import TissueClassifier
from ..components.MacenkoNormalizerTransform import MacenkoNormalizerTransform


def set_worker_sharing_strategy(worker_id: int) -> None:
    set_sharing_strategy('file_system')


def train():
    set_sharing_strategy('file_system')
    set_start_method("spawn")

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),  # reverse 50% of images
        transforms.RandomVerticalFlip(),  # reverse 50% of images
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        MacenkoNormalizerTransform(Configs.COLOR_NORM_REF_IMG),  # gets tensor and output PIL ...
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        MacenkoNormalizerTransform(Configs.COLOR_NORM_REF_IMG),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    train_val_dataset = ImageFolder(Configs.SS_LABELED_TILES_TRAIN_DIR, transform=train_transform)
    test_dataset = ImageFolder(Configs.SS_LABELED_TILES_TEST_DIR, transform=test_transform)
    assert train_val_dataset.class_to_idx == Configs.SS_CLASS_TO_IND and \
           test_dataset.class_to_idx == Configs.SS_CLASS_TO_IND

    if Configs.SS_VALID_SIZE > 0:
        train_dataset, valid_dataset = get_train_test_dataset(train_val_dataset, Configs.SS_VALID_SIZE,
                                                              Configs.RANDOM_SEED)
        Logger.log(f"Created semantic segmentation datasets: \
        {len(train_dataset)}, {len(valid_dataset)}, {len(test_dataset)}",
                   log_importance=1)
        valid_loader = DataLoader(valid_dataset, batch_size=Configs.SS_TRAINING_BATCH_SIZE, shuffle=False,
                                  num_workers=Configs.SS_TRAINING_NUM_WORKERS,
                                  worker_init_fn=set_worker_sharing_strategy)
    else:
        train_dataset = train_val_dataset
        Logger.log(f"Created semantic segmentation datasets: \
                {len(train_dataset)}, {len(test_dataset)}",
                   log_importance=1)
    train_loader = DataLoader(train_dataset, batch_size=Configs.SS_TRAINING_BATCH_SIZE, shuffle=True,
                              num_workers=Configs.SS_TRAINING_NUM_WORKERS, worker_init_fn=set_worker_sharing_strategy)
    test_loader = DataLoader(test_dataset, batch_size=Configs.SS_TRAINING_BATCH_SIZE, shuffle=False,
                             num_workers=Configs.SS_TRAINING_NUM_WORKERS, worker_init_fn=set_worker_sharing_strategy)

    model = TissueClassifier(class_to_ind=Configs.SS_CLASS_TO_IND, learning_rate=Configs.SS_INIT_LR,
                             class_to_weight=Configs.SS_CLASS_TO_WEIGHT)
    mlflow_logger = MLFlowLogger(experiment_name=Configs.SS_EXPERIMENT_NAME, run_name=Configs.SS_RUN_NAME,
                                 save_dir=Configs.MLFLOW_SAVE_DIR,
                                 artifact_location=Configs.MLFLOW_SAVE_DIR,
                                 log_model='all',
                                 tags={"trained_model_path": Configs.SS_TRAINED_MODEL_PATH,
                                       "mlflow.note.content": Configs.SS_RUN_DESCRIPTION})
    Logger.log("Starting Training.", log_importance=1)
    trainer = pl.Trainer(devices=Configs.SS_NUM_DEVICES, accelerator=Configs.SS_DEVICE,
                         deterministic=True,
                         check_val_every_n_epoch=1,
                         enable_checkpointing=True,
                         logger=mlflow_logger,
                         num_sanity_val_steps=2,
                         max_epochs=Configs.SS_NUM_EPOCHS)
    if Configs.SS_VALID_SIZE > 0:
        trainer.fit(model, train_loader, valid_loader, ckpt_path=None)
    else:
        trainer.fit(model, train_loader, ckpt_path=None)
    Logger.log("Done Training.", log_importance=1)
    Logger.log("Starting Test.", log_importance=1)
    trainer.test(model, dataloaders=test_loader)
    Logger.log("Saving checkpoint.", log_importance=1)
    trainer.save_checkpoint(Configs.SS_TRAINED_MODEL_PATH)
    Logger.log(f"Done, trained model save in {Configs.SS_TRAINED_MODEL_PATH}.", log_importance=1)







