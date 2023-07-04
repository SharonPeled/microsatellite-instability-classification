from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import pytorch_lightning as pl
from torch.multiprocessing import set_start_method
from pytorch_lightning.loggers import MLFlowLogger
from ..configs import Configs
from src.components.transformers.MacenkoNormalizerTransform import MacenkoNormalizerTransform
from ..general_utils import get_train_test_dataset
from src.components.objects.Logger import Logger
from src.components.models.TissueClassifier import TissueClassifier


def train():
    set_start_method("spawn")
    dataset = ImageFolder(Configs.TUMOR_LABELED_TILES_DIR)
    assert dataset.class_to_idx == Configs.TUMOR_CLASS_TO_IND

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),  # reverse 50% of images
        transforms.RandomVerticalFlip(),  # reverse 50% of images
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        MacenkoNormalizerTransform(Configs.COLOR_NORM_REF_IMG),
        transforms.ToPILImage(),
        transforms.ColorJitter(brightness=.25, contrast=.25, saturation=.25, hue=.5),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    valid_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        MacenkoNormalizerTransform(Configs.COLOR_NORM_REF_IMG),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    train_dataset, test_dataset = get_train_test_dataset(dataset, Configs.TUMOR_TEST_SIZE,
                                                               Configs.RANDOM_SEED,
                                                               train_transform,
                                                               valid_transform)
    train_dataset, valid_dataset = get_train_test_dataset(dataset, Configs.TUMOR_VALID_SIZE,
                                                                Configs.RANDOM_SEED,
                                                                train_transform,
                                                                valid_transform)
    Logger.log(f"""Created tumor classification datasets: {len(train_dataset)}, {len(valid_dataset)}, {len(test_dataset)}""",
               log_importance=1)

    train_loader = DataLoader(train_dataset, batch_size=Configs.TUMOR_TRAINING_BATCH_SIZE, shuffle=True, persistent_workers=True,
                              num_workers=Configs.TUMOR_TRAINING_NUM_WORKERS)
    valid_loader = DataLoader(valid_dataset, batch_size=Configs.TUMOR_TRAINING_BATCH_SIZE, shuffle=False, persistent_workers=True,
                              num_workers=Configs.TUMOR_TRAINING_NUM_WORKERS)
    test_loader = DataLoader(test_dataset, batch_size=Configs.TUMOR_TRAINING_BATCH_SIZE, shuffle=False, persistent_workers=True,
                             num_workers=Configs.TUMOR_TRAINING_NUM_WORKERS)
    model = TissueClassifier(class_to_ind=Configs.TUMOR_CLASS_TO_IND, learning_rate=Configs.TUMOR_INIT_LR)
    mlflow_logger = MLFlowLogger(experiment_name=Configs.TUMOR_EXPERIMENT_NAME, run_name=Configs.TUMOR_RUN_NAME,
                                 save_dir=Configs.MLFLOW_SAVE_DIR,
                                 artifact_location=Configs.MLFLOW_SAVE_DIR,
                                 log_model='all',
                                 tags={"trained_model_path": Configs.TUMOR_TRAINED_MODEL_PATH})
    Logger.log("Starting Training.", log_importance=1)
    trainer = pl.Trainer(devices=Configs.TUMOR_NUM_DEVICES, accelerator=Configs.TUMOR_DEVICE,
                         deterministic=True,
                         check_val_every_n_epoch=1,
                         enable_checkpointing=True,
                         logger=mlflow_logger,
                         num_sanity_val_steps=2,
                         max_epochs=Configs.TUMOR_NUM_EPOCHS)
    trainer.fit(model, train_loader, valid_loader, ckpt_path=None)
    Logger.log("Done Training.", log_importance=1)
    Logger.log("Starting Test.", log_importance=1)
    trainer.test(model, dataloaders=test_loader)
    Logger.log("Saving checkpoint.", log_importance=1)
    trainer.save_checkpoint(Configs.TUMOR_TRAINED_MODEL_PATH)
    Logger.log(f"Done, trained model save in {Configs.TUMOR_TRAINED_MODEL_PATH}.", log_importance=1)







