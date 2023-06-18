from torch.utils.data import DataLoader
from ..configs import Configs
import pytorch_lightning as pl
from src.components.models.TissueClassifier import TissueClassifier
from torchvision import transforms
from src.components.objects.CustomWriter import CustomWriter
from src.components.transformers.MacenkoNormalizerTransform import MacenkoNormalizerTransform
from src.components.datasets.TumorTileDataset import TumorTileDataset
from ..utils import load_df_pred
from pytorch_lightning.loggers import MLFlowLogger
from src.components.objects.Logger import Logger
import os


def OOD_validation_ss_IRCCS():
    Logger.log(f"""Starting OOD_validation_ss_IRCCS: 
    Device: {Configs.SS_DEVICE}
    Workers: {Configs.SS_INFERENCE_NUM_WORKERS}
    Directory: {Configs.SS_OOD_DATASET_DIR}""", log_importance=1)
    transform = transforms.Compose([
        # transforms.Pad(224-150, fill=0, padding_mode='constant'),
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        MacenkoNormalizerTransform(Configs.COLOR_NORM_REF_IMG),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    ood_dataset = TumorTileDataset(Configs.SS_OOD_DATASET_DIR, img_extension='png',
                                       transform=transform, crop_and_agg=False)
    ood_loader = DataLoader(ood_dataset, batch_size=Configs.SS_INFERENCE_BATCH_SIZE, shuffle=False, persistent_workers=True,
                            num_workers=Configs.SS_INFERENCE_NUM_WORKERS)
    model = TissueClassifier.load_from_checkpoint(Configs.SS_INFERENCE_MODEL_PATH,
                                                  class_to_ind=Configs.SS_CLASS_TO_IND, learning_rate=None)
    Logger.log(f"Loading trained model: {Configs.SS_INFERENCE_MODEL_PATH}", log_importance=1)
    pred_writer = CustomWriter(output_dir=Configs.SS_OOD_DATASET_PREDICT_OUTPUT_PATH,
                               write_interval="epoch", score_names=list(Configs.SS_CLASS_TO_IND.keys()),
                               dataset=ood_dataset)
    trainer = pl.Trainer(accelerator=Configs.SS_DEVICE, devices=Configs.SS_NUM_DEVICES, callbacks=[pred_writer],
                         default_root_dir=Configs.SS_PREDICT_OUTPUT_PATH)
    trainer.predict(model, ood_loader, return_predictions=False)
    # loading the results
    df_pred = load_df_pred(pred_dir=Configs.SS_OOD_DATASET_PREDICT_OUTPUT_PATH, class_to_index=Configs.SS_CLASS_TO_IND)
    df_pred['y_true'] = df_pred.tile_path.apply(lambda p: os.path.basename(os.path.dirname(p))[3:])
    df_pred['y_pred_class'] = df_pred['y_pred'].apply(lambda x: list(Configs.SS_CLASS_TO_IND.keys())[x])
    df_pred['y_pred_class_t'] = df_pred['y_pred_class'].apply(lambda x: Configs.SS_OOD_CLASS_TRANSLATE[x])
    # logging metrics
    mlflow_logger = MLFlowLogger(experiment_name=Configs.SS_EXPERIMENT_NAME, run_name=Configs.SS_RUN_OOD_NAME,
                                 save_dir=Configs.MLFLOW_SAVE_DIR,
                                 artifact_location=Configs.MLFLOW_SAVE_DIR,
                                 tags={"pred_path": Configs.SS_OOD_DATASET_DIR,
                                       "mlflow.note.content": Configs.SS_OOD_RUN_DESCRIPTION})
    TissueClassifier.log_metrics(y_true=df_pred.y_true, y_pred=df_pred.y_pred_class_t, logits=None,
                                 target_names=set(Configs.SS_OOD_CLASS_TRANSLATE.values()), logger=mlflow_logger,
                                 dataset_str='OOD_IRCCS')








