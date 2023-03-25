import torch
from torch.utils.data import DataLoader
from ..configs import Configs
import pytorch_lightning as pl
from ..components.TissueClassifier import TissueClassifier
from torchvision import transforms
from ..components.CustomWriter import CustomWriter
from ..components.MacenkoNormalizerTransform import MacenkoNormalizerTransform
from ..components.TCGATumorTileDataset import TCGATumorTileDataset
from ..utils import load_df_pred
from pytorch_lightning.loggers import MLFlowLogger
from ..utils import generate_confusion_matrix_figure
from sklearn.metrics import precision_recall_fscore_support
from ..components.Logger import Logger


def OOD_validation_tumor_TCGA():
    Logger.log(f"""Starting OOD_validation_tumor_TCGA: 
    Device: {Configs.SS_DEVICE}
    Workers: {Configs.SS_INFERENCE_NUM_WORKERS}
    Directory: {Configs.SS_OOD_DATASET_DIR}""", log_importance=1)
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        MacenkoNormalizerTransform(Configs.COLOR_NORM_REF_IMG),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    ood_dataset = TCGATumorTileDataset(Configs.SS_OOD_DATASET_DIR, img_extension='jpg',
                                       transform=transform, crop_and_agg=True)
    ood_loader = DataLoader(ood_dataset, batch_size=Configs.SS_INFERENCE_BATCH_SIZE, shuffle=False,
                            num_workers=Configs.SS_INFERENCE_NUM_WORKERS)
    model = TissueClassifier.load_from_checkpoint(Configs.SS_INFERENCE_MODEL_PATH,
                                                  class_to_ind=Configs.SS_CLASS_TO_IND, learning_rate=None)
    Logger.log(f"Loading trained model: {Configs.SS_INFERENCE_MODEL_PATH}", log_importance=1)
    pred_writer = CustomWriter(output_dir=Configs.SS_OOD_DATASET_PREDICT_OUTPUT_PATH,
                               write_interval="epoch", class_to_index=Configs.SS_CLASS_TO_IND, dataset=ood_dataset)
    trainer = pl.Trainer(accelerator=Configs.SS_DEVICE, devices=Configs.SS_NUM_DEVICES, callbacks=[pred_writer],
                         default_root_dir=Configs.SS_PREDICT_OUTPUT_PATH)
    trainer.predict(model, ood_loader, return_predictions=False)
    # loading the results
    df_pred = load_df_pred(pred_dir=Configs.SS_OOD_DATASET_PREDICT_OUTPUT_PATH, class_to_index=Configs.SS_CLASS_TO_IND)
    # transforming to binary case - tum / no tum
    df_pred['y_pred'] = (df_pred['y_pred'] == Configs.SS_CLASS_TO_IND[Configs.SS_TUM_CLASS]).astype(int)
    # agg predictions - if at least one of the smaller tiles is tum then the entire tile is tum
    df_pred = df_pred.groupby('tile_path', as_index=False).agg({
        'y_pred': lambda s: int(s.sum() > 0)
    })
    y_pred = df_pred['y_pred']
    y_true = torch.Tensor([1 for _ in range(len(df_pred))])  # OOD contains only tumors
    # logging metrics
    mlflow_logger = MLFlowLogger(experiment_name=Configs.SS_EXPERIMENT_NAME, run_name=Configs.SS_RUN_OOD_NAME,
                                 save_dir=Configs.MLFLOW_SAVE_DIR,
                                 artifact_location=Configs.MLFLOW_SAVE_DIR,
                                 tags={"pred_path": Configs.SS_OOD_DATASET_DIR,
                                       "mlflow.note.content": Configs.SS_OOD_RUN_DESCRIPTION})
    # confusion matrix
    fig = generate_confusion_matrix_figure(y_true, y_pred, list(Configs.SS_CLASS_TO_IND.keys()))
    mlflow_logger.experiment.log_figure(mlflow_logger.run_id, fig, f"confusion_matrix.png")
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary',
                                                               pos_label=1)
    mlflow_logger.experiment.log_metric(mlflow_logger.run_id, f"OOD_TUM_precision", precision)
    mlflow_logger.experiment.log_metric(mlflow_logger.run_id, f"OOD_TUM_recall", recall)
    mlflow_logger.experiment.log_metric(mlflow_logger.run_id, f"OOD_TUM_f1", f1)
    Logger.log(f"{precision, recall, f1}", log_importance=1)








