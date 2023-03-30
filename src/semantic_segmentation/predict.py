from ..components.ProcessedTileDataset import ProcessedTileDataset
from torch.utils.data import DataLoader
from ..configs import Configs
import pytorch_lightning as pl
from ..components.TissueClassifier import TissueClassifier
from torchvision import transforms
from ..components.CustomWriter import CustomWriter
from ..components.MacenkoNormalizerTransform import MacenkoNormalizerTransform


def predict():
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        MacenkoNormalizerTransform(Configs.COLOR_NORM_REF_IMG),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    dataset = ProcessedTileDataset(Configs.PROCESSED_TILES_DIR, transform=transform)
    dataloader = DataLoader(dataset, shuffle=False, batch_size=Configs.SS_INFERENCE_BATCH_SIZE, persistent_workers=True,
                            num_workers=Configs.SS_INFERENCE_NUM_WORKERS)
    model = TissueClassifier.load_from_checkpoint(Configs.SS_INFERENCE_MODEL_PATH,
                                                  class_to_ind=Configs.SS_CLASS_TO_IND, learning_rate=None)
    pred_writer = CustomWriter(output_dir=Configs.SS_PREDICT_OUTPUT_PATH,
                               write_interval="epoch", class_to_index=Configs.SS_CLASS_TO_IND, dataset=dataset)
    trainer = pl.Trainer(accelerator=Configs.SS_DEVICE, devices=Configs.SS_NUM_DEVICES, callbacks=[pred_writer],
                         default_root_dir=Configs.SS_PREDICT_OUTPUT_PATH)
    trainer.predict(model, dataloader, return_predictions=False)




