from ..components.MacenkoNormalizerTransform import MacenkoNormalizerTransform
from ..components.ProcessedTileDataset import ProcessedTileDataset
from torch.utils.data import DataLoader
from ..configs import Configs
from ..components.CustomWriter import CustomWriter
import pytorch_lightning as pl
from ..components.TissueClassifier import TissueClassifier
from torchvision import transforms
from torch.utils.data.dataloader import default_collate


def my_collate(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return default_collate(batch)


def predict():
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        MacenkoNormalizerTransform(Configs.COLOR_NORM_REF_IMG),  # already norm
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    dataset = ProcessedTileDataset(Configs.PROCESSED_TILES_DIR, transform=transform)
    dataloader = DataLoader(dataset, shuffle=False, batch_size=Configs.TUMOR_INFERENCE_BATCH_SIZE,
                            persistent_workers=True,
                            num_workers=Configs.TUMOR_INFERENCE_NUM_WORKERS)
                            # collate_fn=my_collate)
    model = TissueClassifier.load_from_checkpoint(Configs.TUMOR_TRAINED_MODEL_PATH,
                                                  class_to_ind=Configs.TUMOR_CLASS_TO_IND, learning_rate=None)
    pred_writer = CustomWriter(output_dir=Configs.TUMOR_PREDICT_OUTPUT_PATH,
                               write_interval="epoch", class_to_index=Configs.TUMOR_CLASS_TO_IND, dataset=dataset)
    trainer = pl.Trainer(accelerator=Configs.TUMOR_DEVICE, devices=Configs.TUMOR_NUM_DEVICES, callbacks=[pred_writer],
                         default_root_dir=Configs.TUMOR_PREDICT_OUTPUT_PATH)
    trainer.predict(model, dataloader, return_predictions=False)




