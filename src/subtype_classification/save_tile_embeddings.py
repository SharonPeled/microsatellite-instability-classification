from src.configs import Configs
from src.components.models.CombinedLossSubtypeClassifier import CombinedLossSubtypeClassifier
import pandas as pd
import torch
from src.components.objects.Logger import Logger
from torchvision import transforms
from torch.utils.data import DataLoader
from src.components.datasets.ProcessedTileDataset import ProcessedTileDataset
from tqdm import tqdm
import os


def save_tile_trained_embeddings():
    test_transform = get_test_transform()
    for fold in os.listdir(Configs.TS_ARTIFACT_DIR):
        train_dir = os.path.join(Configs.TS_ARTIFACT_DIR, fold, 'train')
        test_dir = os.path.join(Configs.TS_ARTIFACT_DIR, fold, 'test')
        model_path = [os.path.join(train_dir, file) for file in os.listdir(train_dir) if file.endswith(".ckpt")]
        train_pred_path = [os.path.join(train_dir, file) for file in os.listdir(train_dir) if file.startswith("df_pred")]
        test_pred_path = [os.path.join(test_dir, file) for file in os.listdir(test_dir) if file.startswith("df_pred")]
        assert len(model_path) == 1 and len(train_pred_path) == 1 and len(test_pred_path) == 1
        model_path = model_path[0]
        train_pred_path = train_pred_path[0]
        test_pred_path = test_pred_path[0]
        model = init_model(model_path)
        for pred_path in [train_pred_path, test_pred_path]:
            slide_path_list = []
            df_pred = pd.read_csv(pred_path)
            with torch.no_grad():
                backbone = model.backbone.to(0)
                total = df_pred.slide_uuid.nunique()
                for i, (slide_uuid, df_s) in tqdm(enumerate(df_pred.groupby('slide_uuid')), total=total):
                    slide_tile_embed_list = []
                    dataset = ProcessedTileDataset(df_s, transform=test_transform,
                                                   cohort_to_index=Configs.TS_COHORT_TO_IND, pretraining=True)
                    loader = DataLoader(dataset, batch_size=Configs.TS_BATCH_SIZE,
                                        shuffle=False,
                                        num_workers=Configs.TS_NUM_WORKERS)
                    for x, c in loader:
                        x = x.to(0)
                        c = c.to(0)
                        slide_tile_embed_list.append(backbone(x, c).detach().cpu())
                    slide_tile_embed = torch.cat(slide_tile_embed_list)
                    path = os.path.join(os.path.dirname(pred_path), 'tile_embeddings', slide_uuid, 'tile_embeddings.tensor')
                    os.makedirs(os.path.dirname(path), exist_ok=True)
                    torch.save(slide_tile_embed, path)
                    df_s.to_csv(os.path.join(os.path.dirname(path), 'df_slide.csv'), index=False)
                    slide_path_list.append((slide_uuid, path))
                    # if i > 3:
                    #     break
            df_slide_paths = pd.DataFrame(slide_path_list, columns=['slide_uuid', 'tile_embeddings_path'])
            df_slide_paths.to_csv(os.path.join(os.path.dirname(pred_path), 'df_tile_embeddings.csv'), index=False)


def get_test_transform():
    test_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    return test_transform


def init_model(path):
    model = CombinedLossSubtypeClassifier.load_from_checkpoint(path, strict=False,
                                                               combined_loss_args=Configs.SC_COMBINED_LOSS_ARGS,
                                                               tile_encoder_name=Configs.SC_TILE_ENCODER,
                                                               class_to_ind=Configs.SC_CLASS_TO_IND,
                                                               learning_rate=Configs.SC_INIT_LR,
                                                               frozen_backbone=Configs.SC_FROZEN_BACKBONE,
                                                               class_to_weight=Configs.SC_CLASS_WEIGHT,
                                                               num_iters_warmup_wo_backbone=Configs.SC_ITER_TRAINING_WARMUP_WO_BACKBONE,
                                                               cohort_to_ind=Configs.SC_COHORT_TO_IND,
                                                               cohort_weight=Configs.SC_COHORT_WEIGHT,
                                                               **Configs.TS_KW_ARGS)
    Logger.log(f"Model successfully loaded from checkpoint!", log_importance=1)
    return model
