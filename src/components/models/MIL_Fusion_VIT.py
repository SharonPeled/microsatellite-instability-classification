import torch
from torch import nn
from torch.nn.functional import softmax
from sklearn.metrics import roc_auc_score, classification_report
from torch.optim.lr_scheduler import ReduceLROnPlateau
from src.components.objects.Logger import Logger
import numpy as np
from torch.optim.lr_scheduler import StepLR
from src.training_utils import DINO_vit_small_cohort_aware, load_headless_tile_encoder
from src.components.models.PretrainedClassifier import PretrainedClassifier
import pandas as pd
from src.components.datasets.ProcessedTileDataset import ProcessedTileDataset
from torch.utils.data import DataLoader
from src.training_utils import set_worker_sharing_strategy
from src.components.objects.SCELoss import BSCELoss
from src.training_utils import calc_safe_auc
import torch.nn.functional as F
import traceback


class MIL_Fusion_VIT(PretrainedClassifier):
    def __init__(self, dataloader_df_cols, class_to_ind, cohort_to_ind, mil_model_params, tile_encoder_params,
                 learning_rate_params, tile_encoder_inference_params, mil_pooling_strategy,
                 max_tiles_mil,
                 num_iters_warmup_wo_backbone=None, **other_kwargs):
        mil_vit = DINO_vit_small_cohort_aware(ckp_path=mil_model_params[1],
                                              cohort_aware_dict=other_kwargs['cohort_aware_dict'],
                                              load_MIL_version=True)
        super().__init__(tile_encoder_name=None, class_to_ind=class_to_ind,
                         learning_rate=0.0, frozen_backbone=False, class_to_weight=None,
                         num_iters_warmup_wo_backbone=num_iters_warmup_wo_backbone, nn_output_size=None,
                         backbone=mil_vit, num_features=mil_vit.num_features, **other_kwargs)
        self.learning_rate_params = learning_rate_params
        self.max_tiles_mil = max_tiles_mil
        self.cohort_to_ind = cohort_to_ind
        self.dataloader_df_cols = dataloader_df_cols
        self.tile_encoder_inference_params = tile_encoder_inference_params
        self.other_kwargs = other_kwargs
        self.mil_pooling_strategy = mil_pooling_strategy
        self.class_to_ind = class_to_ind
        self.cohort_weight = None
        self.lr_list = []
        self.tile_encoder = DINO_vit_small_cohort_aware(ckp_path=tile_encoder_params[1],
                                                        cohort_aware_dict=other_kwargs['cohort_aware_dict'],
                                                        load_MIL_version=False)
        for param in self.tile_encoder.parameters():
            param.requires_grad = False
        Logger.log(f"""MIL_Fusion_VIT {mil_model_params[0]} created with tile encoder {tile_encoder_params[0]}.""",
                   log_importance=1)

    def on_train_start(self):
        super(MIL_Fusion_VIT, self).on_train_start()
        niter_per_ep = len(self.trainer.train_dataloader)
        self.lr_list = cosine_scheduler(base_value=self.learning_rate_params['base_value'],
                                        final_value=self.learning_rate_params['final_value'],
                                        niter_per_ep=niter_per_ep,
                                        epochs=self.trainer.max_epochs,
                                        warmup_epochs=self.learning_rate_params['warmup_epochs'],
                                        start_warmup_value=self.learning_rate_params['start_warmup_value'])

    def configure_optimizers(self):
        self.set_training_warmup()
        optimizer = torch.optim.AdamW(self.model.parameters())
        return optimizer

    def init_cohort_weight(self, train_dataset):
        if not self.other_kwargs.get('sep_cohort_w_loss', None):
            return
        df_tiles = train_dataset.df_labels
        tiles_per_cohort_subtype = df_tiles.groupby(['y', 'cohort'], as_index=False).tile_path.count()
        tiles_per_cohort = tiles_per_cohort_subtype.groupby('cohort')['tile_path'].sum()
        tiles_per_cohort_subtype['prop'] = tiles_per_cohort_subtype['tile_path'] / tiles_per_cohort_subtype[
            'cohort'].map(tiles_per_cohort)
        tiles_per_cohort_subtype['weight'] = 1 - tiles_per_cohort_subtype['prop']
        self.cohort_weight = tiles_per_cohort_subtype.groupby('cohort').apply(lambda df_c:
                                                                              df_c.sort_values(by='y').weight.values).to_dict()
        Logger.log(f"""MIL_Fusion_VIT update cohort weights: {self.cohort_weight}.""", log_importance=1)

    def forward(self, df_slides, num_tiles_per_slide):
        assert str(self.tile_encoder.device) != 'cpu'
        transform = self.tile_encoder_inference_params['train_transform'] if self.is_training else\
            self.tile_encoder_inference_params['test_transform']
        dataset = ProcessedTileDataset(df_labels=df_slides, transform=transform,
                                       cohort_to_index=self.cohort_to_ind, pretraining=True)
        loader = DataLoader(dataset, batch_size=self.tile_encoder_inference_params['batch_size'],
                            persistent_workers=True, num_workers=self.tile_encoder_inference_params['num_workers'],
                            worker_init_fn=set_worker_sharing_strategy)
        Logger.log(f'Slides: {df_slides.slide_uuid.unique()}', log_importance=1)
        encoded_tiles_batches = {}
        for i, (tiles, c) in enumerate(loader):
            tiles = tiles.to(self.tile_encoder.device)
            c = c.to(self.tile_encoder.device)
            encoded_tiles_batches[i] = self.tile_encoder(tiles, c).detach().cpu()
            if i % 20 == 0:
                Logger.log(f'Iter [{i}/{len(loader)}]', log_importance=1)
        tile_seqs, y, slide_id, patient_id, c = [], [], [], [], []
        lower_index = 0
        for num_rows in num_tiles_per_slide:
            upper_index = lower_index + num_rows - 1
            encoded_tiles_cat = concat_embeddings_per_slide(encoded_tiles_batches, lower_index, upper_index,
                                                            self.tile_encoder_inference_params['batch_size'])
            tile_df = df_slides[lower_index:upper_index+1]
            y.append(tile_df.iloc[0]['y'])
            slide_id.append(tile_df.iloc[0]['slide_id'])
            patient_id.append(tile_df.iloc[0]['patient_id'])
            c.append(self.cohort_to_ind[tile_df.iloc[0]['cohort']])
            tile_seqs.append(self.get_agg_embeddings(tile_df, encoded_tiles_cat))
            lower_index = upper_index + 1
        tile_seqs_batch = torch.stack(tile_seqs, dim=0)
        c = torch.tensor(c)
        scores = self.model(tile_seqs_batch.to(self.backbone.device), c.to(self.backbone.device))
        return scores.squeeze(), torch.tensor(y, device=scores.device), slide_id, patient_id, c

    def general_loop(self, batch, batch_idx):
        try:
            # batch are tensor of dfs
            df_slides = batch[0]
            num_tiles_per_slide = batch[1]
            scores, y, slide_id, patient_id, c = self.forward(df_slides, num_tiles_per_slide)
            loss = self.loss(scores, y, c)
            if self.is_training:
                current_lr = self.lr_list[self.global_step]
                for param_group in self.trainer.optimizers[0].param_groups:
                    param_group['lr'] = current_lr
            return {'loss': loss.cpu(), 'scores': scores.cpu(), 'y': y.cpu(), 'slide_id': slide_id, 'c': c,
                    'patient_id': patient_id}
        except Exception as e:
            traceback.print_exc()

    def get_agg_embeddings(self, tile_df, encoded_tiles_cat):
        # should return tensor of vector directly to MIL_fustion.
        num_tile_H = tile_df['tile_row'].max() + 1
        num_tile_W = tile_df['tile_col'].max() + 1

        slide_tensor = torch.full((num_tile_H, num_tile_W, self.tile_encoder.num_features), torch.nan)
        row_indices = tile_df['tile_row'].values
        col_indices = tile_df['tile_col'].values
        slide_tensor[row_indices, col_indices, :] = encoded_tiles_cat
        pooled_tiles = self.custom_convolution(slide_tensor)  # shape n_pooled_tiles, embed_dim
        if pooled_tiles.shape[0] > self.max_tiles_mil:
            indices = np.random.choice(list(range(pooled_tiles.shape[0])), size=self.max_tiles_mil, replace=False)
            pooled_tiles = pooled_tiles[indices]
        if pooled_tiles.shape[0] < self.max_tiles_mil:
            pooled_tiles = F.pad(pooled_tiles, (0, 0, 0, self.max_tiles_mil - pooled_tiles.shape[0]))
        return pooled_tiles

    def custom_convolution(self, input_tensor):
        """
        Perform custom convolution-like operation on the input tensor.

        Args:
            input_tensor (torch.Tensor): Input tensor of shape [channels, height, width].
            kernel_size (int): Size of the kernel.
            kernel_operation (callable): A function that takes a tensor of shape [channels, num_tiles_h, kernel_size, num_tiles_w, kernel_size]
                and returns a modified tensor of shape [channels, 1, 1].

        Returns:
            torch.Tensor: Output tensor after applying the custom convolution operation.
        """
        height, width, dim = input_tensor.shape
        kernel_size = self.mil_pooling_strategy['kernel_size']
        # Pad the input tensor if necessary
        pad_h = kernel_size - (height % kernel_size) if height % kernel_size != 0 else 0
        pad_w = kernel_size - (width % kernel_size) if width % kernel_size != 0 else 0
        if pad_h > 0 or pad_w > 0:
            input_tensor = torch.nn.functional.pad(input_tensor, (0, 0, 0, pad_w, 0, pad_h), value=torch.nan)
        height, width, dim = input_tensor.shape
        input_reshaped = input_tensor.reshape(height, width//kernel_size, kernel_size, dim)  # breaks the rows in kernel_size chunks
        input_reshaped = input_reshaped.permute(1, 0, 2, 3).reshape(-1, kernel_size**2, dim)
        if self.mil_pooling_strategy['type'] == 'max':
            input_reshaped = torch.where(input_reshaped.isnan(), float('-inf'), input_reshaped)
            pooled_tiles = input_reshaped.max(dim=1)[0]
            return pooled_tiles[(~(torch.isinf(pooled_tiles) & (pooled_tiles < 0))).all(axis=-1)]
        if self.mil_pooling_strategy['type'] == 'mean':
            pooled_tiles = torch.nanmean(input_reshaped, dim=1)
            return pooled_tiles[(~pooled_tiles.isnan()).all(axis=-1)]
        raise ValueError(f"pool_operation not recognize {self.pool_operation}.")

    def loss(self, scores, y, c):
        if self.cohort_weight is None or c is None:
            return super().loss(scores, y)
        y = y.to(scores.dtype)
        loss_list = []
        for c_name, c_ind in self.cohort_to_ind.items():
            scores_c = scores[c == c_ind]
            if scores_c.shape[0] == 0:
                continue
            y_c = y[c == c_ind]
            if len(self.cohort_weight[c_name]) == 2:
                pos_weight = self.cohort_weight[c_name][1] / \
                             self.cohort_weight[c_name][0]
            else:
                pos_weight = 1
            # loss_c = F.binary_cross_entropy_with_logits(scores_c, y_c, reduction='mean',
            #                                             pos_weight=torch.tensor(pos_weight))
            loss_c = BSCELoss.functional(scores_c, y_c, pos_weight=torch.tensor(pos_weight),
                                         alpha=2, beta=3)
            loss_list.append(loss_c * y_c.shape[0])
        return sum(loss_list) / y.shape[0]

    def log_epoch_level_metrics(self, outputs, dataset_str):
        scores = torch.concat([out["scores"] for out in outputs])
        if len(scores.shape) == 1:
            cin_scores = scores
        else:
            cin_scores = scores[:, 1]
        y_true = torch.concat([out["y"] for out in outputs]).numpy()
        cohort = torch.concat([out["c"] for out in outputs]).numpy()
        slide_id = np.concatenate([out["slide_id"] for out in outputs])
        patient_id = np.concatenate([out["patient_id"] for out in outputs])
        df = pd.DataFrame({
            "y_true": y_true,
            "cohort": cohort,
            "slide_id": slide_id,
            "patient_id": patient_id,
            "CIN_score": cin_scores
        })

        slide_cin_auc = calc_safe_auc(df.y_true, df.CIN_score)
        self.logger.experiment.log_metric(self.logger.run_id, f"{dataset_str}_slide_CIN_AUC",
                                          slide_cin_auc)
        self.metrics[f"{dataset_str}_slide_CIN_AUC"] = slide_cin_auc

        df_patient = df.groupby(['patient_id', 'cohort'], as_index=False).agg({
            'y_true': 'max',
            'CIN_score': 'mean'
        })
        patient_cin_auc = calc_safe_auc(df_patient.y_true, df_patient.CIN_score)
        self.logger.experiment.log_metric(self.logger.run_id, f"{dataset_str}_patient_CIN_AUC",
                                          patient_cin_auc)
        self.metrics[f"{dataset_str}_patient_CIN_AUC"] = patient_cin_auc

        df_patient_cohort = df_patient.groupby('cohort').apply(lambda df_group:
                                                           calc_safe_auc(df_group.y_true,
                                                                         df_group.CIN_score))
        for cohort, auc in df_patient_cohort.iteritems():
            self.logger.experiment.log_metric(self.logger.run_id, f"{dataset_str}_patient_{cohort}_CIN_AUC",
                                              auc)
            self.metrics[f"{dataset_str}_C{cohort}_AUC"] = auc


def concat_embeddings_per_slide(batches, lower_index, upper_index, batch_size):
    # Calculate the number of elements per batch
    elements_per_batch = batch_size

    # Initialize the concatenated results list
    concatenated_results = []

    # Calculate the batch indices for the lower and upper bounds
    lower_batch_index = lower_index // elements_per_batch
    upper_batch_index = upper_index // elements_per_batch

    # Calculate the element indices within the batches
    lower_element_index = lower_index % elements_per_batch
    upper_element_index = upper_index % elements_per_batch

    # Iterate through batches and concatenate results
    for batch_index, batch in batches.items():
        # Check if the batch is within the specified range
        if batch_index < lower_batch_index:
            continue
        elif batch_index > upper_batch_index:
            break

        # Determine the elements to select from this batch
        start_index = lower_element_index if batch_index == lower_batch_index else 0
        end_index = upper_element_index + 1 if batch_index == upper_batch_index else elements_per_batch

        # Select and append the elements from this batch
        selected_elements = batch[start_index:end_index]
        concatenated_results.append(selected_elements)

    return torch.cat(concatenated_results, 0)


def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule
