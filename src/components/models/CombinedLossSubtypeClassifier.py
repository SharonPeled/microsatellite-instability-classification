import torch
import torch.nn as nn
from src.components.models.SubtypeClassifier import SubtypeClassifier
from src.components.models.PretrainedClassifier import PretrainedClassifier
from src.components.objects.Logger import Logger
from src.components.datasets.ProcessedTileDataset import ProcessedTileDataset


class CombinedLossSubtypeClassifier(SubtypeClassifier):
    def __init__(self, combined_loss_args, tile_encoder_name, class_to_ind, learning_rate, frozen_backbone, class_to_weight=None,
                 num_iters_warmup_wo_backbone=None, cohort_to_ind=None, cohort_weight=None, nn_output_size=None,
                 **other_kwargs):
        super(CombinedLossSubtypeClassifier, self).__init__(tile_encoder_name, class_to_ind, learning_rate, frozen_backbone,
                                                            class_to_weight, num_iters_warmup_wo_backbone,
                                                            cohort_to_ind, cohort_weight, nn_output_size,
                                                            **other_kwargs)
        self.loss_weights = [1.0, 0.0, 0.0]
        self.combined_loss_args = combined_loss_args
        if self.combined_loss_args['cohort_loss_w'] is None:
            self.cohort_head = nn.Identity()
        else:
            self.cohort_head = self.create_aux_head(head_name='cohort', head_out_size=len(cohort_to_ind))
            self.loss_weights[1] = self.combined_loss_args['cohort_loss_w']
        self.slide_head = None  # initialize with the training data
        self.slide_to_ind = None
        self.automatic_optimization = False
        Logger.log(f"""CombinedLossSubtypeClassifier created with: {self.combined_loss_args}.""", log_importance=1)

    def create_aux_head(self, head_name, head_out_size):
        num_layers = self.combined_loss_args[f'n_nn_{head_name}_head']['num_layers']
        dropout_value = self.combined_loss_args[f'n_nn_{head_name}_head']['dropout_value']
        if num_layers == 1:
            head = nn.Linear(self.num_features, head_out_size)
        else:
            head = nn.Sequential(
                *(PretrainedClassifier.head_layer_block(self.num_features, dropout_value=dropout_value)
                  for _ in range(num_layers - 1)),
                nn.Linear(self.num_features, head_out_size)
            )
        Logger.log(
            f"{head_name} head: {num_layers} layered head of size {self.num_features} with output size {head_out_size} created.",
            log_importance=1)
        return head

    def on_train_start(self):
        super(CombinedLossSubtypeClassifier, self).on_train_start()
        loader = self.trainer.train_dataloader
        dataset = loader.dataset
        if not isinstance(dataset, ProcessedTileDataset):
            dataset = dataset.datasets
        df_labels = dataset.df_labels
        unique_values = df_labels['slide_id'].unique()
        self.slide_to_ind = {value: index for index, value in enumerate(unique_values)}
        if self.combined_loss_args['slide_loss_w'] is None:
            self.slide_head = nn.Identity().to(self.head.device).to(self.backbone.device)
        else:
            self.slide_head = self.create_aux_head(head_name='slide', head_out_size=len(self.slide_to_ind)).to(self.backbone.device)
            self.loss_weights[2] = self.combined_loss_args['slide_loss_w']
        optimizer3 = torch.optim.Adam([
            {"params": [p for p in self.slide_head.parameters()], 'lr': self.learning_rate[1]},
        ])
        self.optimizers.append(optimizer3)
        self.loss_weights = [w/sum(self.loss_weights) for w in self.loss_weights]

    def configure_optimizers(self):
        self.set_training_warmup()

        optimizer1 = torch.optim.Adam([
            {"params": [p for p in self.backbone.parameters()], 'lr': self.learning_rate[0]},
            {"params": [p for p in self.head.parameters()], 'lr': self.learning_rate[1]},
        ])
        optimizer2 = torch.optim.Adam([
            {"params": [p for p in self.cohort_head.parameters()], 'lr': self.learning_rate[1]},
        ])
        return optimizer1, optimizer2

    def general_loop(self, batch, batch_idx, test=False):
        # try:
        #     lr = self.trainer.lr_scheduler_configs[0].scheduler.optimizer.param_groups[0]["lr"]
        #     self.logger.experiment.log_metric(self.logger.run_id, f"lr", lr)
        # except:
        #     pass
        x, c, y, slide_ids, patient_id, tile_path = batch
        if test:
            scores = self.forward(x, c, s=None, test=True)
            loss = torch.tensor(-1)
            return loss, {'loss': loss.detach().cpu(), 'c': c.detach().cpu(),
                          'scores': scores.detach().cpu(), 'y': y, 'slide_id': slide_ids, 'patient_id': patient_id,
                          'tile_path': tile_path}
        opt1, opt2, opt3 = self.optimizers()
        s = torch.tensor([self.slide_to_ind[slide_id] for slide_id in slide_ids], device=x.device)
        scores, aux_c_scores, aux_s_scores = self.forward(x, c, s=s, test=False)
        task_loss, aux_c_loss, aux_s_loss = self.loss(scores, aux_c_scores, aux_s_scores, y, c, s, tile_path)

        combined_loss = self.loss_weights[0]*task_loss - self.loss_weights[1]*aux_c_loss - self.loss_weights[2]*aux_s_loss
        opt1.zero_grad()
        self.manual_backward(combined_loss)
        opt1.step()

        opt2.zero_grad()
        self.manual_backward(aux_c_loss)
        opt2.step()

        opt3.zero_grad()
        self.manual_backward(aux_s_loss)
        opt3.step()
        return combined_loss, {'loss': combined_loss.detach().cpu(), 'c': c.detach().cpu(),
                      'scores': scores.detach().cpu(), 'y': y, 'slide_id': slide_ids, 'patient_id': patient_id,
                      'tile_path': tile_path}

    def forward(self, x, c, s, test=False):
        x_embed = self.backbone(x, c)
        task_scores = self.head(x_embed).squeeze()
        if test:
            return task_scores
        aux_c_scores = self.cohort_head(x_embed).squeeze()
        aux_s_scores = self.slide_head(x_embed).squeeze()
        return task_scores, aux_c_scores, aux_s_scores

    def loss(self, scores, aux_c_scores, aux_s_scores, y, c, s, tile_path):
        task_loss = super(CombinedLossSubtypeClassifier, self).loss(scores, y, c=None, tile_path=tile_path)
        aux_c_loss = super(CombinedLossSubtypeClassifier, self).loss(scores, y=c, c=None, tile_path=tile_path)
        aux_s_loss = super(CombinedLossSubtypeClassifier, self).loss(scores, y=s, c=None, tile_path=tile_path)
        print(round(task_loss.item(),2), round(aux_c_loss.item(),2 ), round(aux_s_loss.item(),2 ))
        return task_loss, aux_c_loss, aux_s_loss



