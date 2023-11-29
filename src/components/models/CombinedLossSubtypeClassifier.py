import torch
import torch.nn as nn
from src.components.models.SubtypeClassifier import SubtypeClassifier
from src.components.models.PretrainedClassifier import PretrainedClassifier
from src.components.objects.Logger import Logger
from src.components.datasets.ProcessedTileDataset import ProcessedTileDataset
from src.components.datasets.BagDataset import BagDataset
from src.training_utils import lr_scheduler_linspace_steps


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
        self.slide_to_ind_axu = None
        self.cohort_to_ind_aux = None
        self.automatic_optimization = False
        self.optimizers_list = []
        self.global_iter = 0
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
        if not isinstance(dataset, (ProcessedTileDataset, BagDataset)):
            dataset = dataset.datasets
        df_labels = dataset.df_labels
        self.cohort_to_ind_aux = {value: index for index, value in enumerate(df_labels.cohort.map(self.cohort_to_ind).unique())}
        self.slide_to_ind_axu = {value: index for index, value in enumerate(df_labels['slide_id'].unique())}
        if self.combined_loss_args['slide_loss_w'] is None:
            self.slide_head = nn.Identity().to(self.backbone)
        else:
            self.slide_head = self.create_aux_head(head_name='slide', head_out_size=len(self.slide_to_ind_axu)).to(self.backbone.device)
            self.loss_weights[2] = self.combined_loss_args['slide_loss_w']
        optimizer3 = torch.optim.Adam([
            {"params": [p for p in self.slide_head.parameters()], 'lr': self.learning_rate[1]},
        ])
        self.optimizers_list.append(optimizer3)
        self.create_loss_weights_schedulers()

    def create_loss_weights_schedulers(self):
        loader = self.trainer.train_dataloader
        tot_iters = len(loader) * self.trainer.max_epochs + self.trainer.max_epochs
        if self.combined_loss_args['cohort_warmup'] is not None:
            cohort_w_list = lr_scheduler_linspace_steps(lr_pairs=[(0.0, self.num_iters_warmup_wo_backbone),
                                                                  (0.0, self.combined_loss_args['cohort_warmup']), (self.loss_weights[1], -1),
                                                                  (self.loss_weights[1], None)],
                                                        tot_iters=tot_iters)
        else:
            cohort_w_list = lr_scheduler_linspace_steps(lr_pairs=[(0.0, self.num_iters_warmup_wo_backbone),
                                                                  (0.0, 0), (self.loss_weights[1], -1),
                                                                  (self.loss_weights[1], None)],
                                                        tot_iters=tot_iters)
        if self.combined_loss_args['slide_warmup'] is not None:
            slide_w_list = lr_scheduler_linspace_steps(lr_pairs=[(0.0, self.num_iters_warmup_wo_backbone),
                                                                 (0.0, self.combined_loss_args['slide_warmup']), (self.loss_weights[2], -1),
                                                                 (self.loss_weights[2], None)],
                                                       tot_iters=tot_iters)
        else:
            slide_w_list = lr_scheduler_linspace_steps(lr_pairs=[(0.0, self.num_iters_warmup_wo_backbone),
                                                                 (0.0, 0), (self.loss_weights[2], -1),
                                                                 (self.loss_weights[2], None)],
                                                       tot_iters=tot_iters)
        self.loss_weights = [[1.0 for _ in range(len(cohort_w_list))], cohort_w_list, slide_w_list]
        # normalizing the loss weights
        # for i in range(len(cohort_w_list)):
        #     # Calculate the sum of elements at index i across all sublists
        #     total = sum(self.loss_weights[j][i] for j in range(3))
        #     for j in range(3):
        #         self.loss_weights[j][i] /= total
        Logger.log(f'Weight loss initialized, Total steps: {tot_iters}', log_importance=1)

    def configure_optimizers(self):
        self.set_training_warmup()

        optimizer1 = torch.optim.Adam([
            {"params": [p for p in self.backbone.parameters()], 'lr': self.learning_rate[0]},
            {"params": [p for p in self.head.parameters()], 'lr': self.learning_rate[1]},
        ])
        optimizer2 = torch.optim.Adam([
            {"params": [p for p in self.cohort_head.parameters()], 'lr': self.learning_rate[1]},
        ])
        self.optimizers_list.append(optimizer1)
        self.optimizers_list.append(optimizer2)
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
        opt1, opt2, opt3 = self.optimizers_list
        s_aux = torch.tensor([self.slide_to_ind_axu[slide_id] for slide_id in slide_ids], device=x.device)
        c_aux = torch.tensor([self.cohort_to_ind_aux[cohort_ind.item()] for cohort_ind in c], device=x.device) # cohort_ind an cohort_aux_id are different

        scores, aux_c_scores, aux_s_scores = self.forward(x, c, s=s_aux, test=False)
        task_loss, aux_c_loss, aux_s_loss = self.loss(scores, aux_c_scores, aux_s_scores, y, c_aux, s_aux, tile_path)
        i = self.global_iter
        combined_loss = self.loss_weights[0][i]*task_loss - self.loss_weights[1][i]*aux_c_loss - self.loss_weights[2][i]*aux_s_loss
        opt1.zero_grad()
        self.manual_backward(combined_loss)
        opt1.step()

        _, aux_c_scores, aux_s_scores = self.forward(x, c, s=s_aux, test=False)
        _, aux_c_loss, aux_s_loss = self.loss(scores, aux_c_scores, aux_s_scores, y, c_aux, s_aux, tile_path)
        opt2.zero_grad()
        self.manual_backward(aux_c_loss, retain_graph=True)
        opt2.step()

        opt3.zero_grad()
        self.manual_backward(aux_s_loss)
        opt3.step()

        self.logger.experiment.log_metric(self.logger.run_id, "task_loss", task_loss.detach().cpu())
        self.logger.experiment.log_metric(self.logger.run_id, "cohort_loss", aux_c_loss.detach().cpu())
        self.logger.experiment.log_metric(self.logger.run_id, "slide_loss", aux_s_loss.detach().cpu())

        # print([self.loss_weights[j][i] for j in range(3)])
        # print(combined_loss.detach().cpu(), task_loss.detach().cpu(), aux_c_loss.detach().cpu(), aux_s_loss.detach().cpu())

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
        aux_c_loss = super(CombinedLossSubtypeClassifier, self).loss(aux_c_scores, y=c, c=None, tile_path=tile_path)
        aux_s_loss = super(CombinedLossSubtypeClassifier, self).loss(aux_s_scores, y=s, c=None, tile_path=tile_path)
        return task_loss, aux_c_loss, aux_s_loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        self.global_iter += 1



