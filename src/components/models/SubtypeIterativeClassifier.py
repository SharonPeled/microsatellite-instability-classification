from src.components.models.SubtypeClassifier import SubtypeClassifier
from src.components.objects.Logger import Logger
from tqdm import tqdm
import torch
import numpy as np
import os
import datetime
from src.training_utils import lr_scheduler_linspace_steps
from src.components.datasets.ProcessedTileDataset import ProcessedTileDataset
from torch.utils.data import DataLoader


class SubtypeIterativeClassifier(SubtypeClassifier):
    def __init__(self, iter_args, tile_encoder_name,
                 class_to_ind, learning_rate, frozen_backbone, class_to_weight=None,
                 num_iters_warmup_wo_backbone=None, cohort_to_ind=None, cohort_weight=None, nn_output_size=None,
                 other_kwargs=None):
        self.save_hyperparameters()
        # mlflow log_parameter bug patch
        iter_args = eval(iter_args)
        class_to_ind = eval(class_to_ind)
        learning_rate = eval(learning_rate)
        class_to_weight = eval(class_to_weight)
        cohort_to_ind = eval(cohort_to_ind)
        other_kwargs = eval(other_kwargs)
        super(SubtypeIterativeClassifier, self).__init__(tile_encoder_name, class_to_ind, learning_rate,
                                                         frozen_backbone, class_to_weight,
                                                         num_iters_warmup_wo_backbone, cohort_to_ind,
                                                         cohort_weight, nn_output_size,
                                                         **other_kwargs)
        self.iter_args = iter_args
        self.full_df = None
        self.test_df = None 
        self.lr_list = None
        self.global_iter = None
        Logger.log(f"""SubtypeIterativeClassifier created.""", log_importance=1)

    def on_train_start(self):
        super(SubtypeIterativeClassifier, self).on_train_start()
        reduction_factor = self.iter_args['reduction_factor']
        tot_iters = sum([np.ceil(len(self.trainer.train_dataloader)*(reduction_factor**i))
                     for i in range(self.trainer.max_epochs)])
        self.lr_list = lr_scheduler_linspace_steps(lr_pairs=self.iter_args['lr_pairs'],
                                                   tot_iters=tot_iters)
        self.global_iter = 0
        Logger.log(f'Total steps: {tot_iters}', log_importance=1)

    def on_train_batch_end(self, outputs, batch, batch_idx):
        for param_group in self.trainer.optimizers[0].param_groups:
            param_group['lr'] = self.lr_list[self.global_iter]
        self.global_iter += 1

    def on_train_epoch_end(self) -> None:
        loader = self.trainer.train_dataloader
        dataset = loader.dataset
        if not isinstance(dataset, ProcessedTileDataset):
            dataset = dataset.datasets
        if self.full_df is None:
            self.full_df = dataset.df_labels.copy(deep=True)
            self.full_df.index = self.full_df.tile_path
        path = os.path.join(self.iter_args['save_path'], f"model_iter{self.current_epoch}.ckpt")
        self.trainer.save_checkpoint(path)
        Logger.log(f"""Model iter{self.current_epoch} saved in {path}""", log_importance=1)
        iter_model = SubtypeIterativeClassifier.load_from_checkpoint(path)
        device = self.device
        self = self.to('cpu')
        iter_model = iter_model.to(device)
        Logger.log(f"""Model iter{self.current_epoch} loaded.""", log_importance=1)
        Logger.log(f"""Starting train inference.""", log_importance=1)
        scores, tile_paths = self._apply_iter_model(loader, iter_model)
        self.full_df.loc[tile_paths, f'score{self.current_epoch}'] = scores
        dataset.apply_dataset_reduction(self.iter_args, scores)
        if self.iter_args.get('save_path', None) is not None and self.trainer.max_epochs-1 == self.current_epoch:
            time_str = datetime.datetime.now().strftime('%d_%m_%Y_%H_%M')
            os.makedirs(os.path.join(self.iter_args['save_path']), exist_ok=True)
            path = os.path.join(self.iter_args['save_path'], f"df_tile_with_scores_{time_str}.csv")
            self.full_df.to_csv(path, index=False)
            Logger.log(f"df_iter_with_scores saved in: {path}", log_importance=1)
        Logger.log(f"""Dataset reduced to size {len(dataset)}""", log_importance=1)
        del iter_model
        self = self.to(device)
    
    def _apply_iter_model(self, loader, iter_model):
        with torch.no_grad():
            scores = []
            tile_paths = []
            for i, b in tqdm(enumerate(loader), total=len(loader)):
                b = [elem.to(iter_model.device) if isinstance(elem, torch.Tensor) else elem for elem in b]
                b_scores = iter_model.general_loop(b, i)
                scores.append(b_scores[1]['scores'].numpy())
                tile_paths.append(b[-1])
            if scores[-1].ndim == 0:
                scores[-1] = np.array(scores[-1], ndmin=1)
            scores = np.concatenate(scores)
            scores = torch.sigmoid(torch.from_numpy(scores)).numpy()
            tile_paths = np.concatenate(tile_paths)
            return scores, tile_paths
        
    def on_train_end(self):
        pass

    def test_step(self, batch, batch_idx):
        return None

    def on_test_epoch_end(self):
        loader = self.trainer.test_dataloaders
        if not isinstance(loader, DataLoader):
            loader = loader[0]
        dataset = loader.dataset
        if not isinstance(dataset, ProcessedTileDataset):
            dataset = dataset.datasets
        self.test_df = dataset.df_labels.copy(deep=True)
        self.test_df.index = self.test_df.tile_path
        device = self.device
        self = self.to('cpu')
        for epoch in range(self.trainer.max_epochs):
            path = os.path.join(self.iter_args['save_path'], f"model_iter{epoch}.ckpt")
            iter_model = SubtypeIterativeClassifier.load_from_checkpoint(path)
            iter_model = iter_model.to(device)
            Logger.log(f"""Model iter{self.current_epoch} loaded.""", log_importance=1)
            scores, tile_paths = self._apply_iter_model(loader, iter_model)
            self.test_df.loc[tile_paths, f'score{epoch}'] = scores
            # dataset.apply_dataset_reduction(self.iter_args, scores)
        self.test_df.to_csv(os.path.join(self.iter_args['save_path'], f"test_df.csv")) # TODO: change this
        self.log_epoch_level_metrics(self.test_df, dataset_str='test')

    def _get_df_for_metric_logging(self, outputs):
        outputs = outputs.copy(deep=True)
        outputs['y_true'] = outputs.subtype.apply(lambda s: self.class_to_ind[s])
        outputs['cohort'] = outputs.cohort.apply(lambda c: self.cohort_to_ind[c])
        if self.iter_args['schedule_type'] == 'step':
            for epoch in range(self.trainer.max_epochs - 1):
                outputs = outputs.groupby('slide_uuid', as_index=False).apply(
                    lambda d: d.sort_values(f'score{epoch}', ascending=False)[
                              :int(len(d) * self.iter_args['reduction_factor'])]).reset_index(drop=True)
            outputs['CIN_score'] = outputs[f'score{self.trainer.max_epochs-1}']
            return outputs
