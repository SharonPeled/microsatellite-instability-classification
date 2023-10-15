from src.components.models.SubtypeClassifier import SubtypeClassifier
from src.components.objects.Logger import Logger
from tqdm import tqdm
import torch
import numpy as np
import os
import datetime


class SubtypeIterativeClassifier(SubtypeClassifier):
    def __init__(self, iter_args, tile_encoder_name,
                 class_to_ind, learning_rate, frozen_backbone, class_to_weight=None,
                 num_iters_warmup_wo_backbone=None, cohort_to_ind=None, cohort_weight=None, nn_output_size=None,
                 **other_kwargs):
        super(SubtypeIterativeClassifier, self).__init__(tile_encoder_name, class_to_ind, learning_rate,
                                                         frozen_backbone, class_to_weight,
                                                         num_iters_warmup_wo_backbone, cohort_to_ind,
                                                         cohort_weight, nn_output_size,
                                                         **other_kwargs)
        self.iter_args = iter_args
        self.train_loader = None
        self.full_df = None
        Logger.log(f"""SubtypeIterativeClassifier created.""", log_importance=1)

    def on_train_epoch_end(self) -> None:
        assert self.train_loader is not None
        if self.full_df is None:
            self.full_df = self.train_loader.dataset.df_labels.copy(deep=True)
            self.full_df.index = self.full_df.tile_path
        Logger.log(f"""Starting train inference.""", log_importance=1)
        with torch.no_grad():
            scores = []
            for i, b in tqdm(enumerate(self.train_loader), total=len(self.train_loader)):
                b = [elem.to(self.device) if isinstance(elem, torch.Tensor) else elem for elem in b]
                b_scores = self.general_loop(b, i)
                scores.append(b_scores[1]['scores'].numpy())
            scores = np.concatenate(scores)
            self.full_df.loc[self.train_loader.dataset.df_labels.tile_path.values,
                             f'score{self.current_epoch}'] = scores
        self.train_loader.dataset.apply_dataset_reduction(self.iter_args, scores)
        if self.iter_args.get('save_path', None) is not None and self.trainer.max_epochs-1 == self.current_epoch:
            time_str = datetime.datetime.now().strftime('%d_%m_%Y_%H_%M')
            self.full_df.to_csv(os.path.join(self.iter_args['save_path'], f"df_tile_with_scores_{time_str}.csv"), index=False)
        Logger.log(f"""Dataset reduced to size {len(self.train_loader.dataset)}""", log_importance=1)

    def on_train_end(self):
        pass
