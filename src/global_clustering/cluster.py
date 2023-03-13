import numpy as np
import pandas as pd
from ..components.MacenkoNormalizerTransform import MacenkoNormalizerTransform
from ..components.TileDataset import TileDataset
from torch.utils.data import DataLoader
from ..configs import Configs
from glob import glob
from sklearn.cluster import MiniBatchKMeans
import torch
import random
import mlflow
import os
import matplotlib.pyplot as plt
from .utils import get_tile_and_slide_avg_clusters_alignment
import torchvision.models.vision_transformer

# TODO: go over and fix the non-code
# TODO: add mlflow expirement tracking
# TODO: create infastructure for evaluation the clustering (slilluete, in-image boundries)
# TODO: num clusteres
def cluster():
    experiment = mlflow.get_experiment_by_name(Configs.GC_EXPERIMENT_NAME)
    experiment_id = experiment.experiment_id if experiment is not None else \
        mlflow.create_experiment(Configs.GC_EXPERIMENT_NAME)
    # encoded_features_files = glob(f"{Configs.ENCODED_FEATURES_OUTPUT_DIR}/**/batch*", recursive=True)
    # random.shuffle(encoded_features_files)
    # batches = np.array_split(encoded_features_files, Configs.GC_BATCH_SIZE)
    for n_clusters in Configs.GC_NUM_CLUSTERS:
        with mlflow.start_run(experiment_id=experiment_id, run_name=Configs.GC_RUN_NAME+f'_{n_clusters}') as run:
            kmeans = MiniBatchKMeans(n_clusters=n_clusters, batch_size=Configs.GC_BATCH_SIZE)
            # for feature_batch_files in batches:
            #     X_batch = torch.concat([torch.load(features) for features in feature_batch_files])
            #     kmeans.partial_fit(X_batch)

            clusters = kmeans.predict()
            df_metadata = clusters.join(metadata_tiles)

            path = os.path.join(Configs.GC_OUTPUT_DIR, Configs.GC_RUN_NAME+f'_{n_clusters}.csv')
            df_metadata.to_csv(path, index=False)
            mlflow.log_artifact(path)

            silhouette = calc()
            mlflow.log_metric('silhouette', silhouette)

            # histogram of degree of each tile + avg
            # histogram of avg degree of tiles per slide + avg
            slide_avgs, tile_avgs = get_tile_and_slide_avg_clusters_alignment(df_metadata)
            fig, (ax1, ax2) = plt.subplots(2)
            ax1[0].hist(slide_avgs)
            ax1.title(f"Slide level percentages, avg={np.mean(slide_avgs).round(1)}")
            ax2[1].hist(tile_avgs)
            ax2.title(f"Tile level percentages, avg={np.mean(tile_avgs).round(1)}")
            mlflow.log_figure(fig, 'cluster_alignment_histograms.png')



