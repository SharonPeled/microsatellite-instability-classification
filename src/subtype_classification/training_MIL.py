from src.subtype_classification.init_task_MIL import init_task
from src.training_utils import train as train_general
from src.components.datasets.TileGroupDataset import TileGroupDataset


def train():
    df, train_transform, test_transform, logger, callbacks, model = init_task()
    train_general(df, train_transform, test_transform, logger, callbacks, model, dataset_fn=TileGroupDataset)



