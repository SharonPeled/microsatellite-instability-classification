from src.subtype_classification.init_task_generic import init_task
from src.training_utils import train as train_general


def train():
    df, train_transform, test_transform, logger, callbacks, model = init_task()
    train_general(df, train_transform, test_transform, logger, callbacks, model)









