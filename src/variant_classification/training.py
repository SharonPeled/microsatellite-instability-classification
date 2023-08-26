from src.variant_classification.init_task import init_task
from src.training_utils import train as train_general
from src.configs import Configs
from random import shuffle
from src.components.objects.Logger import Logger

def train():
    df, train_transform, test_transform, logger, callbacks, model = init_task()
    if Configs.VC_SAMPLE_SNPS is not None:
        df['full_y'] = df['y'].copy()
        num_snps = len(df['full_y'].iloc[0])
        indices = list(range(num_snps))
        shuffle(indices)
        chunk_size = Configs.VC_SAMPLE_SNPS
        for i, shift in enumerate(range(0, num_snps, chunk_size)):
            Logger.log(f'Sample number {i}.')
            df['y'] = df.full_y.apply(lambda y: y[indices[shift:shift + chunk_size]])
            df['snps_indices'] = str(indices[shift:shift + chunk_size])
            train_general(df, train_transform, test_transform, logger, callbacks, model)
    else:
        train_general(df, train_transform, test_transform, logger, callbacks, model)







