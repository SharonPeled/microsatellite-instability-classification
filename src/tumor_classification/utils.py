from sklearn.model_selection import StratifiedShuffleSplit
from ..components.MacenkoNormalizerTransform import MacenkoNormalizerTransform
from ..components.SubDataset import SubDataset
from torch.utils.data import Subset


def get_train_test_valid_dataset(dataset, test_size, valid_size, random_state,
                                 train_transform, valid_transform):
    # train test split
    train_test_split = StratifiedShuffleSplit(n_splits=1, test_size=test_size,
                                              random_state=random_state)
    train_inds, test_inds = next(train_test_split.split(dataset, y=dataset.targets))

    # train validation split
    train_valid_split = StratifiedShuffleSplit(n_splits=1, test_size=valid_size,
                                               random_state=random_state)
    train_inds, valid_inds = next(train_valid_split.split(train_inds, y=[dataset.targets[i] for i in train_inds]))

    train_dataset = SubDataset(Subset(dataset, train_inds), transform=train_transform)
    valid_dataset = SubDataset(Subset(dataset, valid_inds), transform=valid_transform)
    test_dataset = SubDataset(Subset(dataset, test_inds), transform=valid_transform)
    return train_dataset, valid_dataset, test_dataset

