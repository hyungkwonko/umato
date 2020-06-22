"""Manifold datasets."""
import numpy as np
from sklearn.datasets import make_s_curve, make_swiss_roll
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from .topo_dataset.spheres import create_sphere_dataset
import os
import pandas as pd

def normalize_features(data_train, data_test):
    """Normalize features to zero mean and unit variance.

    Args:
        data:

    Returns:
        (transformed_data_train, transformed_data_test)

    """
    mean = np.mean(data_train, axis=0, keepdims=True)
    std = np.std(data_train, axis=0, keepdims=True)
    transformed_train = (data_train - mean) / std
    transformed_test = (data_test - mean) / std
    return transformed_train, transformed_test

def normalize_x(data_train):
    mean = np.mean(data_train, axis=0, keepdims=True)
    std = np.std(data_train, axis=0, keepdims=True)
    transformed_train = (data_train - mean) / std
    return transformed_train

def file_exist(dname):
    os.path.isfile(dname)

class ManifoldDataset(Dataset):
    def __init__(self, data, position, train, test_fraction, random_seed):
        if test_fraction > 0:
            train_data, test_data, train_pos, test_pos = train_test_split(
                data, position, test_size=test_fraction, random_state=random_seed)
            self.train_data, self.test_data = normalize_features(
                train_data, test_data)
            self.train_pos, self.test_pos = train_pos, test_pos
            self.data = self.train_data if train else self.test_data
            self.pos = self.train_pos if train else self.test_pos
        else: # get 100 % training
            self.data = normalize_x(data)
            self.pos = position

    def __getitem__(self, index):
        return self.data[index], self.pos[index]

    def __len__(self):
        return len(self.data)


class SwissRoll(ManifoldDataset):
    def __init__(self, train=True, n_samples=6000, noise=0.05,
                 test_fraction=0.1, seed=42):
        _rnd = np.random.RandomState(seed)
        data, pos = make_swiss_roll(n_samples, noise, seed)
        data = data.astype(np.float32)
        pos = pos.astype(np.float32)
        super().__init__(data, pos, train, test_fraction, _rnd)


class SCurve(ManifoldDataset):
    def __init__(self, train=True, n_samples=6000, noise=0.05,
                 test_fraction=0.1, seed=42):
        _rnd = np.random.RandomState(seed)
        data, pos = make_s_curve(n_samples, noise, _rnd)
        data = data.astype(np.float32)
        pos = pos.astype(np.float32)
        super().__init__(data, pos, train, test_fraction, _rnd)

class Spheres(ManifoldDataset):
    def __init__(self, train=True, n_samples=500, d=100, n_spheres=11, r=5,
                test_fraction=0.0, seed=42): # test fraction set to 0.0 to put all into training dataset
        #here pos are actually class label, just conforming with parent class!
        path = os.path.join('..', '..', '..', 'data', 'spheres')

        if not os.path.isdir(path):
            os.makedirs(path) # this makes directories including all paths

        if os.path.isfile(os.path.join(path, 'spheres.csv')):
            df = pd.read_csv(os.path.join(path, 'spheres.csv')) # load data
            data = df.drop(columns=['label']).to_numpy()
            label = df['label'].to_numpy()
        else:
            data, label = create_sphere_dataset(n_samples, d, n_spheres, r, seed=seed) # create data
            df = pd.DataFrame(data)
            df['label'] = label
            df.to_csv(os.path.join(path, 'spheres.csv'), index=False)

        pos = label
        data = data.astype(np.float32)
        pos = pos.astype(np.float32)
        _rnd = np.random.RandomState(seed)
        super().__init__(data, pos, train, test_fraction, _rnd)


