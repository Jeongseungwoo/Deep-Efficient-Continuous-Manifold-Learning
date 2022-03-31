from torch.utils.data import Dataset

import os
import numpy as np
import pandas as pd

from scipy.io import arff
from scipy.signal import resample

def load_UEA(path, dataset):

    train_file = os.path.join(path, dataset, dataset + "_TRAIN.arff")
    test_file = os.path.join(path, dataset, dataset + "_TEST.arff")
    train_df = pd.DataFrame(arff.loadarff(train_file)[0])
    test_df = pd.DataFrame(arff.loadarff(test_file)[0])

    label_dic = {j: i for i, j in enumerate(pd.unique(train_df[train_df.columns[1]]))}
    train_att = train_df[train_df.columns[0]]
    test_att = test_df[test_df.columns[0]]
    train_size = len(train_att)
    test_size = len(test_att)
    nb_dims = len(train_att[0])
    length = len(train_att[0][0])

    train_label = train_df[train_df.columns[1]].replace(label_dic).to_numpy()
    test_label = test_df[test_df.columns[1]].replace(label_dic).to_numpy()

    train = np.zeros(shape=(train_size, nb_dims, length))
    test = np.zeros(shape=(test_size, nb_dims, length))

    for i in range(len(train_att)):
        train[i, :, :] = np.array(train_att[i].tolist())

    for i in range(len(test_att)):
        test[i, :, :] = np.array(test_att[i].tolist())
    np.nan_to_num(train, 0)
    np.nan_to_num(test, 0)
    # Normalizing dimensions independently
    for j in range(nb_dims):
        mean = np.mean(np.concatenate([train[:, j], test[:, j]]))
        var = np.var(np.concatenate([train[:, j], test[:, j]]))
        train[:, j] = (train[:, j] - mean) / np.sqrt(var)
        test[:, j] = (test[:, j] - mean) / np.sqrt(var)

    return train, train_label, test, test_label

class BaseDataset(Dataset):
    def __init__(self, raw, target):
        self.raw = raw
        self.target = target

    def __getitem__(self, index):
        # return self.raw[index].astype('float32'), self.target[index]
        return self.raw[index], self.target[index]

    def __len__(self):
        return len(self.raw)
