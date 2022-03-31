from torch.utils.data import Dataset
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score, f1_score, cohen_kappa_score
import numpy as np
import csv

# class BaseDataset(Dataset):
#     def __init__(self, raw, target):
#         self.raw = raw
#         self.target = target
#
#     def __getitem__(self, index):
#         return self.raw[index].astype('float32'), self.target[index]
#
#     def __len__(self):
#         return len(self.raw)

class BaseDataset(Dataset):
    def __init__(self, raw, target, args):
        self.raw = raw
        self.target = target
        self.args = args

    def __getitem__(self, index):
        x = []
        y = []
        for t in range(self.args.T):
            x.append(self.raw[index+t].astype('float32'))
            y.append(self.target[index+t])
        x = np.stack(x, axis=0)
        y = np.stack(y, axis=0)
        return x, y

    def __len__(self):
        return len(self.raw) - self.args.T


def save_csv(filename, cache):
    with open("./{}_results.csv".format(filename), 'a') as f:
        w = csv.writer(f)
        if f.tell() == 0: w.writerow(cache.keys())
        w.writerow(cache.values())

def calculate_metrics(y, y_pred):
    #acc, sens, spec, mf1, kappa
    def average_sen_spec(y, y_pred):
        tn = multilabel_confusion_matrix(y, y_pred)[:, 0, 0]
        fn = multilabel_confusion_matrix(y, y_pred)[:, 1, 0]
        tp = multilabel_confusion_matrix(y, y_pred)[:, 1, 1]
        fp = multilabel_confusion_matrix(y, y_pred)[:, 0, 1]
        sens = tp / (tp + fn)
        spec = tn / (tn + fp)
        return sens.mean(), spec.mean()
    acc = accuracy_score(y, y_pred)
    sens, spec = average_sen_spec(y, y_pred)
    mf1 = f1_score(y, y_pred, average='macro')
    kappa = cohen_kappa_score(y, y_pred)
    return acc, sens, spec, mf1, kappa