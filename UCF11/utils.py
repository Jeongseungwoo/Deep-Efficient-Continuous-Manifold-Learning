from torch.utils.data import Dataset
import csv

class BaseDataset(Dataset):
    def __init__(self, raw, target):
        self.raw = raw
        self.target = target

    def __getitem__(self, index):
        return self.raw[index].astype('float32'), self.target[index]

    def __len__(self):
        return len(self.raw)

def save_csv(filename, cache):
    with open("./{}_results.csv".format(filename), 'a') as f:
        w = csv.writer(f)
        if f.tell() == 0: w.writerow(cache.keys())
        w.writerow(cache.values())

