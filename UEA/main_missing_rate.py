import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import collections as co
import numpy as np
import os, argparse, datetime
from sklearn.model_selection import KFold

from model_mr import ODERGRU
from train import train_op, test_op
from data_loader import load_UEA, BaseDataset

import sklearn.model_selection

def split_data(tensor, stratify):
    # 0.7/0.15/0.15 train/val/test split
    (train_tensor, testval_tensor,
     train_stratify, testval_stratify) = sklearn.model_selection.train_test_split(tensor, stratify,
                                                                                  train_size=0.7,
                                                                                  random_state=0,
                                                                                  shuffle=True,
                                                                                  stratify=stratify)

    val_tensor, test_tensor = sklearn.model_selection.train_test_split(testval_tensor,
                                                                       train_size=0.5,
                                                                       random_state=1,
                                                                       shuffle=True,
                                                                       stratify=testval_stratify)
    return train_tensor, val_tensor, test_tensor

def main(args):
    # GPU setting
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(1111)

    # Log
    directory = "./{}/{}/{}/".format(args.task, args.dataset, str(datetime.datetime.now().strftime('%Y%m%d%H%M%S')))
    if not os.path.exists(directory):
        os.makedirs(directory)

    def writelog(file, line):
        file.write(line + "\n")
        print(line)

    f = open(directory + "log.txt", 'a')
    writelog(f, "-" * 20)
    writelog(f, 'TRAINING PARAMETER')
    writelog(f, 'Dataset: ' + str(args.dataset))
    writelog(f, 'Learning Rate : ' + str(args.lr))
    writelog(f, 'Weight Decay : ' + str(args.weight_decay))
    writelog(f, 'Batch Size : ' + str(args.batch_size))
    writelog(f, "Time : " + str(args.time))
    writelog(f, "Dilation : " + str(args.dilation))
    writelog(f, "Kernels : " + str(args.kernels))
    writelog(f, "Filters : " + str(args.filters))
    writelog(f, "Matrix_dim : " + str(args.matrix_dim))
    writelog(f, "Rnn_dim : " + str(args.rnn_dim))
    writelog(f, "-" * 20)
    writelog(f, 'TRAINING LOG')

    # Load dataset
    print("Loading {} dataset: {}".format(args.task, args.dataset))

    path = os.path.join(args.path, "Multivariate_arff")
    ds = load_UEA(path, args.dataset)
    train, train_label, test, test_label = ds
    X = np.concatenate((train, test), axis=0)
    X = torch.Tensor(X)
    X = X.transpose(-1, -2) # batch, length, channel
    y = np.concatenate((train_label, test_label), axis=0)
    times = torch.linspace(0, X.size(1)-1, X.size(1))
    lengths = torch.tensor([len(Xi[0]) for Xi in X])
    final_index = lengths - 1

    generator = torch.Generator().manual_seed(56789)
    for Xi in X:
        removed_point = torch.randperm(X.size(1), generator=generator)[:int(X.size(1)*args.missing_rate)].sort().values
        Xi[removed_point] = 0.0#float("nan")
    targets = co.OrderedDict()
    counter = 0
    for yi in y:
        if yi not in targets:
            targets[yi] = counter
            counter += 1
    y = torch.tensor([targets[yi] for yi in y])
    # X.shape = [2858, 182, 3]
    augmented_X = []
    # if True: # append_time
    #     augmented_X.append(times.unsqueeze(0).repeat(X.size(0), 1).unsqueeze(-1))
    # if False: # Intensity
    #     intensity = ~torch.isnan(X)  # of size (batch, stream, channels)
    #     intensity = intensity.to(X.dtype).cumsum(dim=1)
    #     augmented_X.append(intensity)
    augmented_X.append(X)
    if len(augmented_X) == 1:
        X = augmented_X[0]
    else:
        X = torch.cat(augmented_X, dim=2)
    train_X, val_X, test_X = split_data(X.transpose(-1, -2), y)
    train_y, val_y, test_y = split_data(y, y)
    train_final_index, val_final_index, test_final_index = split_data(final_index, y)

    train_ds = BaseDataset(train_X, train_y)
    test_ds = BaseDataset(test_X, test_y)
    val_ds = BaseDataset(val_X, val_y)

    trDL = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    teDL = DataLoader(test_ds, batch_size=args.batch_size)
    valDL = DataLoader(val_ds, batch_size=args.batch_size)
    # Define model
    model = ODERGRU(missing_rate=args.missing_rate,
                    n_class=np.unique(train_label).shape[0],
                    n_layers=3,
                    n_units=32,
                    units=args.rnn_dim,
                    channel=train.shape[1],
                    latents=args.matrix_dim,
                    kernels=args.kernels,
                    filters=args.filters,
                    dilation=args.dilation,
                    bi=False,
                    device=device).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    cache = {
        "Task": args.task, "Dataset": args.dataset, 'Epoch': 0, "Acc": 0
    }

    best_loss = 10
    for epoch in range(args.epochs):
        _, _ = train_op(model, trDL, optimizer, criterion, device)
        test_acc, test_loss = test_op(model, teDL, criterion, device, type='Test')
        val_acc, val_loss = test_op(model, valDL, criterion, device, type='Valid')
        if val_loss < best_loss:
            best_loss = val_loss
            cache['Acc'] = test_acc
            cache['Epoch'] = epoch
            state = {"model": model.state_dict(), 'acc': test_acc}
            torch.save(state, directory + "{}_ckpt.t7".format(args.dataset))
        writelog(f, "------Epoch: {}({}), Acc: {:.4f}------".format(epoch, cache["Epoch"], cache['Acc']))
    writelog(f, "-" * 20)
    writelog(f, 'Summary of Dataset :{}'.format(args.dataset))
    writelog(f, 'Best Epoch : {}'.format(cache['Epoch']))
    writelog(f, 'ACC : {:.4f}'.format(cache['Acc']))
    f.close()


if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser(prog="Time series classification")
    parser.add_argument("-g", '--gpu', type=str, default="0")
    parser.add_argument("--missing_rate", type=float, default=0.3)
    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--task', type=str, default="UEA")  # UCR, UEA
    parser.add_argument('--dataset', type=str, default='CharacterTrajectories')
    parser.add_argument('--path', type=str, default='./')
    # Model parameters
    parser.add_argument('--time', type=int, default=5)  # time percent.
    parser.add_argument('--dilation', type=int, default=1)
    parser.add_argument('--kernels', type=str, default="1,1")
    parser.add_argument('--filters', type=str, default="32,32")
    parser.add_argument('--matrix_dim', type=int, default=32)
    parser.add_argument('--rnn_dim', type=int, default=32)
    args = parser.parse_args()

    main(args)
