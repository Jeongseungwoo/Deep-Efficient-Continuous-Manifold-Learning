import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import numpy as np
import os, argparse, datetime
from sklearn.model_selection import KFold

from model import ODERGRU
from train import train_op, test_op
from data_loader import load_UEA, BaseDataset

def main(args):
    # GPU setting
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(1111)

    # Log
    directory= "./{}/{}/{}/".format(args.task, args.dataset,str(datetime.datetime.now().strftime('%Y%m%d%H%M%S')))
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
    if args.task == "UCR":
        path = os.path.join(args.path, "Univariate_arff")
        ds = load_UCR(path, args.dataset)
    else:
        if args.dataset == "DuckDuckGeese":
            path = os.path.join(args.path, "Multivariate_arff", "DuckDuckGeese")
            train = np.load(path + "/X_train.npy")
            train_label = np.load(path + "/y_train.npy")
            test = np.load(path + "/X_test.npy")
            test_label = np.load(path + "/y_test.npy")

        else:
            path = os.path.join(args.path, "Multivariate_arff")
            ds = load_UEA(path, args.dataset)
            train, train_label, test, test_label = ds


    # split time-window
    train_re = np.zeros((train.shape[0], args.time, train.shape[1], train.shape[2]//args.time))
    test_re = np.zeros((test.shape[0], args.time, test.shape[1], test.shape[2]//args.time))
    for i in range(args.time):
        train_re[:, i, :] = train[:, :, train.shape[2]//args.time * i : train.shape[2]//args.time * (i+1)]
        test_re[:, i, :] = test[:, :, test.shape[2]//args.time * i : test.shape[2]//args.time * (i+1)]
    # shape: [batch, seq, channel, time-feature)
    train = train_re
    test = test_re

    train_ds = BaseDataset(train, train_label)
    test_ds = BaseDataset(test, test_label)
        # valid_ds = BaseDataset(valX, valY)
    trDL = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    teDL = DataLoader(test_ds, batch_size=args.batch_size)
        # valDL = DataLoader(valid_ds, batch_size=len(valX))
        # Define model
    model = ODERGRU(n_class= np.unique(train_label).shape[0],
                    n_layers= 1,
                    n_units= 32,
                    units=args.rnn_dim,
                    channel=train.shape[2],
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
        train_acc, train_loss = train_op(model, trDL, optimizer, criterion, device)
        test_acc, test_loss = test_op(model, teDL, criterion, device, type='Test')
        if train_loss < best_loss:
            best_loss = train_loss
            cache['Acc'] = test_acc
            cache['Epoch'] = epoch
            state = {"model": model.state_dict(), 'acc': test_acc}
            torch.save(state, directory + "{}_ckpt.t7".format(args.dataset))
        writelog(f, "------Epoch: {}({}), Acc: {:.4f}------".format(epoch, cache["Epoch"], cache['Acc']))
    writelog(f, "-"*20)
    writelog(f, 'Summary of Dataset :{}'.format(args.dataset))
    writelog(f, 'Best Epoch : {}'.format(cache['Epoch']))
    writelog(f, 'ACC : {:.4f}'.format(cache['Acc']))
    f.close()

if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser(prog="Time series classification")
    parser.add_argument("-g", '--gpu', type=str, default="7")
    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--task', type=str, default="UEA") # UCR, UEA
    parser.add_argument('--dataset', type=str, default='ArticularyWordRecognition')
    parser.add_argument('--path', type=str, default='/DataRead/swjeong/')
    # Model parameters
    parser.add_argument('--time', type=int, default=20) # time percent.
    parser.add_argument('--dilation', type=int, default=1)
    parser.add_argument('--kernels', type=str, default= "2, 2, 2")
    parser.add_argument('--filters', type=str, default="256, 256, 256")
    parser.add_argument('--matrix_dim', type=int, default=32)
    parser.add_argument('--rnn_dim', type=int, default=32)
    args = parser.parse_args()

    main(args)
