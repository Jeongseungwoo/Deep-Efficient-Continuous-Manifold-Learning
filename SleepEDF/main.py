import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import numpy as np
import os, argparse
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn import covariance
from scipy import signal


from models.ODERGRU import ODERGRU
from train import train_op, test_op
from utils import BaseDataset, save_csv, calculate_metrics

# channel
''' Sleep-EDF ch names
'EEG Fpz-Cz',
 'EEG Pz-Oz',
 'EOG horizontal',
 'Resp oro-nasal',
 'EMG submental',
 'Temp rectal',
 'Event marker'
'''

def main(args):

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = 'cuda' if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    #load dataset
    ## label => wake 0, N1 1, N2 2, N3+N4 3, Rem 5
    path = args.path + '/Sleep_EDF/Sleep_Raw/{}{:02d}_{}.npy' #0-19 (20 subject)
    X_train = []
    X_test = []
    X_valid = []
    Y_train = []
    Y_test  = []
    Y_valid = []

    idx = np.random.permutation(78)
    valid_idx = idx[:7]
    test_idx = idx[7*(args.fold+1):7*(args.fold+2)]

    for i in tqdm(range(78)):
        X1 = np.load(path.format("data", i, 1))
        Y1 = np.load(path.format('label', i, 1))
        try:
            X2 = np.load(path.format("data", i, 2))
            Y2 = np.load(path.format('label', i, 2))
            X1 = np.concatenate((X1, X2))
            Y1 = np.concatenate((Y1, Y2))
        except:
            pass

        Y1 = np.where(Y1 == 5, 4, Y1) # change 5 -> 4
        X1 = X1[:, :args.channel, :]

        _, _, X1 = signal.stft(X1, fs=100, window='hamming', nperseg=200, noverlap=100, nfft=256, boundary=None)
        logX1 = np.log(np.abs(X1))

        # if i == args.sub:
        #     X_test.append(logX1)
        #     Y_test.append(Y1)
        # else:

        if i in valid_idx:
            X_valid.append(logX1)
            Y_valid.append(Y1)
        elif i in test_idx:
            X_test.append(logX1)
            Y_test.append(Y1)
        else:
            X_train.append(logX1)
            Y_train.append(Y1)

    X_train = np.concatenate(X_train)
    X_valid = np.concatenate(X_valid)
    X_test = np.concatenate(X_test)

    Y_train = np.concatenate(Y_train)
    Y_valid = np.concatenate(Y_valid)
    Y_test = np.concatenate(Y_test)

    # Normalization
    X_train_mean = X_train.mean((0), keepdims=True)
    X_train_std = X_train.std((0), keepdims=True)

    X_train = (X_train - X_train_mean) / X_train_std
    X_test = (X_test - X_train_mean) / X_train_std
    X_valid = (X_valid - X_train_mean) / X_train_std

    # Dataset
    train_ds = BaseDataset(X_train, Y_train, args)
    valid_ds = BaseDataset(X_valid, Y_valid, args)
    test_ds = BaseDataset(X_test, Y_test, args)

    # DataLoader
    trDL = DataLoader(train_ds, batch_size=args.batch, shuffle=True, drop_last=True)
    valDL = DataLoader(valid_ds, batch_size=args.T, shuffle=False, drop_last=True)
    teDL = DataLoader(test_ds, batch_size=args.T, shuffle=False, drop_last=True)

    cache = {'Model': args.model, 'Fold': args.fold, "Acc":0, 'F1':0, 'Kappa':0, 'Sens':0, 'Spec':0, 'val_loss':10, 'Epoch':0}

    if args.model == "ODEManifold":
        model = ODEManifold(n_class=5, n_layers=1, n_units=100, latents=args.latents, units=args.units, channel=args.channel, bi=args.bi, device=device).to(device)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-3)

    for epoch in range(args.epochs):
        print("Epoch {} Start.".format(epoch))
        _ = train_op(model, device, trDL, optimizer, criterion)
        test_loss, ys, h_ys = test_op(model, device, teDL, criterion, type='Test')
        val_loss, _, _ = test_op(model, device, valDL, criterion, type='Valid')
        preds = []
        targets = []
        ys = np.concatenate(ys)
        h_ys = np.concatenate(h_ys)
        for i in range(len(ys)-args.T): #이부분 다시코딩함.
            y = 0
            for t in reversed(range(args.T)):
                y += np.log(h_ys[i+t][args.T - 1 - t])
            preds.append(np.argmax(y))
            targets.append(ys[i][-1])
        acc, sens, spec, mf1, kappa = calculate_metrics(np.array(targets), np.array(preds))
        print("Test||Acc:{:.4f}, F1:{:.4f}, Kappa:{:.4f}, Sens:{:.4f}, Spec:{:.4f}".format(acc, mf1, kappa, sens, spec))
        if val_loss <= cache['val_loss']:
            # acc, sens, spec, mf1, kappa = calculate_metrics(targets, preds)
            cache['val_loss'] = val_loss
            cache['Acc'] = round(acc, 4)
            cache['F1'] = round(mf1, 4)
            cache['Kappa'] = round(kappa, 4)
            cache["Sens"] = round(sens, 4)
            cache['Spec'] = round(spec, 4)
            cache['Epoch'] = epoch
            state = {
                'model': model.state_dict(),
                'pred': h_ys
            }
            if not os.path.isdir('checkpoint_sleep'):
                os.mkdir('checkpoint_sleep')
            torch.save(state, './checkpoint_sleep/{}_ckpt.t7'.format(args.fold))
        print("-"*5, "Fold:{}, Epoch:{} || Acc:{:.4f}, F1:{:.4f}, Kappa:{:.4f}, Sens:{:.4f}, Spec:{:.4f}".format(cache['Fold'],
                                                                                             cache['Epoch'],
                                                                                             cache['Acc'],
                                                                                             cache['F1'],
                                                                                             cache['Kappa'],
                                                                                             cache['Sens'],
                                                                                             cache['Spec']))
    save_csv('Sleep_EEG', cache)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='Sleep stage classification')
    parser.add_argument('-g', '--gpu', default='6', help='GPU number')
    parser.add_argument('--epochs', default=20, type=int, help='Number of epochs')
    parser.add_argument('--latents', default=16, type=int, help='Latents dimension')
    parser.add_argument('--units', default=50, type=int)
    parser.add_argument('-T', default=20, type=int)
    parser.add_argument('-channel', default=2, type=int)
    parser.add_argument('-bi', default=True)

    # parser.add_argument('--time', default=1500, type=int, help='time interval for spd matrix')
    parser.add_argument('--lr', default=0.0001, type=float, help='Learning rate')
    parser.add_argument('--seed', default=100, help='Random seed')
    parser.add_argument('--model', default='ODEManifold') #SPDSRU

    parser.add_argument('--ode', default = True, type=bool)

    parser.add_argument('--path', default='/DataRead/swjeong', help='Directory')
    # parser.add_argument('--path', default='../../../datasets', help='Directory')
    parser.add_argument('--batch', default=32, type=int, help='Batch size')
    # parser.add_argument('--sub', default=13, type=int, help='Subject number') #0-19
    parser.add_argument('--fold', default=0, type=int, help='Fold number') #0-9
    args = parser.parse_args()

    main(args)
