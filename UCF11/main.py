import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import numpy as np
import os, argparse
from models.ODERGRU import ODERGRU

from models.RNN import GRU, LSTM
from models.TT_RNN import TT_GRU, TT_LSTM
from models.SPDSRU import SPDSRU
from models.DCNN import manifoldDCNN

from train import train_op, test_op
from utils import BaseDataset, save_csv

def main():
    #setting
    parser = argparse.ArgumentParser(prog='ODE-SPD')
    parser.add_argument('-g', '--gpu', default='1', help='GPU number')
    parser.add_argument('--n-epochs', default=1000, type=int, help='Number of epochs')
    parser.add_argument('--latents', default=32, type=int, help='Latents dimension')
    parser.add_argument('--units', default=100, type=int)
    parser.add_argument('--lr', default=0.0001, type=float, help='Learning rate')
    parser.add_argument('--seed', default=100, help='Random seed')
    parser.add_argument('--model', default='ODERGRU') #SPDSRU

    parser.add_argument('--ode', default = True, type=bool)

    parser.add_argument('--path', default='/DataRead/swjeong', help='Directory')
    # parser.add_argument('--path', default='../../../datasets', help='Directory')
    parser.add_argument('--batch', default=8, type=int, help='Batch size')
    parser.add_argument('-f', '--fold', default=1, type=int, help='Index of fold') #5
    parser.add_argument('--dataset', default='UCF11')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = 'cuda' if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)

    path = args.path + '/UCF11/UCF11_CV/{}/'.format(args.fold)
    X_train = np.load(path + 'X_train.npy', allow_pickle=True)
    Y_train = np.argmax(np.load(path + 'Y_train.npy', allow_pickle=True), 1)
    X_test = np.load(path + 'X_test.npy', allow_pickle=True)
    Y_test = np.argmax(np.load(path + 'Y_test.npy', allow_pickle=True), 1)

    #normalization
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    for i in range(3):
        X_train[..., i] = (X_train[..., i] - mean[i])/std[i]
        X_test[..., i] = (X_test[..., i] - mean[i])/std[i]

    train_ds = BaseDataset(X_train, Y_train)
    test_ds = BaseDataset(X_test, Y_test)
    trDL = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
    teDL = DataLoader(test_ds, batch_size=args.batch, shuffle=False)

    cache = {'Model': args.model, 'Fold': 0, 'Acc': 0, 'train_loss': 10, 'Epoch': 0}

    print("----------{} Fold start----------".format(args.fold))
    cache['Fold'] = args.fold

    #model setting
    decay_steps = 1000
    decay_rate = 1

    if args.model == 'ODERGRU':
        model = ODERGRU(n_class=11, n_layers=1, n_units=100, latents=args.latents, units=args.units, ode=args.ode, device=device).to(device)

    elif args.model == 'LSTM':
        model = LSTM(device=device).to(device)

    elif args.model == 'GRU':
        model = GRU(device=device).to(device)

    elif args.model == 'TT_GRU':
        tt_input_shape = [8, 20, 20, 18]
        tt_output_shape = [4, 4, 4, 4]
        tt_ranks = [1, 4, 4, 4, 1]
        model = TT_GRU(tt_input_shape, tt_output_shape, tt_ranks, device=device).to(device)
    elif args.model == 'TT_LSTM':
        tt_input_shape = [8, 20, 20, 18]
        tt_output_shape = [4, 4, 4, 4]
        tt_ranks = [1, 4, 4, 4, 1]
        model = TT_LSTM(tt_input_shape, tt_output_shape, tt_ranks, device=device).to(device)

    elif args.model == 'SPDSRU':
        model = SPDSRU(device = device).to(device)
        # args.lr = 1e-3

    elif args.model == 'DCNN':
        model = manifoldDCNN(device=device).to(device)
        # decay_steps = 40
        # decay_rate = 0.99

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=decay_rate)

    for epoch in range(args.n_epochs):
        print("Epoch:{} Start.".format(epoch))
        train_loss, train_acc= train_op(model, device, trDL, optimizer, criterion)
        test_loss, test_acc = test_op(model, device, teDL, criterion, type='Test')
        if epoch % decay_steps == 0:
            scheduler.step()
        if train_loss <= cache['train_loss']:
            cache['train_loss'] = train_loss
            cache['Acc'] = test_acc
            cache['Epoch'] = epoch
            state = {
                'model' : model.state_dict(),
                'acc' : cache['Acc']
            }
            if not os.path.isdir('checkpoint_UCF'):
                os.mkdir('checkpoint_UCF')
            torch.save(state, './checkpoint_UCF/{}{}_2_ckpt.t7'.format(args.model, args.fold))
        print("-"*10, "Fold:{}, Epoch:{}|Acc:{}".format(cache['Fold'], cache['Epoch'], cache['Acc']))
    save_csv(args.dataset, cache)

if __name__ == '__main__':
    main()
