import torch
import numpy as np
from tqdm import tqdm

def train_op(model, device, ds, optimizer, criterion):
    model.train()
    train_loss, correct, total = 0, 0, 0
    bar = tqdm(ds)
    ys = []
    h_ys = []
    for data, target in bar:
        data, target = data.to(device), target.to(device)
        output = model(data.float())
        loss = criterion(output[:, 0, :], target[:, 0].long())
        for t in range(1, output.shape[1]):
            loss += criterion(output[:, t, :], target[:, t].long())
        loss /= output.shape[1]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.data.item()
        # _, pred = torch.max(output.data, 2)
        # ys.append(target.cpu().numpy())
        # h_ys.append(pred.cpu().numpy())
        total += target.size(0)
        # correct += pred.eq(target.data).cpu().sum().data.item()
        # acc = 100. * correct / total
        # bar.set_description('Train|Loss:{:.4f}, Acc:{:.4f} ({}/{})'.format(train_loss / total, acc, correct, total))
        bar.set_description('Train|Loss:{:.4f} ({})'.format(train_loss / total, total))
    return train_loss / total

def test_op(model, device, ds, criterion, type='Test'):
    with torch.no_grad():
        model.eval()
        test_loss, correct, total = 0, 0, 0
        bar = tqdm(ds)
        ys = []
        h_ys = []
        for data, target in bar:
            data, target = data.to(device), target.to(device)
            output = model(data.float())
            loss = criterion(output[:, 0, :], target[:, 0].long())
            for t in range(1, output.shape[1]):
                loss += criterion(output[:, t, :], target[:, t].long())
            loss /= output.shape[1]

            test_loss += loss.data.item()
            # _, pred = torch.max(output.data, 1)
            ys.append(target.cpu().numpy())
            h_ys.append(output.softmax(-1).cpu().numpy())
            total += target.size(0)
            # correct += pred.eq(target.data).cpu().sum().data.item()
            # acc = 100.*correct/total
            bar.set_description('{}|Loss:{:.4f} ({})'.format(type, test_loss/total, total))
    return test_loss/total, ys, h_ys