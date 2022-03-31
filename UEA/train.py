import torch
import numpy as np

from tqdm import tqdm

def train_op(model, ds, optimizer, criterion, device):
    model.train()
    bar = tqdm(ds)
    train_loss, correct, total = 0, 0.0, 0.0
    for i, data in enumerate(bar):
        data, target = data[0].to(device), data[1].type(torch.LongTensor).to(device)
        output = model(data)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.data.item()
        _, pred = torch.max(output.data, 1)
        total += target.size(0)
        correct += pred.eq(target.data).cpu().sum().data.item()

        bar.set_description('Train| Loss: {:.4f}, Acc: {:.4f} ({}/{})'.format((train_loss/(i+1)), 100.*correct/total, correct, total))
    acc = 100. * correct / total
    return acc, train_loss / (i+1)

def test_op(model, ds, criterion, device, type="Test"):
    model.eval()
    bar = tqdm(ds)
    test_loss, correct, total = 0, 0.0, 0.0
    for i, data in enumerate(bar):
        data, target = data[0].to(device), data[1].type(torch.LongTensor).to(device)
        output = model(data)
        loss = criterion(output, target)

        test_loss += loss.data.item()
        _, pred = torch.max(output.data, 1)
        total += target.size(0)
        correct += pred.eq(target.data).cpu().sum().data.item()

        bar.set_description(
            '{}| Loss: {:.4f}, Acc: {:.4f} ({}/{})'.format(type, (test_loss / (i+1)), 100. * correct / total, correct, total))
    acc = 100. * correct / total
    return acc, test_loss / (i+1)
