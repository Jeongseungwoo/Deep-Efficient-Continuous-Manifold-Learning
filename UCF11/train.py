import torch
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

def train_op(model, device, ds, optimizer, criterion):
    model.train()
    train_loss, acc, correct, total = 0, 0, 0, 0
    bar = tqdm(ds)
    # bar = ds
    for data, target in bar:
        data, target = data.to(device), target.to(device)
        output = model(data)
        l2_penalty = 1e-4 * sum([(p ** 2).sum() for p in model.parameters()])
        loss = criterion(output, target) # + l2_penalty

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.data.item()
        _, pred = torch.max(output.data, 1)
        total += target.size(0)
        correct += pred.eq(target.data).cpu().sum().data.item()
        acc = 100.*correct/total
        bar.set_description('Train|Loss:{:.4f}, Acc:{:.4f} ({}/{})'.format(train_loss/total, acc, correct, total))
    return train_loss/total, acc

def test_op(model, device, ds, criterion, type='Test'):
    with torch.no_grad():
        model.eval()
        test_loss, acc, correct, total= 0, 0, 0, 0
        bar = tqdm(ds)
        # bar = ds
        for data, target in bar:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            test_loss += loss.data.item()
            _, pred = torch.max(output.data, 1)
            total += target.size(0)
            correct += pred.eq(target.data).cpu().sum().data.item()
            acc = 100.*correct/total
            bar.set_description('{}|Loss:{:.4f}, Acc:{:.4f}({}/{})'.format(type, test_loss/total, acc, correct, total))
    return test_loss/total, acc
