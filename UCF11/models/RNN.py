import torch
import torch.nn as nn

class GRU(nn.Module):
    def __init__(self, device):
        super(GRU, self).__init__()
        self.cnn = CNN()
        self.rnncell = nn.GRUCell(7500, 750)
        self.cls = nn.Linear(750, 11)
        self.device = device

    def forward(self, x):
        x = x.permute(0, 1, 4, 2, 3)
        b, s, d, w, h = x.shape
        x = x.reshape(-1, d, w, h)
        x = self.cnn(x)
        x = x.reshape(b, s, -1)
        h = torch.zeros(x.shape[0], 750, device=self.device)
        for i in range(x.shape[1]):
            h = self.rnncell(x[:, i, :], h)
        return self.cls(h)

class LSTM(nn.Module):
    def __init__(self, device):
        super(LSTM, self).__init__()
        self.cnn = CNN()
        self.rnncell = nn.LSTMCell(7500, 750)
        self.cls = nn.Linear(750, 11)
        self.device = device

    def forward(self, x):
        x = x.permute(0, 1, 4, 2, 3)
        b, s, d, w, h = x.shape
        x = x.reshape(-1, d, w, h)
        x = self.cnn(x)
        x = x.reshape(b, s, -1)
        h = torch.zeros(x.shape[0], 750).to(self.device)
        c = torch.zeros(x.shape[0], 750).to(self.device)
        for i in range(x.shape[1]):
            h, c = self.rnncell(x[:, i, :], (h, c))
        return self.cls(h)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layers = nn.ModuleList()
        filters = [3, 10, 15, 25]
        for i in range(1, len(filters)):
            self.layers.append(
                nn.Sequential(
                    nn.Conv2d(in_channels=filters[i-1], out_channels=filters[i], kernel_size=7, padding=3),
                    nn.BatchNorm2d(filters[i]),
                    nn.ReLU(),
                    nn.MaxPool2d(2, 2)
                )
            )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x