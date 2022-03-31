import torch
import torch.nn as nn
import torch.nn.functional as F


class manifoldDCNN(nn.Module):
    def __init__(self, device):
        super(manifoldDCNN, self).__init__()
        filters1 = [1, 3, 4]
        filters2 = [3, 3, 4]
        filters3 = [3, 4, 4]
        k = 5
        self.depth = len(filters1)
        self.cnn = CNN(device=device)
        self.dcnn_stack = nn.ModuleList()
        for i in range(self.depth):
            self.dcnn_stack.append(
                DCNN(k, 2 ** i, filters1[i], filters2[i], filters3[i])
            )
        self.cls = nn.Sequential(
            nn.BatchNorm1d(112),
            nn.Dropout(),
            nn.Linear(112, 11)
        )

    def forward(self, x):
        x = x.permute(0, 1, 4, 2, 3)
        b, s, d, w, h = x.shape
        x = x.reshape(-1, d, w, h)
        x = self.cnn(x)
        _, n, n = x.shape
        x = x.reshape(b, s, x.shape[1] * x.shape[2], 1)
        for layer in self.dcnn_stack:
            x = layer(x)
        x = x[:, -1, :, :]
        x = x.transpose(-1, -2)
        x = x.reshape(-1, n, n)
        x = self.Chol_de(x, n, b * 4)
        x = x.reshape(b, 4 * n * (n + 1) // 2)
        return self.cls(x)

    def Chol_de(self, A, n, batch_size):
        L = A
        result = L[:, 0:1, 0:1]
        for i in range(1, n):
            j = i
            result = torch.cat([result, L[:, i:i + 1, :j + 1]], dim=2)
        result = result.reshape((-1, n * (n + 1) // 2))
        return result


class DCNN(nn.Module):
    def __init__(self, k, skip, in_ch, mid_ch, out_ch):
        super(DCNN, self).__init__()
        self.w_1 = nn.Parameter(torch.rand(k, in_ch, mid_ch))
        self.w_2 = nn.Parameter(torch.rand(k, mid_ch, out_ch))
        self.wFM = nn.Parameter(torch.rand(in_ch + out_ch, out_ch))

        self.k = k
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.skip = skip

    def forward(self, x):
        # x = [b, s, n*n, 1]
        y1 = self.dcnn(x, self.skip, self.w_1)
        y2 = self.dcnn(y1, self.skip, self.w_2)
        output = self.res_wFM(y2, x, self.wFM)
        return output

    def res_wFM(self, dx, x, W):
        W = W ** 2
        W_sum = W.sum(0)
        W = W / W_sum

        x_dx = torch.cat([x, dx], dim=3)
        x_dx = x_dx.reshape(-1, x.shape[3] + dx.shape[3])
        x_dx = x_dx.matmul(W)
        x_dx = x_dx.reshape(dx.shape)
        return x_dx

    def dcnn(self, x, d, W):
        b, s, n, _ = x.shape
        W = W ** 2
        padding = (self.k - 1) * d
        x_pad = F.pad(x, (0, 0, 0, 0, padding, 0, 0, 0))

        in_ch = W.shape[1]
        out_ch = W.shape[2]

        W = W.reshape(-1, out_ch)
        W_sum = W.sum(0)
        W = W / W_sum
        W = W.reshape(1, self.k, in_ch, out_ch)
        W = W.permute(3, 2, 0, 1)
        # [1, k, in, out] #height, width, in, out
        x = x_pad.permute(0, 2, 1, 3)
        x = x.permute(0, 3, 1, 2)  # b, ch, s, n
        x = F.conv2d(x, W, dilation=d)
        x = x.permute(0, 2, 3, 1)  # b, s, n, ch
        x = x.reshape(b, s, n, -1)
        x = x.permute(0, 2, 1, 3)
        return x

class CNN(nn.Module):
    def __init__(self, device):
        super(CNN, self).__init__()
        self.layers = nn.ModuleList()
        filters = [3, 4, 6]
        for i in range(1, len(filters)):
            self.layers.append(
                nn.Sequential(
                    nn.Conv2d(in_channels=filters[i - 1], out_channels=filters[i], kernel_size=7, padding=3),
                    nn.BatchNorm2d(filters[i]),
                    nn.ReLU(),
                    nn.MaxPool2d(2, 2)
                )
            )

        self.output_dim = filters[-1]
        self.reduced_spatial_dim = 120 * 160 // (4 ** 2)
        self.beta = .3
        self.device = device

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        P2 = x.permute(0, 3, 2, 1)
        F1 = P2.reshape((x.shape[0], self.output_dim, self.reduced_spatial_dim))
        mean_b = torch.mean(F1, dim=2)
        mean_t = torch.tile(mean_b.unsqueeze(2), [1, 1, self.reduced_spatial_dim])
        F1_m = F1 - mean_t

        mean_b = mean_b.unsqueeze(2)
        mean_cov = mean_b.matmul(mean_b.transpose(-1, -2))

        cov_feat = F1_m.matmul(F1_m.transpose(-1, -2)) + self.beta * self.beta * mean_cov
        cov_feat = torch.cat([cov_feat, self.beta * mean_b], dim=2)

        mean_b_t = torch.cat([self.beta * mean_b, torch.ones(x.shape[0], 1, 1, device=self.device)], dim=1)
        mean_b_t = mean_b_t.transpose(-1, -2)

        cov_feat = torch.cat([cov_feat, mean_b_t], dim=1)

        return cov_feat