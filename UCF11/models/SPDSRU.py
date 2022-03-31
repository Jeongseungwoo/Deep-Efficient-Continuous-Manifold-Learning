import torch
import torch.nn as nn


class SPDSRU(nn.Module):
    def __init__(self, device):
        super(SPDSRU, self).__init__()
        self.CNN = CNN(device=device)
        self.Cells = nn.ModuleList()
        self.n = 8
        self.alpha = [0.01, 0.25, 0.5, 0.9, 0.99]
        self.depth = 5
        for _ in range(self.depth):
            self.Cells.append(SPDSRUcell(device=device))
        self.cls = nn.Sequential(
            nn.BatchNorm1d(self.n * (self.n + 1) // 2),
            nn.Dropout(),
            nn.Linear(self.n * (self.n + 1) // 2, 11)
        )
        self.device = device

    def forward(self, x):
        x = x.permute(0, 1, 4, 2, 3)
        b, s, d, w, h = x.shape
        x = x.reshape(-1, d, w, h)
        x = self.CNN(x)
        x = x.reshape(b, s, x.shape[1], x.shape[2])  # 8 x 8 spd matrix

        states = [torch.tile(torch.eye(self.n, device=self.device) * 1e-5, [b, len(self.alpha), 1, 1]).reshape(b, -1)
                  for _ in range(self.depth)]

        for i in range(x.shape[1]):
            states_ = []
            h, state = self.Cells[0](x[:, i, :, :], states[0])
            states_.append(state)
            for j in range(1, len(self.Cells)):
                h, state = self.Cells[j](h, states[j])
                states_.append(state)
            states = states
        outputs = h
        outputs = outputs.reshape((b, self.n, self.n))
        output_s = self.Chol_de(outputs, self.n, b)
        return self.cls(output_s)

    def Chol_de(self, A, n, batch_size):
        L = A
        result = L[:, 0:1, 0:1]
        for i in range(1, n):
            j = i
            result = torch.cat([result, L[:, i:i + 1, :j + 1]], dim=2)
        result = result.reshape((-1, n * (n + 1) // 2))
        return result


class SPDSRUcell(nn.Module):
    def __init__(self, alpha=[0.01, 0.25, 0.5, 0.9, 0.99], matrix_size=8, n_labels=11, depth=5, eps=1e-10,
                 device='cpu'):
        super(SPDSRUcell, self).__init__()
        self.alpha = alpha
        self.a_num = len(alpha)
        self.n = matrix_size
        self.state_size = int(self.a_num * self.n * self.n)
        self.eps = eps
        self.depth = depth

        self.n = matrix_size
        self.n_labels = n_labels

        # SPDSRU build
        self.kernel_r = nn.Parameter(torch.rand(self.a_num, 1, device=device))
        self.kernel_t = nn.Parameter(torch.rand(1, 1, device=device))
        self.kernel_phi = nn.Parameter(torch.rand(1, 1, device=device))
        self.kernel_s = nn.Parameter(torch.rand(self.a_num, 1, device=device))

        self.bias_r = nn.Parameter(torch.rand(self.n * (self.n - 1) // 2, 1, device=device))
        self.bias_t = nn.Parameter(torch.rand(self.n * (self.n - 1) // 2, 1, device=device))
        self.bias_y = nn.Parameter(torch.rand(self.n * (self.n - 1) // 2, 1, device=device))

        self.device = device

    def forward(self, x, states):
        output, states = self.spdsrucell(x, states)
        return output, states

    def spdsrucell(self, inputs, states):
        Xt = inputs.reshape((-1, self.n, self.n))
        Mt_1 = states.reshape((-1, self.a_num, self.n, self.n))
        Yt = self.NUS(Mt_1, self.kernel_r, self.state_size, (self.kernel_r ** 2).sum() + self.eps)
        Rt = self.Translation(Yt, self.bias_r, self.n)

        Tt = self.FM(Xt, Rt, (self.kernel_t ** 2) / (self.kernel_t ** 2 + self.kernel_phi ** 2 + self.eps), self.n)
        Phit = self.Translation(Tt, self.bias_t, self.n)

        next_state = []
        for j in range(self.a_num):
            next_state.append(self.FM(Mt_1[:, j, :, :], Phit, self.alpha[j], self.n).unsqueeze(1))
        Mt = torch.cat(next_state, 1)
        St = self.NUS(Mt, self.kernel_s, self.state_size, (self.kernel_s ** 2).sum() + self.eps)
        Ot = self.Translation(St, self.bias_y, self.n)
        out_state = Mt.reshape((-1, int(self.a_num * self.n * self.n)))
        output = Ot.reshape((-1, int(self.n * self.n)))

        return output, out_state

    def FM(self, A, B, a, n):
        return (1 - a) * A + a * B

    def NUS(self, A, W, a_num, tot):
        W = W ** 2
        if a_num == 1:
            return (W[0] / tot) * A
        else:
            result = A[:, 0, :, :] * (W[0] / tot)
            for i in range(1, A.shape[1]):
                result = result + (A[:, i, :, :] * (W[i] / tot))
            return result

    def Translation(self, A, B, n):
        power_matrix = 5
        B = B.reshape((1, -1))
        line_B = [torch.zeros(1, n, device=self.device)]
        for i in range(n - 1):
            temp_line = torch.cat([B[0:1, i:i + i + 1], torch.zeros(1, n - i - 1, device=self.device)], axis=1)
            line_B.append(temp_line)

        lower_triangel = torch.cat(line_B, axis=0)

        B_matrix = lower_triangel - lower_triangel.transpose(-1, -2)

        B_matrix = self.MatrixExp(B_matrix, power_matrix, n)

        B_matrix = torch.tile(B_matrix.unsqueeze(0), [A.shape[0], 1, 1])
        Tresult = torch.matmul(B_matrix, A)  # B * A
        Tresult = torch.matmul(Tresult, B_matrix.transpose(-1, -2))  # B * A * B.T
        return Tresult

    def MatrixExp(self, B, l, n):
        Result = torch.eye(n, device=self.device)
        return torch.matmul(torch.inverse(Result - B), Result + B)


class CNN(nn.Module):
    def __init__(self, device):
        super(CNN, self).__init__()
        self.layers = nn.ModuleList()
        filters = [3, 5, 7]
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