import torch
import torch.nn as nn
import torchvision.models as models

import numpy as np
import math

from torchdiffeq import odeint as odeint


class ODERGRU(nn.Module):
    def __init__(self, n_class, n_layers, n_units, latents, units, ode=True, device='cpu'):
        super(ODERGRU, self).__init__()

        self.latents = latents
        self.units = units
        self.drop = drop # rate

        self.cnn = CNN(latents=latents)
        self.odefunc = ODEFunc(n_inputs=units * 2, n_layers=n_layers, n_units=n_units)
        self.rgru_d = RGRUCell(latents, units, True)
        self.rgru_l = RGRUCell(latents * (latents - 1) // 2, units, False)
        self.softplus = nn.Softplus()
        self.cls = nn.Linear(units * 2, n_class)

        self.ode = ode
        self.device = device

    def forward(self, x):
        x = x.permute(0, 1, 4, 2, 3)
        b, s, d, w, h = x.shape
        x = x.reshape(-1, d, w, h)
        x = self.cnn(x)
        x = x.reshape(b, s, self.latents, self.latents)
        x_d, x_l = self.chol_de(x)
        # x_d = self.softplus(x[:, :, :self.latents])
        # x_l = x[:, :, self.latents:]
        h_d = torch.ones(x.shape[0], self.units, device=self.device)
        h_l = torch.zeros(x.shape[0], self.units, device=self.device)
        times = torch.from_numpy(np.arange(s + 1)).float().to(self.device)
        out = []
        for i in range(x.shape[1]):
            if self.ode ==True:
                hp = odeint(self.odefunc, torch.cat((h_d.log(), h_l), dim=1), times[i:i+2], rtol=1e-4, atol=1e-5, method='euler')[1]
                h_d = hp[:, :self.units].tanh().exp()
                h_l = hp[:, self.units:]
            h_d = self.rgru_d(x_d[:, i, :], h_d)
            h_l = self.rgru_l(x_l[:, i, :], h_l)
            out.append(torch.cat((h_d.log(), h_l), dim=1))
        # h = torch.cat((h_d.log(), h_l), dim=1)
        h = torch.stack(out).mean(0)
        # return h
        return self.cls(h)

    def chol_de(self, x):
        b, s, n, n = x.shape
        x = x.reshape(-1, n, n)
        L = x.cholesky()
        d = x.new_zeros(b * s, n)
        l = x.new_zeros(b * s, n * (n - 1) // 2)
        for i in range(b * s):
            d[i] = L[i].diag()
            l[i] = torch.cat([L[i][j: j + 1, :j] for j in range(1, n)], dim=1)[0]
        return d.reshape(b, s, -1), l.reshape(b, s, -1)

class RGRUCell(nn.Module):
    """
    An implementation of RGRUCell.

    """

    def __init__(self, input_size, hidden_size, diag=True):
        super(RGRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.diag = diag
        if diag:
            layer = PosLinear
            self.nonlinear = nn.Softplus()
        else:
            layer = nn.Linear
            self.nonlinear = nn.Tanh()
        self.x2h = layer(input_size, 3 * hidden_size, bias=False)
        self.h2h = layer(hidden_size, 3 * hidden_size, bias=False)
        self.bias = nn.Parameter(torch.rand(hidden_size * 3))
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, x, hidden):
        x = x.view(-1, x.size(1))

        gate_x = self.x2h(x)
        gate_h = self.h2h(hidden)

        gate_x = gate_x.squeeze()
        gate_h = gate_h.squeeze()

        i_r, i_i, i_n = gate_x.chunk(3, 1)
        h_r, h_i, h_n = gate_h.chunk(3, 1)
        b_r, b_i, b_n = self.bias.chunk(3, 0)

        if self.diag:
            resetgate = (b_r.abs() * (i_r.log() + h_r.log()).exp()).sigmoid()
            inputgate = (b_i.abs() * (i_i.log() + h_i.log()).exp()).sigmoid()
            newgate = self.nonlinear((b_n.abs() * (i_n.log() + (resetgate * h_n).log()).exp()))
            hy = (newgate.log() * (1 - inputgate) + inputgate * hidden.log()).exp()
        else:
            resetgate = (i_r + h_r + b_r).sigmoid()
            inputgate = (i_i + h_i + b_i).sigmoid()
            newgate = self.nonlinear(i_n + (resetgate * h_n) + b_n)
            hy = newgate + inputgate * (hidden - newgate)

        return hy


class CNN(nn.Module):
    def __init__(self, latents):
        super(CNN, self).__init__()
        self.layers = nn.ModuleList()
        filters = [3, 8, 16, 32]
        for i in range(1, len(filters)):
            self.layers.append(
                nn.Sequential(
                    nn.Conv2d(in_channels=filters[i - 1], out_channels=filters[i], kernel_size=7, padding=3),
                    nn.BatchNorm2d(filters[i]),
                    nn.LeakyReLU(),
                    nn.MaxPool2d(2, 2)
                )
            )
        # self.flatten = nn.Flatten()
        # self.linear = nn.Linear(32 * 15 * 20, latents * (latents + 1) // 2)
        self.layers.append(nn.Conv2d(in_channels=filters[i],
                                     out_channels=latents,
                                     kernel_size=7,
                                     padding=3))

        self.layers.append(nn.Flatten(2, 3))
        # self.layers.append(nn.AdaptiveAvgPool2d(1))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = x.matmul(x.transpose(-1, -2))
        # x = self.flatten(x)
        # x = self.linear(x)
        return x


class PosLinear(nn.Module):
    def __init__(self, in_dim, out_dim, bias=False):
        super(PosLinear, self).__init__()
        self.weight = nn.Parameter(torch.randn((in_dim, out_dim)))

    def forward(self, x):
        return torch.matmul(x, torch.abs(self.weight))


class ODEFunc(nn.Module):
    def __init__(self, n_inputs, n_layers, n_units):
        super(ODEFunc, self).__init__()
        self.gradient_net = odefunc(n_inputs, n_layers, n_units)

    def forward(self, t_local, y, backwards=False):
        grad = self.get_ode_gradient_nn(t_local, y)
        if backwards:
            grad = -grad
        return grad

    def get_ode_gradient_nn(self, t_local, y):
        return self.gradient_net(y)

    def sample_next_point_from_prior(self, t_local, y):
        return self.get_ode_gradient_nn(t_local, y)

class odefunc(nn.Module):
    def __init__(self, n_inputs, n_layers, n_units):
        super(odefunc, self).__init__()
        self.Layers = nn.ModuleList()
        self.Layers.append(nn.Linear(n_inputs, n_units))
        for i in range(n_layers):
            self.Layers.append(
                nn.Sequential(
                    nn.Tanh(),
                    nn.Linear(n_units, n_units)
                )
            )
        self.Layers.append(nn.Tanh())
        self.Layers.append(nn.Linear(n_units, n_inputs))

    def forward(self, x):
        for layer in self.Layers:
            x = layer(x)
        return x
