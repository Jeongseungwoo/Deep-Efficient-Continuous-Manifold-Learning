import torch
import torch.nn as nn

import numpy as np
import math

from torchdiffeq import odeint as odeint


class ODERGRU(nn.Module):
    def __init__(self, n_class, n_layers, n_units, units, channel, latents, kernels, filters, dilation, bi, device='cpu'):
        super(ODERGRU, self).__init__()

        self.latents = latents
        self.units = units
        self.bi = bi

        self.encoder = Encoder(channel=channel, latents=latents, kernels=kernels, filters=filters, dilation=dilation)
        self.odefunc = ODEFunc(n_inputs=units * (units + 1) // 2, n_layers=n_layers, n_units=n_units)
        self.rgru_d = RGRUCell(latents, units, True)
        self.rgru_l = RGRUCell(latents * (latents - 1) // 2, units * (units - 1) // 2, False)
        if bi:
            self.odefunc_re = ODEFunc(n_inputs=units * (units + 1) // 2, n_layers=n_layers, n_units=n_units)
            self.rgru_d_re = RGRUCell(latents, units, True)
            self.rgru_l_re = RGRUCell(latents * (latents - 1) // 2, units * (units - 1) // 2, False)

        # self.att = nn.Sequential(
        #     nn.Linear(units * (units + 1) // 2, 1),
        #     nn.Tanh(),
        #     nn.Softmax(1)
        # )
        self.cls = nn.Linear(units * (units + 1) // 2, n_class)

        self.device = device

    def forward(self, x):
        x_d, x_l = self.encoder(x)
        b, t, c, c = x.shape
        h_d = torch.ones(b, self.units, device=self.device)
        h_l = torch.zeros(b, self.units * (self.units - 1) // 2, device=self.device)
        times = torch.from_numpy(np.arange(t + 1)).float().to(self.device)
        out = []
        if self.bi:
            h_d_re = torch.ones(b, self.units, device=self.device)
            h_l_re = torch.zeros(b, self.units * (self.units - 1) // 2, device=self.device)
            out_re = []

        for i in range(x.shape[1]):
            hp = odeint(self.odefunc, torch.cat((h_d.log(), h_l), dim=1), times[i:i + 2].flip(0), rtol=1e-4, atol=1e-5,
                        method='euler')[1]
            h_d = hp[:, :self.units].tanh().exp()
            h_l = hp[:, self.units:]
            h_d = self.rgru_d(x_d[:, i, :], h_d)
            h_l = self.rgru_l(x_l[:, i, :], h_l)
            out.append(torch.cat((h_d.log(), h_l), dim=1))
            if self.bi:
                hp_re = \
                odeint(self.odefunc_re, torch.cat((h_d_re.log(), h_l_re), dim=1), times[i:i + 2].flip(0), rtol=1e-4,
                       atol=1e-5, method='euler')[1]
                h_d_re = hp_re[:, :self.units].tanh().exp()
                h_l_re = hp_re[:, self.units:]
                h_d_re = self.lgru_d_re(x_d[:, x.shape[1] - i - 1, :], h_d_re)
                h_l_re = self.lgru_l_re(x_l[:, x.shape[1] - i - 1, :], h_l_re)
                out_re.append(torch.cat((h_d_re.log(), h_l_re), dim=1))
        h = torch.stack(out).mean(0)
        if self.bi:
            h_re = torch.stack(out_re, dim=1)
            h = h + h_re
        return self.cls(h).squeeze()

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


class Encoder(nn.Module):
    def __init__(self, channel=9, latents=32, kernels="2, 2, 2", filters="8, 12, 16", dilation=1):
        super(Encoder, self).__init__()
        kernels = kernels.split(",")
        filters = filters.split(",")
        filters.insert(0, channel)
        dilation = dilation
        self.layers = nn.ModuleList()
        for i in range(len(filters) - 1):
            self.layers.append(nn.Conv1d(int(filters[i]),
                                         int(filters[i + 1]),
                                         kernel_size=int(kernels[i]),
                                         stride=1,
                                         dilation=dilation))
            self.layers.append(nn.BatchNorm1d(int(filters[i + 1])))
            self.layers.append(nn.LeakyReLU())
        self.layers.append(nn.Conv1d(int(filters[i + 1]), latents, kernel_size=1, padding=1))

    def forward(self, x):
        b, s, c, t = x.shape
        x = x.reshape(-1, c, t)
        for layer in self.layers:
            x = layer(x)
        x_cov = x.new_ones(x.shape[0], x.shape[1], x.shape[1])
        for i in range(b * s):
            x_cov[i] = oas_cov(x[i].transpose(-1, -2))
        x_d, x_l = self.chol_de(x_cov)
        return x_d.reshape(b, s, -1), x_l.reshape(b, s, -1)

    def chol_de(self, x):
        b, n, n = x.shape
        x = x.reshape(-1, n, n)
        L = x.cholesky()
        d = x.new_zeros(b, n)
        l = x.new_zeros(b, n * (n - 1) // 2)
        for i in range(b):
            d[i] = L[i].diag()
            l[i] = torch.cat([L[i][j: j + 1, :j] for j in range(1, n)], dim=1)[0]
        return d.reshape(b, -1), l.reshape(b, -1)



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

def cov(x, rowvar=False, bias=False, ddof=None, aweights=None):
    """Estimates covariance matrix like numpy.cov"""
    # ensure at least 2D
    if x.dim() == 1:
        x = x.view(-1, 1)

    # treat each column as a data point, each row as a variable
    if rowvar and x.shape[0] != 1:
        x = x.t()

    if ddof is None:
        if bias == 0:
            ddof = 1
        else:
            ddof = 0

    w = aweights
    if w is not None:
        if not torch.is_tensor(w):
            w = torch.tensor(w, dtype=torch.float)
        w_sum = torch.sum(w)
        avg = torch.sum(x * (w/w_sum)[:,None], 0)
    else:
        avg = torch.mean(x, 0)

    # Determine the normalization
    if w is None:
        fact = x.shape[0] - ddof
    elif ddof == 0:
        fact = w_sum
    elif aweights is None:
        fact = w_sum - ddof
    else:
        fact = w_sum - ddof * torch.sum(w * w) / w_sum

    xm = x.sub(avg.expand_as(x))

    if w is None:
        X_T = xm.t()
    else:
        X_T = torch.mm(torch.diag(w), xm).t()

    c = torch.mm(X_T, xm)
    c = c / fact

    return c.squeeze()


def oas_cov(X):
    n_samples, n_features = X.shape
    emp_cov = cov(X)
    mu = emp_cov.diag().sum() / n_features

    alpha = (emp_cov ** 2).mean()
    num = alpha + mu ** 2
    den = (n_samples + 1.) * (alpha - (mu ** 2) / n_features)

    shrinkage = 1. if den == 0 else torch.minimum((num / den), mu.new_ones(1))
    shrunk_cov = (1. - shrinkage) * emp_cov
    shrunk_cov.flatten()[::n_features + 1] += shrinkage * mu

    return shrunk_cov

