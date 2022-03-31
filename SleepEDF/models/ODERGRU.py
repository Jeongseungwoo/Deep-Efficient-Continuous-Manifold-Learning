import torch
import torch.nn as nn

import numpy as np
import math

from torchdiffeq import odeint as odeint


class ODERGRU(nn.Module):
    def __init__(self, n_class, n_layers, n_units, latents, units, channel, bi, device='cpu'):
        super(ODERGRU, self).__init__()

        self.latents = latents
        self.units = units
        self.bi = bi

        self.cnn = F1(C=channel, latents=latents)
        # self.odefunc = ODEFunc(n_inputs=units * (units + 1) // 2, n_layers=n_layers, n_units=n_units)
        self.rgru_d = RGRUCell(latents, units, True)
        self.rgru_l = RGRUCell(latents * (latents - 1) // 2, units * (units - 1) // 2, False)
        if bi:
            self.odefunc_re = ODEFunc(n_inputs=units * (units + 1) // 2, n_layers=n_layers, n_units=n_units)
            self.rgru_d_re = RGRUCell(latents, units, True)
            self.rgru_l_re = RGRUCell(latents * (latents - 1) // 2, units * (units - 1) // 2, False)

        self.softplus = nn.Softplus()
        # self.att = nn.Sequential(
        #     nn.Linear(units * (units + 1) // 2, 1),
        #     nn.Tanh(),
        #     nn.Softmax(1)
        # )
        self.cls = nn.Linear(units * (units + 1) // 2, n_class)

        self.device = device

    def forward(self, x):
        x = self.cnn(x)
        b, s, c, c = x.shape
        x_d, x_l = self.chol_de(x)
        h_d = torch.ones(b, self.units, device=self.device)
        h_l = torch.zeros(b, self.units * (self.units - 1) // 2, device=self.device)
        times = torch.from_numpy(np.arange(s + 1)).float().to(self.device)
        out = []
        if self.bi:
            h_d_re = torch.ones(b, self.units, device=self.device)
            h_l_re = torch.zeros(b, self.units * (self.units - 1) // 2, device=self.device)
            out_re = []
        hp = torch.cat((h_d.log(), h_l), dim=1)
        hp_re = torch.cat((h_d_re.log(), h_l_re), dim=1)
        for i in range(x.shape[1]):
            h_d = hp[:, :self.units].tanh().exp()
            h_l = hp[:, self.units:]
            h_d = self.rgru_d(x_d[:, i, :], h_d)
            h_l = self.rgru_l(x_l[:, i, :], h_l)
            out.append(torch.cat((h_d.log(), h_l), dim=1))
            if self.bi:
                h_d_re = hp_re[:, :self.units].tanh().exp()
                h_l_re = hp_re[:, self.units:]
                h_d_re = self.rgru_d_re(x_d[:, x.shape[1] - i - 1, :], h_d_re)
                h_l_re = self.rgru_l_re(x_l[:, x.shape[1] - i - 1, :], h_l_re)
                out_re.append(torch.cat((h_d_re.log(), h_l_re), dim=1))
        h = torch.stack(out, dim=1)
        if self.bi:
            h_re = torch.stack(out_re, dim=1)
            h = h + h_re
        # a = self.att(h)
        # c = a * h
        return self.cls(h).squeeze()

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

class F1(nn.Module):
    def __init__(self, C, latents):
        super(F1, self).__init__()
        M = 32
        self.filter_bank = filter_bank(in_f=129, out_f=M)
        self.layers = nn.ModuleList()
        filters = [M * C, M * C, latents]
        for i in range(len(filters)-1):
            self.layers.append(nn.Conv1d(filters[i],
                                         filters[i+1],
                                         kernel_size=5))
            self.layers.append(nn.BatchNorm1d(filters[i+1]))
            self.layers.append(nn.LeakyReLU())

    def forward(self, x):
        # x.shape : [b, s, c, t, f]
        x = self.filter_bank(x) # [B, S, C, F, T] -> [B, S, C*D, T]
        b, s, _, _ = x.shape
        x = x.flatten(0, 1)
        for layer in self.layers:
            x = layer(x)
        _, d, t = x.shape
        x = x.reshape(b, s, d, t)
        cov = x.new_ones(b, s, d, d)
        for i in range(b):
            for j in range(s):
                cov[i][j] = oas_cov(x[i][j].transpose(-1, -2))
        return cov


class filter_bank(nn.Module):
    def __init__(self, in_f, out_f):
        super(filter_bank, self).__init__()
        self.fw = nn.Parameter(torch.randn(in_f, out_f).sigmoid())

        filterbank = FilterbankShape()
        self.s = filterbank.lin_tri_filter_shape(nfilt=out_f, nfft=256, samplerate=100)

    def forward(self, x):
        x = x.transpose(-1, -2)
        x = torch.matmul(x, self.fw * torch.from_numpy(self.s).float().to(x.device))
        return x.transpose(-1, -2).flatten(2, 3)


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


import numpy as np
import os

class FilterbankShape():

    def tri_filter_shape(self, ndim, nfilter):
        f = np.arange(ndim)
        f_high = f[-1]
        f_low = f[0]
        H = np.zeros((nfilter, ndim))

        M = f_low + np.arange(nfilter+2)*(f_high-f_low)/(nfilter+1)
        for m in range(nfilter):
            k = np.logical_and(f >= M[m], f <= M[m+1])   # up-slope
            H[m][k] = 2*(f[k]-M[m]) / ((M[m+2]-M[m])*(M[m+1]-M[m]))
            k = np.logical_and(f >= M[m+1], f <= M[m+2]) # down-slope
            H[m][k] = 2*(M[m+2] - f[k]) / ((M[m+2]-M[m])*(M[m+2]-M[m+1]))

        H = np.transpose(H)
        H.astype(np.float32)
        return H

    def hz2mel(self, hz):
        """Convert a value in Hertz to Mels
        :param hz: a value in Hz. This can also be a numpy array, conversion proceeds element-wise.
        :returns: a value in Mels. If an array was passed in, an identical sized array is returned.
        """
        return 2595 * np.log10(1+hz/700.)

    def mel2hz(self, mel):
        """Convert a value in Mels to Hertz
        :param mel: a value in Mels. This can also be a numpy array, conversion proceeds element-wise.
        :returns: a value in Hertz. If an array was passed in, an identical sized array is returned.
        """
        return 700*(10**(mel/2595.0)-1)

    def mel_tri_filter_shape(self, nfilt=20, nfft=512, samplerate=16000, lowfreq=0, highfreq=None):
        """Compute a Mel-filterbank. The filters are stored in the rows, the columns correspond
        to fft bins. The filters are returned as an array of size nfilt * (nfft/2 + 1)
        :param nfilt: the number of filters in the filterbank, default 20.
        :param nfft: the FFT size. Default is 512.
        :param samplerate: the samplerate of the signal we are working with. Affects mel spacing.
        :param lowfreq: lowest band edge of mel filters, default 0 Hz
        :param highfreq: highest band edge of mel filters, default samplerate/2
        :returns: A numpy array of size nfilt * (nfft/2 + 1) containing filterbank. Each row holds 1 filter.
        """
        highfreq = highfreq or samplerate/2
        assert highfreq <= samplerate/2, "highfreq is greater than samplerate/2"

        # compute points evenly spaced in mels
        lowmel = self.hz2mel(lowfreq)
        highmel = self.hz2mel(highfreq)
        melpoints = np.linspace(lowmel,highmel,nfilt+2)
        # our points are in Hz, but we use fft bins, so we have to convert
        #  from Hz to fft bin number
        bin = np.floor((nfft+1)*self.mel2hz(melpoints)/samplerate)

        fbank = np.zeros([nfilt,nfft//2+1])
        for j in range(0,nfilt):
            for i in range(int(bin[j]), int(bin[j+1])):
                fbank[j,i] = (i - bin[j]) / (bin[j+1]-bin[j])
            for i in range(int(bin[j+1]), int(bin[j+2])):
                fbank[j,i] = (bin[j+2]-i) / (bin[j+2]-bin[j+1])
        fbank = np.transpose(fbank)
        fbank.astype(np.float32)
        return fbank

    def lin_tri_filter_shape(self, nfilt=20, nfft=512, samplerate=16000, lowfreq=0, highfreq=None):
        """Compute a linear-filterbank. The filters are stored in the rows, the columns correspond
        to fft bins. The filters are returned as an array of size nfilt * (nfft/2 + 1)
        :param nfilt: the number of filters in the filterbank, default 20.
        :param nfft: the FFT size. Default is 512.
        :param samplerate: the samplerate of the signal we are working with. Affects mel spacing.
        :param lowfreq: lowest band edge of mel filters, default 0 Hz
        :param highfreq: highest band edge of mel filters, default samplerate/2
        :returns: A numpy array of size nfilt * (nfft/2 + 1) containing filterbank. Each row holds 1 filter.
        """
        highfreq = highfreq or samplerate/2
        assert highfreq <= samplerate/2, "highfreq is greater than samplerate/2"

        # compute points evenly spaced in mels
        #lowmel = self.hz2mel(lowfreq)
        #highmel = self.hz2mel(highfreq)
        #melpoints = np.linspace(lowmel,highmel,nfilt+2)
        hzpoints = np.linspace(lowfreq,highfreq,nfilt+2)
        # our points are in Hz, but we use fft bins, so we have to convert
        #  from Hz to fft bin number
        bin = np.floor((nfft+1)*hzpoints/samplerate)

        fbank = np.zeros([nfilt,nfft//2+1])
        for j in range(0,nfilt):
            for i in range(int(bin[j]), int(bin[j+1])):
                fbank[j,i] = (i - bin[j]) / (bin[j+1]-bin[j])
            for i in range(int(bin[j+1]), int(bin[j+2])):
                fbank[j,i] = (bin[j+2]-i) / (bin[j+2]-bin[j+1])
        fbank = np.transpose(fbank)
        fbank.astype(np.float32)
        return fbank
