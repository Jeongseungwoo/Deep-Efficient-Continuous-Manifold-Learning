import torch
import torch.nn as nn

import numpy as np


class TT_GRU(nn.Module):
    def __init__(self, tt_input_shape, tt_output_shape, tt_ranks, device):
        super(TT_GRU, self).__init__()

        self.units = np.prod(np.array(tt_output_shape))
        self.activation = nn.Tanh()
        self.recurrent_activation = nn.Hardsigmoid()

        self.dropout = nn.Dropout(.25)
        self.recurrent_dropout = nn.Dropout(.25)

        self.tt_input_shape = np.array(tt_input_shape)
        self.tt_output_shape = np.array(tt_output_shape)
        self.tt_output_shape[0] *= 3
        self.tt_ranks = np.array(tt_ranks)
        self.num_dim = self.tt_input_shape.shape[0]

        self.kernel = nn.Parameter(torch.rand(np.sum(self.tt_input_shape * self.tt_output_shape * self.tt_ranks[1:] * self.tt_ranks[:-1]), device=device), requires_grad=True)
        self.recurrent_kernel = nn.Parameter(torch.rand(self.units, self.units * 3, device=device), requires_grad=True)
        self.bias = nn.Parameter(torch.rand(np.prod(self.tt_output_shape), device=device), requires_grad=True)

        self.inds = np.zeros(self.num_dim, dtype=int)
        self.shapes = np.zeros((self.num_dim, 2), dtype=int)
        self.cores = [None] * self.num_dim

        for k in range(self.num_dim - 1, -1, -1):
            self.shapes[k][0] = self.tt_input_shape[k] * self.tt_ranks[k + 1]
            self.shapes[k][1] = self.tt_ranks[k] * self.tt_output_shape[k]
            self.cores[k] = self.kernel[self.inds[k]:self.inds[k] + np.prod(self.shapes[k])]
            if 0 < k:
                self.inds[k - 1] = self.inds[k] + np.prod(self.shapes[k])

        self.cls = nn.Linear(self.units, 11)
        self.device = device

    def cell(self, x, states):

        h_tm1 = states[0]
        res = self.dropout(x)

        for k in range(self.num_dim - 1, -1, -1):
            res = (res.reshape(-1, self.shapes[k][0])).matmul(
                self.cores[k].reshape(self.shapes[k][0], self.shapes[k][1]))
            res = res.reshape(-1, self.tt_output_shape[k]).transpose(-1, -2)
        res = res.reshape(-1, x.shape[0]).transpose(-1, -2)

        matrix_x = res
        # bias
        matrix_x += self.bias

        matrix_inner = (self.recurrent_dropout(h_tm1)).matmul(self.recurrent_kernel[:, :2 * self.units])
        x_z = matrix_x[:, :self.units]
        x_r = matrix_x[:, self.units: 2 * self.units]

        recurrent_z = matrix_inner[:, :self.units]
        recurrent_r = matrix_inner[:, self.units:2 * self.units]

        z = self.recurrent_activation(x_z + recurrent_z)
        r = self.recurrent_activation(x_r + recurrent_r)

        x_h = matrix_x[:, 2 * self.units:]
        recurrent_h = (r * self.recurrent_dropout(h_tm1)).matmul(self.recurrent_kernel[:, 2 * self.units:])
        hh = self.activation(x_h + recurrent_h)

        h = z * h_tm1 + (1 - z) * hh
        return h, [h]

    def forward(self, x, states=None):
        if states == None:
            states = [torch.zeros(x.shape[0], self.units, device=self.device)]

        for i in range(x.shape[1]):
            h, states = self.cell(x[:, i, :], states)
        return self.cls(h)


class TT_LSTM(nn.Module):
    def __init__(self, tt_input_shape, tt_output_shape, tt_ranks, device):
        super(TT_LSTM, self).__init__()
        self.units = np.prod(np.array(tt_output_shape))

        self.tt_input_shape = np.array(tt_input_shape)
        self.tt_output_shape = np.array(tt_output_shape)
        self.tt_output_shape[0] *= 4
        self.tt_ranks = np.array(tt_ranks)
        self.num_dim = self.tt_input_shape.shape[0]

        self.kernel = nn.Parameter(torch.rand(np.sum(self.tt_input_shape * self.tt_output_shape * self.tt_ranks[1:] * self.tt_ranks[:-1]), device=device), requires_grad=True)
        self.recurrent_kernel = nn.Parameter(torch.rand(self.units, self.units * 4, device=device), requires_grad=True)

        self.inds = np.zeros(self.num_dim, dtype=int)
        self.shapes = np.zeros((self.num_dim, 2), dtype=int)
        self.cores = [None] * self.num_dim

        for k in range(self.num_dim - 1, -1, -1):
            self.shapes[k][0] = self.tt_input_shape[k] * self.tt_ranks[k + 1]
            self.shapes[k][1] = self.tt_ranks[k] * self.tt_output_shape[k]
            self.cores[k] = self.kernel[self.inds[k]:self.inds[k] + np.prod(self.shapes[k])]
            if 0 < k:
                self.inds[k - 1] = self.inds[k] + np.prod(self.shapes[k])

        self.recurrent_activation = nn.Hardsigmoid()
        self.cls = nn.Sequential(nn.Linear(self.units, 50),
                                 nn.LeakyReLU(),
                                 nn.Linear(50, 11))
        self.activation = nn.Tanh()

        self.device = device

    def cell(self, x, states):

        h_tm1 = states[0]
        c_tm1 = states[1]

        res = x

        for k in range(self.num_dim - 1, -1, -1):
            res = (res.reshape(-1, self.shapes[k][0])) @ (self.cores[k].reshape(self.shapes[k][0], self.shapes[k][1]))
            res = res.reshape(-1, self.tt_output_shape[k]).transpose(-1, -2)
        res = res.reshape(-1, x.shape[0]).transpose(-1, -2)

        z = res

        z += h_tm1 @ self.recurrent_kernel

        z0 = z[:, :self.units]
        z1 = z[:, self.units: 2 * self.units]
        z2 = z[:, 2 * self.units: 3 * self.units]
        z3 = z[:, 3 * self.units:]

        i = self.recurrent_activation(z0)
        f = self.recurrent_activation(z1)
        c = f * c_tm1 + i * self.activation(z2)
        o = self.recurrent_activation(z3)

        h = o * self.activation(c)

        return h, [h, c]

    def forward(self, x, states=None):
        if states == None:
            states = [torch.zeros(x.shape[0], self.units, device=self.device), torch.zeros(x.shape[0], self.units, device=self.device)]

        x = x.flatten(2, 4)

        for i in range(x.shape[1]):
            h, states = self.cell(x[:, i, :], states)
        return self.cls(h)
