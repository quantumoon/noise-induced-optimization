import torch
import numpy as np
import torch.nn as nn
from matrix_utils import *


def random_init(A):
    rand_nums = torch.randn(A.shape, dtype=A.dtype, device=A.device)
    with torch.no_grad():
        A.copy_(rand_nums)
    return A


def _2local_conv(state, U):
    for i in range(state.n // 2):
        state.apply_unitary(U, [2 * i, 2 * i + 1])
    if state.n > 2:
        for i in range(state.n // 2):
            state.apply_unitary(U, [2 * i + 1, (2 * i + 2) % state.n])
    return state


def _2local_pool(state):
    if state.n >= 2:
        for i in range(state.n // 2):
            _ = state.measure(i)
    return state


def _noise(state, mu):
    if mu > 0.:
        state.apply_noise(mu)
    return state


class QCNN(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.noise_on = False
        self.L = int(np.ceil(np.log2(n)))
        
        self.params = nn.ParameterDict()
        for i in range(self.L):
            self.params["H{}".format(i)] = nn.Parameter(torch.randn(4, 4).type(torch.complex64))
        self.reset_parameters()

    def reset_parameters(self):
        for a in self.params:
            random_init(self.params[a])

    def forward(self, state, mu=0.):
        for i, p in enumerate(self.params):
            U = torch.matrix_exp(self.params[p] - conjugate_transpose(self.params[p]))
            if self.noise_on:
                state = _noise(state, mu)
                state = state.convert_to_density_matrix()
                state = _2local_conv(state, U)
                state = _2local_pool(state)
            else:
                state = _2local_conv(state, U)
                state = _2local_pool(state)
                state = state.convert_to_density_matrix()
        
        out = state.measure(0)
        return out[:, 0].real