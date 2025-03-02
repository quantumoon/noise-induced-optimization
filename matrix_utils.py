import torch
import numpy as np


def conjugate_transpose(H):
    return torch.conj(torch.transpose(H,-1,-2))


def random_init(A):
    rand_nums = torch.randn(A.shape, dtype=A.dtype, device=A.device)
    with torch.no_grad():
        # A.copy_(torch.matrix_exp(rand_nums - conjugate_transpose(rand_nums)))
        A.copy_(rand_nums)
    return A


def get_random_computational_ids(n_qubits, n_states=32):
    return np.random.randint(2 ** n_qubits, size=(n_states))


def convert_id_to_dm(ids, n_qubits):
    N = len(ids)
    dim = 2 ** n_qubits
    out = np.zeros((N, dim, dim), dtype=np.complex64)
    for i, id_i in enumerate(ids):
        out[i, id_i, id_i] = 1.
    out_tensor = torch.tensor(out, dtype=torch.complex64)
    out_tensor = out_tensor.reshape([N] + [2] * (2 * n_qubits))
    return out_tensor


def convert_id_to_state(ids, n_qubits):
    out = np.zeros((len(ids), 1, 2**n_qubits))
    for i, id_i in enumerate(ids):
        out[i,0,id_i] = 1.
    return torch.tensor(out.reshape([len(ids),1] + [2]*(n_qubits))).type(torch.complex64)