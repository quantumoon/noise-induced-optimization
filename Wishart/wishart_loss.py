import numpy as np
import jax.numpy as jnp
from functools import reduce

def gen_wishart_batch(dim, rank, num_matrices, seed=42):
    np.random.seed(seed=seed)
    X = [np.random.randn(dim, rank, 2).view(np.complex128).reshape(dim, rank) for _ in range(num_matrices)]
    W = [1/rank * (x @ x.conj().T) for x in X]
    return W
    

def gen_loss(dim, wmatrix):
    def loss(params):
        w = [jnp.array([jnp.cos(params[i] / 2), jnp.sin(params[i] / 2)]) for i in range(dim)]
        w = reduce(jnp.kron, w)
        return jnp.real(w.T @ (wmatrix @ w))
    return loss


def gen_noisy_loss(dim, wmatrix):
    def noisy_loss(params, mu):
        w = [jnp.array([[1. + (1 - mu) * jnp.cos(params[i]), (1 - mu) * jnp.sin(params[i])],
                        [(1 - mu) * jnp.sin(params[i]), 1. - (1 - mu) * jnp.cos(params[i])]]) for i in range(dim)]
        w = reduce(jnp.kron, w)
        return jnp.real(jnp.trace(jnp.matmul(jnp.transpose(wmatrix), w))) / (1 << dim)
    return noisy_loss