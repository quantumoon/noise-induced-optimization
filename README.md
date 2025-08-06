**Noise‑Induced Optimization**

This repository accompanies the paper [*Regularizing Quantum Loss Landscapes by Noise Injection*](https://arxiv.org/abs/2505.08759).
The code here reproduces the experiments from that article. In our numerical experiments we used NVIDIA RTX A6000 with 48 GB of memory.

**Repository structure**
```
.
├── QCNN/
│   ├── QCNN.py                         # definition of the quantum convolutional neural network
│   ├── matrix_utils.py                 # helper functions
│   ├── simulators.py                   # state and mixture simulators with unitary and noise channels
│   ├── qcnn_pytorch_experiments.ipynb  # PyTorch notebook for training a QCNN with and without noise
│   └── tests.py                        # simple tests validating the simulators and noise channel
└── Wishart/
    ├── wishart_loss.py                 # functions to generate Wishart matrices and define loss/noisy loss
    ├── wishart_experiments.ipynb       # JAX notebook with optimization
```
**Dependencies**
```
- Jax (0.4.30)
- Jaxopt (0.8.5)
- Optax (0.1.8)
- Numpy (1.26.4)
- Matplotlib (3.8.2)
- Tqdm (4.66.1)
- Pytorch (2.4.0)
- Pandas (2.1.4)
- Seaborn (0.13.1)
```
