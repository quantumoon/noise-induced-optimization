from simulators import *
from matrix_utils import *


def test_both_simulators():
    batch_size = 10
    for q in range(1, 7):
        data_ids = get_random_computational_ids(q, batch_size)
        data_states = convert_id_to_state(data_ids, q)
        data_dm = convert_id_to_dm(data_ids, q)

        sms = state_mixture_simulator(q, state_init=data_states).convert_to_density_matrix()
        ss = state_simulator(q, state_init=data_dm)
        assert torch.all(sms.state == ss.state).item()


def test_1q_noise():
    q = 1
    batch_size = 1
    for _ in range(100):
        a = torch.rand(1)
        b, c = torch.randn(2)
        initial_density_matrix = torch.tensor([[a, b + 1j * c],
                                               [b - 1j * c, 1 - a]], dtype=torch.complex64)
        
        sim = state_simulator(q, state_init=initial_density_matrix.unsqueeze(0))
        mu = torch.rand(1).item()
        sim.apply_noise(mu)
        final_state_simulator = sim.state

        X = torch.tensor([[0, 1],
                          [1, 0]], dtype=torch.complex64)
        
        final_state_explicit = (1 - mu) * initial_density_matrix + mu * X @ initial_density_matrix @ X
        
        assert torch.allclose(final_state_simulator.squeeze(0), final_state_explicit)


def test_nq_noise():
    from functools import reduce
    I = torch.eye(2, dtype=torch.complex64)
    X = torch.tensor([[0, 1],
                      [1, 0]], dtype=torch.complex64)
    for n in range(2, 9):
        dim = 1 << n
        for _ in range(50):
            M = torch.randn((dim, dim), dtype=torch.complex64)
            M = (M + conjugate_transpose(M)) / 2
            new_diag = torch.rand(1<<n)
            init_rho = M - torch.diagflat(torch.diag(M) - new_diag / torch.sum(new_diag).item())

            mu = torch.rand(1).item()
            sim = state_simulator(n, state_init=init_rho.reshape([1] + [2] * 2 * n))
            sim.apply_noise(mu)
            final_state_simulator = sim.state.reshape(1<<n, 1<<n)

            for q in range(n):
                X_q = reduce(torch.kron, [I if i != q else X for i in range(n)])
                init_rho = (1 - mu) * init_rho + mu * X_q @ init_rho @ X_q
            assert np.allclose(torch.sum(torch.abs(final_state_simulator - init_rho)**2).item().real, 0)