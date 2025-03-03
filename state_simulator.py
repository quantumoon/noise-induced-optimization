import torch
from matrix_utils import *


class state_simulator:
    # simulator which simulates states as density matrices
    def __init__(self,n,batch_size = 1, state_init = None, device = None, dtype = torch.complex64):
        self.dtype = dtype
        self.n = n
        if device is None:
            if state_init is not None:
                self.device = state_init.device
            else:
                self.device = 'cpu'
        else:
            self.device = device
        if state_init is None:
            self.state = torch.zeros([batch_size, 2**n, 2**n],device = self.device, dtype = dtype)
            self.state[:,0,0] = 1
            self.state = torch.reshape(self.state, [batch_size]+[2]*(2*n))
            self.batch_size = batch_size
        else:
            self.state = state_init.clone().detach().requires_grad_(True)
            self.batch_size = state_init.shape[0]

    def apply_unitary(self, U,qubits = [0]):
        self.state = self.apply_matrix_(self.state, U, qubits = qubits)

    def apply_matrix_(self, state, U, qubits = [0]):
        U = U.to(self.device)
        qubit_ids = [i+1 for i in qubits]
        full_ids = list(range(self.n*2+1))
        for i,qubit in enumerate(qubits):
            full_ids[qubit+1] = self.n*2+1+i
        state = torch.einsum(state,
                             list(range(self.n*2+1)),
                             torch.reshape(U,[2]*(len(qubits)*2)),
                             [self.n*2+1+i for i in range(len(qubits))]+qubit_ids,
                             full_ids)

        qubit_ids = [i+1 for i in qubits]
        full_ids = list(range(self.n*2+1))
        # full_ids = [0] + [i for i in range(self.n+1,3*self.n+1)] 
        for i,qubit in enumerate(qubits):
            full_ids[qubit+1+self.n] = self.n*2+1+i
        state = torch.einsum(state,
                             list(range(self.n*2+1)),
                             torch.reshape(conjugate_transpose(U),
                                           [2]*(len(qubits)*2)),
                             [i+self.n for i in qubit_ids]+ [self.n*2+1+i for i in range(len(qubits))],
                             full_ids)
        return state

    def measure(self, qubit, update_state = True, post_unitary = None, post_unitary_qubit = 0):
        einsum_ids = [i for i in range(self.n+1)] + [i+1 for i in range(self.n)]
        einsum_ids[self.n+1+qubit] = self.n+1
        measurement = torch.diagonal(torch.einsum(self.state, einsum_ids ), dim1 = -1, dim2 = -2)

        if update_state:
            einsum_ids = [i for i in range(2*self.n+1)]
            if post_unitary is not None:
                post_unitary = post_unitary.to(self.device)
                einsum_ids[1+qubit+self.n] = 1+qubit
                out_ids = [i for i in range(2*self.n+1)]
                out_ids.pop(1+qubit+self.n)
                out_ids.pop(1+qubit)
                out_ids = [1+qubit]+out_ids
                temp_state = torch.einsum(self.state, einsum_ids, out_ids)
                self.n -=1
                if post_unitary_qubit >= qubit:
                    post_unitary_qubit -= 1
                self.state = (self.apply_matrix_(temp_state[0], post_unitary[0], qubits=[post_unitary_qubit]) + \
                              self.apply_matrix_(temp_state[1], post_unitary[1], qubits=[post_unitary_qubit]))

            else:
                einsum_ids[1+qubit+self.n] = 1+qubit
                self.state = torch.einsum(self.state, einsum_ids)
                self.n -=1

        return measurement

    def apply_noise(self, mu):
        I = torch.eye(2, dtype=self.dtype, device=self.device)
        X = torch.tensor([[0, 1], [1, 0]], dtype=self.dtype, device=self.device)
        
        K0 = torch.sqrt(torch.tensor(1.0 - mu, dtype=self.dtype, device=self.device)) * I
        K1 = torch.sqrt(torch.tensor(mu, dtype=self.dtype, device=self.device)) * X
    
        state = self.state
        for qubit in range(self.n):
            new_state = torch.zeros_like(state)
            for K in [K0, K1]:
                new_state = new_state + self.apply_matrix_(state, K, qubits=[qubit])
            state = new_state
        self.state = state
        
    def convert_to_density_matrix(self):
        return self


class state_mixture_simulator:
    def __init__(self,n,batch_size = 1, state_init = None, probs_init = None, device = None, dtype = torch.complex64):
        self.dtype = dtype
        self.n = n
        if device is None:
            if state_init is not None:
                self.device = state_init.device
            else:
                self.device = 'cpu'
        else:
            self.device = device
        if state_init is None:
            self.state = torch.zeros([batch_size, 1, 2**n],device = self.device, dtype = dtype)
            self.state[:,:,0] = 1
            self.state = torch.reshape(self.state, [batch_size,1]+[2]*n)
            self.batch_size = batch_size
        else:
            self.state = state_init.clone().detach()
            self.batch_size = state_init.shape[0]
        if probs_init is None:
            self.probs = torch.ones(self.state.shape[:2] ,device = self.device, dtype = dtype)/self.state.shape[1]
        else:
            self.probs = probs

    def apply_unitary(self, U,qubits = [0]):
        self.state = self.apply_matrix_(self.state, U, qubits = qubits)

    def apply_matrix_(self, state, U, qubits = [0]):
        U = U.to(self.device)
        qubit_ids = [i+2 for i in qubits]
        full_ids = list(range(self.n+2))
        for i,qubit in enumerate(qubit_ids):
            full_ids[qubit] = self.n+2+i
        if len(U.shape) == 3:
            U_qubits = [0] + [self.n+2+i for i in range(len(qubits))]+qubit_ids
            U_reshape = [-1] + [2]*(len(qubits)*2)
        elif len(U.shape) == 2:
            U_qubits = [self.n+2+i for i in range(len(qubits))]+qubit_ids
            U_reshape = [2]*(len(qubits)*2)
        state = torch.einsum(state,
                             list(range(self.n+2)),
                             torch.reshape(U,U_reshape),
                             U_qubits, full_ids)
        return state

    def copy(self):
        return state_mixture_simulator(self.n, state_init = self.state.detach().clone(), device = self.state.device)

    def measure(self, qubit, update_state = True, post_unitary = None, post_unitary_qubits = [0]):
        einsum_ids = [i for i in range(self.n+2)] 
        measurement = torch.einsum(self.state, einsum_ids, torch.conj(self.state),einsum_ids, [0,1,qubit+2] )
        if update_state:
            self.probs = torch.reshape(torch.einsum(self.probs,
                                                    [0,1],
                                                    measurement.real,
                                                    [0,1,2],
                                                    [0,1,2]),
                                       [self.batch_size,-1])
            einsum_ids = [i for i in range(self.n+2)]
            einsum_ids[2+qubit] = 2
            einsum_ids[2] = qubit+2
            self.state = torch.permute(self.state, einsum_ids)
            self.state = torch.einsum(self.state,
                                      list(range(self.n+2)),
                                      1./(torch.sqrt(measurement)+1e-8),
                                      [0,1,2], list(range(self.n+2)))
            self.n -=1
            if post_unitary is not None:
                post_unitary = post_unitary.to(self.device)
                self.state[:,:,0] = self.apply_matrix_(self.state[:,:,0], post_unitary[0], post_unitary_qubits)
                self.state[:,:,1] = self.apply_matrix_(self.state[:,:,1], post_unitary[1], post_unitary_qubits)

            self.state = torch.reshape(self.state, [self.batch_size, -1]+[2]*self.n)

        return measurement
    
    def apply_noise(self, mu):
        
        # It works only when initial mixture contains just one state vector |j>
        B = self.state.shape[0]
        n = self.n

        state_flat = self.state.reshape(B, 2**n)
        initial_index = state_flat.to(torch.float32).argmax(dim=1)  # shape: [B]
        noise_patterns = torch.arange(2**n, device=self.device)  # shape: [2**n]

        bits = ((noise_patterns[:, None] >> torch.arange(n, device=self.device)) & 1).to(torch.float32)
        weight = bits.sum(dim=1)  # shape: [2**n]
        probs_noise = ( (1 - mu) ** (n - weight) ) * ( mu ** weight )  # shape: [2**n]

        new_indices = initial_index.unsqueeze(1) ^ noise_patterns.unsqueeze(0)  # shape: [B, 2**n]

        one_hot_new = torch.nn.functional.one_hot(new_indices, num_classes=2**n)
        new_state = one_hot_new.reshape(B, 2**n, *([2] * n)).to(self.dtype)

        self.state = new_state
        self.probs = probs_noise.unsqueeze(0).repeat(B, 1)
    
    def convert_to_density_matrix(self):
        temp_state = torch.einsum(self.state, list(range(self.n+2)), torch.sqrt(self.probs), [0,1], list(range(self.n+2)))
        return state_simulator(self.n,
                               batch_size = self.batch_size, 
                               state_init = torch.einsum(temp_state, list(range(self.n+2)),
                                                         torch.conj(temp_state), [0,1] + [i+2+self.n for i in range(self.n)],
                                                         [0] + [i+2 for i in range(2*self.n)]),
                               device = self.device)