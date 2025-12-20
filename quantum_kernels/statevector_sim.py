from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Callable, Optional, Sequence, Dict

# Basic single-qubit gates
I2 = np.array([[1,0],[0,1]], dtype=np.complex128)
X = np.array([[0,1],[1,0]], dtype=np.complex128)
Y = np.array([[0,-1j],[1j,0]], dtype=np.complex128)
Z = np.array([[1,0],[0,-1]], dtype=np.complex128)
H = (1/np.sqrt(2))*np.array([[1,1],[1,-1]], dtype=np.complex128)

def RX(theta: float) -> np.ndarray:
    c = np.cos(theta/2); s = np.sin(theta/2)
    return np.array([[c, -1j*s],[-1j*s, c]], dtype=np.complex128)

def RY(theta: float) -> np.ndarray:
    c = np.cos(theta/2); s = np.sin(theta/2)
    return np.array([[c, -s],[s, c]], dtype=np.complex128)

def RZ(theta: float) -> np.ndarray:
    return np.array([[np.exp(-1j*theta/2), 0],[0, np.exp(1j*theta/2)]], dtype=np.complex128)

def kron_n(ops: Sequence[np.ndarray]) -> np.ndarray:
    out = ops[0]
    for op in ops[1:]:
        out = np.kron(out, op)
    return out

def apply_single(state: np.ndarray, gate: np.ndarray, qubit: int, n_qubits: int) -> np.ndarray:
    """Apply single-qubit gate using tensor reshaping (O(2^n) instead of O(4^n)).

    This is significantly faster than full Kronecker product for n > 4.
    """
    # Reshape state to (2, 2, ..., 2) tensor
    shape = [2] * n_qubits
    psi = state.reshape(shape)

    # Apply gate by contracting over the target qubit axis
    # Move target qubit to last axis, apply gate, move back
    psi = np.moveaxis(psi, qubit, -1)
    psi = np.tensordot(psi, gate, axes=([-1], [1]))
    psi = np.moveaxis(psi, -1, qubit)

    return psi.reshape(-1)

def apply_cnot(state: np.ndarray, control: int, target: int, n_qubits: int) -> np.ndarray:
    """Apply CNOT gate using direct bit manipulation (faster than full matrix).

    CNOT flips target qubit when control qubit is |1>.
    
    IMPORTANT: Uses MSB convention to match apply_single's tensor reshaping.
    Qubit 0 is the most significant bit (leftmost in tensor indices).
    Qubit n-1 is the least significant bit (rightmost in tensor indices).
    """
    dim = 2**n_qubits
    result = state.copy()

    # Convert to MSB convention: qubit k corresponds to bit position (n-1-k)
    ctrl_bit = n_qubits - 1 - control
    tgt_bit = n_qubits - 1 - target
    
    ctrl_mask = 1 << ctrl_bit
    tgt_mask = 1 << tgt_bit

    for i in range(dim):
        if (i & ctrl_mask) and not (i & tgt_mask):
            # control=1, target=0 -> swap with control=1, target=1
            j = i | tgt_mask
            result[i], result[j] = state[j], state[i]

    return result

def apply_single_slow(state: np.ndarray, gate: np.ndarray, qubit: int, n_qubits: int) -> np.ndarray:
    """Original Kronecker-based implementation (kept for reference/testing)."""
    ops = [I2]*n_qubits
    ops[qubit] = gate
    U = kron_n(ops)
    return U @ state

def measure_probs(state: np.ndarray, n_qubits: int) -> np.ndarray:
    p = np.abs(state)**2
    return p

def state_inner(a: np.ndarray, b: np.ndarray) -> complex:
    return np.vdot(a, b)

def fidelity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(state_inner(a,b))**2)

@dataclass
class Circuit:
    """A minimal statevector circuit."""
    n_qubits: int

    def zero_state(self) -> np.ndarray:
        s = np.zeros((2**self.n_qubits,), dtype=np.complex128)
        s[0] = 1.0
        return s

    def run(self, ops: List[Tuple[str, Tuple]]) -> np.ndarray:
        state = self.zero_state()
        for name, args in ops:
            if name == "H":
                (q,) = args
                state = apply_single(state, H, q, self.n_qubits)
            elif name == "RX":
                q, th = args
                state = apply_single(state, RX(th), q, self.n_qubits)
            elif name == "RY":
                q, th = args
                state = apply_single(state, RY(th), q, self.n_qubits)
            elif name == "RZ":
                q, th = args
                state = apply_single(state, RZ(th), q, self.n_qubits)
            elif name == "CNOT":
                c, t = args
                state = apply_cnot(state, c, t, self.n_qubits)
            else:
                raise ValueError(f"Unknown op: {name}")
        return state
