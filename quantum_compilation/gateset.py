import qujax

import jax.numpy as jnp
import jax
from jax import Array
from typing import List, Dict

def generate_gate_all_to_all(gate_seq: List[str], n_qubits: int) -> tuple[Dict,List]:
    dim = 2**n_qubits
    gates = []
    gate_names = {}
    k = 0
    for g in gate_seq:
        # control gates
        if g[0] == 'C':
            if g[1] == 'C':
                for i in range(n_qubits):
                    for j in range(i+1,n_qubits):
                        for l in range(n_qubits):
                            if i!=l and j!=l:
                                mat = qujax.get_params_to_unitarytensor_func([g],[[i,j,l]],[[]],n_qubits)
                                gates.append(mat().reshape(dim,dim).astype(jnp.complex64))
                                gate_names[k] = f"{g}_({i},{j}){l}"
                                k+=1
            else:
                for i in range(n_qubits):
                    for j in range(n_qubits):
                        if i!=j:
                            mat = qujax.get_params_to_unitarytensor_func([g],[[i,j]],[[]],n_qubits)
                            gates.append(mat().reshape(dim,dim).astype(jnp.complex64))
                            gate_names[k] = f"{g}_({i}){j}"
                            k+=1
        # single gates
        else:
            for i in range(n_qubits):
                mat = qujax.get_params_to_unitarytensor_func([g],[[i]],[[]],n_qubits)
                gates.append(mat().reshape(dim,dim).astype(jnp.complex64))
                gate_names[k] = f"{g}_{i}"
                k+=1
    return gate_names,jnp.stack(gates)

def generate_gate_with_ancilla(gate_seq: List[str], n_qubits: int, n_ancilla: int) -> tuple[Dict, List]:
    dim = 2**(n_qubits + n_ancilla)  # Total dimension including ancilla
    gates = []
    gate_names = {}
    k = 0
    for g in gate_seq:
        # Control gates
        if g[0] == 'C':
            # Control gates can have the following configurations:
            # 1. Control on qubits, target on ancilla
            # 2. Control on ancilla, target on qubits
            # 3. Control and target both on ancilla
            for control in range(n_qubits + n_ancilla):
                for target in range(n_qubits + n_ancilla):
                    if control != target:
                        # Ensure the valid configurations
                        if (control < n_qubits and target >= n_qubits) or \
                           (control >= n_qubits and target < n_qubits) or \
                           (control >= n_qubits and target >= n_qubits):
                            mat = qujax.get_params_to_unitarytensor_func([g], [[control, target]], [[]], n_qubits + n_ancilla)
                            gates.append(mat().reshape(dim, dim).astype(jnp.complex64))
                            gate_names[k] = f"{g}_({control}){target}"
                            k += 1
        # Single-qubit gates
        else:
            # Only applied to ancilla
            for i in range(n_qubits, n_qubits + n_ancilla):
                mat = qujax.get_params_to_unitarytensor_func([g], [[i]], [[]], n_qubits + n_ancilla)
                gates.append(mat().reshape(dim, dim).astype(jnp.complex64))
                gate_names[k] = f"{g}_{i}"
                k += 1
    return gate_names, jnp.stack(gates)

def commutations(gates: List) -> Array:
    dim = gates[0].shape
    commute = []
    # testing commutation relations
    l = len(gates)
    for i in range(l):
        c = [] 
        for j in range(l):
            if (gates[i]@gates[j] - gates[j]@gates[i] == jnp.zeros(dim)).all():
                c.append(j)
        commute.append(c)
    # padding by repeating last value
    max_len = max([len(_) for _ in commute])
    for c in commute:
        diff = max_len-len(c)
        for i in range(diff):
            c.append(c[-1])
    return jnp.array(commute)

def redundancies(gates: list) -> Array:
    id = jnp.eye(gates[0].shape[0], dtype=jnp.complex64)
    redundant = []
    l = len(gates)
    for i in range(l):
        r = []
        for j in range(l):
            if jnp.allclose(gates[i]@gates[j],id,atol=1e-3):
                r.append(j)
        redundant.append(r)
    max_len = max([len(_) for _ in redundant])
    for c in redundant:
        diff = max_len-len(c)
        for i in range(diff):
            c.append(c[-1])
    return jnp.array(redundant)

def ancilla_first_redundant(gates: list, n_qubits: int, n_ancilla: int) -> Array:
    dim = int(2**(n_qubits+n_ancilla))
    dim_obs = int(2**n_qubits)
    ii = jnp.eye(dim_obs, dtype=jnp.complex64)
    redudant = []
    for gate in gates:
        g = jax.lax.slice(gate, (0,0), (dim, dim), (2*n_ancilla,2*n_ancilla))
        if jnp.all(g == ii):
            redudant.append(True)
        else:
            redudant.append(False)
    return jnp.array(redudant)


def is_redundant(gate, circuit: Array, len_circuit,  commutation: Array, redundancy: Array) -> bool:
    i = len_circuit-1

    # stop when reaching the begging of circuit or if redundancy detected
    def cond_fn(state):
        i, commutes, redudant = state
        return (i >= 0) & commutes & ~redudant

    def body_fn(state):
        i, commutes, redudant = state
        c_gate = circuit[i]
        # is redudant with c_gate
        redundant = jnp.isin(gate, redundancy[c_gate])
        # commutes with c_gate
        commutes = jnp.isin(c_gate,commutation[gate])
        # condition is commute & not redudant
        return i-1, commutes, redundant

    _, commutes, redundant = jax.lax.while_loop(cond_fn, body_fn, (i, True, False))
    return redundant
