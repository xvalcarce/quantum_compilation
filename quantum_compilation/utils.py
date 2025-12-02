import jax
import jax.numpy as jnp

from jax import Array
from typing import List

def commutations(gates: List) -> Array:
    """
    Compute pairwise commutation relations.
    Returns an array where each row lists the indices of gates that commute
    with the corresponding gate of the row index -- padded so all rows have 
    equal length for compatibility with jax.
    """
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
    """
    Pairwise redundancies of gates, i.e. pair of gates multiplying to 
    identity. Padded for jax compatibility.
    """
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

def ancilla_first_redundant(init_state: Array, gates: list, n_qubits: int, n_ancilla: int) -> Array:
    """
    List of gates that are trivial when applied to the ancillaes, e.g. T on |0>.
    """
    if n_ancilla == 0:
        return jnp.array([],dtype=jnp.bool)
    dim = int(1<<(n_qubits+n_ancilla))
    dim_obs = int(1<<n_qubits)
    ii = jnp.eye(dim_obs, dtype=jnp.complex64)
    redudant = []
    for gate in gates:
        u = jnp.matmul(gate,init_state)
        u = jax.lax.slice(gate, (0,0), (dim, dim), (2*n_ancilla,2*n_ancilla))
        if jnp.all(u == ii):
            redudant.append(True)
        else:
            redudant.append(False)
    return jnp.array(redudant)


def is_redundant(gate, circuit: Array, len_circuit,  commutation: Array, redundancy: Array) -> bool:
    """
    Check whether a gate is redundant relative to a circuit.

    Scans backward through the circuit, stopping if a commuting but redundant
    pair is found. Returns whether a redundancy has been spotted.
    """
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
