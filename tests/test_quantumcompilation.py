from quantum_compilation.quantumcompilation import QuantumCompilation
from quantum_compilation.quantumcompilation import DIM, DIM_OBS, GATES, DEPTH, MAX_TARGET_DEPTH, HAS_ANCILLA
import quantum_compilation.quantumcompilation as qc
import jax
import jax.numpy as jnp

env = QuantumCompilation()
init = jax.jit(env.init)
step = jax.jit(env.step)
observe = jax.jit(env.observe)

identity = jnp.eye(DIM_OBS, dtype=jnp.complex64)
ancilla_slice = max(2*qc.N_ANCILLA,1)

key = jax.random.PRNGKey(420)

def test_init(key=key):
    state = init(key=key)
    # circuit unitary starts at identity
    u = state._circuit_unitary.reshape(DIM,DIM)
    u = jax.lax.slice(u, (0,0), (DIM, DIM), (ancilla_slice,ancilla_slice))
    assert jnp.allclose(u, identity, atol=1e-3)
    # target unitary is a unitary
    vt = state._target_unitary
    vvt = jnp.matmul(vt.conjugate().transpose(),vt)
    if HAS_ANCILLA:
        r = vvt.trace()/DIM_OBS
        vvt = vvt/r
    assert jnp.allclose(vvt, identity, atol=1e-3)

def test_step(key=key):
    _, key = jax.random.split(key)
    # we check if the gate 1 is well applied to composed_unitary
    s = init(key=key)
    u = s._circuit_unitary.reshape(DIM,DIM)
    gate = 1
    s2 = step(s, gate, key)
    u2 = s2._circuit_unitary.reshape(DIM,DIM)
    u_check = jnp.matmul(GATES[gate], u)
    diff = jnp.abs(u2-u_check)
    assert jnp.all(diff < 1e-5)
    assert s2._circuit[0] == gate

def test_terminated(key=key):
    assert DEPTH >= MAX_TARGET_DEPTH
    # Terminated when fidelity one
    # Playing a game with the correct actions should result in fid 1
    _, key = jax.random.split(key)
    s = init(key=key)
    for a in s._target_circuit[:s._target_depth]:
        s = step(s, a, key)
        if s.terminated:
            break
    assert s.rewards == 1 
    assert s.terminated
    # Terminated when reaching max depth
    _, key = jax.random.split(key)
    for a in range(DEPTH):
        s = step(s, 1, key)
    assert s.terminated

def test_observe(key=key):
    s = init(key=key)
    obs = observe(s, 0)
    assert obs.shape == (2,DIM_OBS,DIM_OBS)
    for a in s._target_circuit[:s._target_depth]:
        s = step(s, a, key)
        if s.terminated:
            break
    obs_ = observe(s, 0)
    # should match identity (up to a global phase)
    obs = obs_[0]+1j*obs_[1]
    obs = obs*jnp.exp(-1j*jnp.angle(obs[0][0]))
    assert jnp.allclose(obs, identity, atol=1e-3)

def test_all(key=key):
    test_init(key)
    test_step(key)
    test_terminated(key)
    test_observe(key)

def n_test_all(n,key=key):
    for _ in range(n):
        key, _ = jax.random.split(key)
        try:
            test_all(key)
        except:
            print(f"Key: {key}")

if __name__ == '__main__':
    n = 1_000
    print(f"Running all tests with {n} diff keys")
    n_test_all(n)
