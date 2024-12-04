from quantum_compilation.quantumcompilation import QuantumCompilation, State
from quantum_compilation.quantumcompilation import DIM, GATES, DEPTH, MAX_TARGET_DEPTH
import jax
import jax.numpy as jnp

env = QuantumCompilation()
init = jax.jit(env.init)
step = jax.jit(env.step)
observe = jax.jit(env.observe)


identity = jnp.eye(DIM, dtype=jnp.complex64)

def test_init():
    key = jax.random.PRNGKey(0)
    state = init(key=key)
    # Check unitary
    u = state._composed_unitary.reshape(DIM,DIM)
    diff = jnp.abs(jnp.matmul(u.conjugate().transpose(),u)-identity)
    assert jnp.all(diff < 1e-3)

def test_step():
    key = jax.random.PRNGKey(1)
    _, key = jax.random.split(key)
    # we check if the gate 1 is well applied to composed_unitary
    s = init(key=key)
    u = s._composed_unitary.reshape(DIM,DIM)
    gate = 1
    s2 = step(s, gate, key)
    u2 = s2._composed_unitary.reshape(DIM,DIM)
    u_check = jnp.matmul(GATES[gate], u)
    diff = jnp.abs(u2-u_check)
    assert jnp.all(diff < 1e-5)
    assert s2._circuit[0] == gate

def test_terminated():
    key = jax.random.PRNGKey(42)
    assert DEPTH >= MAX_TARGET_DEPTH
    # Terminated when fidelity one
    _, key = jax.random.split(key)
    s = init(key=key)
    for a in s._target_circuit[:s._target_depth]:
        s = step(s, a, key)
    assert s.rewards == 1
    assert s.terminated
    # Terminated when reaching max depth
    _, key = jax.random.split(key)
    for a in range(DEPTH):
        s = step(s, 1, key)
    assert s.terminated

def test_observe():
    key = jax.random.PRNGKey(42)
    _, key = jax.random.split(key)
    s = init(key=key)
    obs = observe(s, 0)
    assert obs.shape == (2,DIM,DIM)
    for a in s._target_circuit[:s._target_depth]:
        s = step(s, a, key)
    obs_ = observe(s, 0)
    # should match identity
    assert jnp.allclose(obs_[0],jnp.eye(DIM), atol=1e-3) # real
    assert jnp.allclose(obs_[1],jnp.zeros((DIM,DIM)), atol=1e-3) # complex
    
    

