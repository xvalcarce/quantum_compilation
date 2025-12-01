import jax
import jax.numpy as jnp

import pgx.core as core
from pgx._src.struct import dataclass
from pgx._src.types import Array, PRNGKey

from quantum_compilation.config import DIM, DIM_OBS, TWO_ANCILLA, LENGTH_GATES, DEPTH, M_TARGET_DEPTH, GATES, GATES_NUM, GATE_NAMES, FALSE, EYE_DIM_OBS, COMMUTATIONS, REDUNDANCIES, HAS_ANCILLA, ANCILLA_REDUDANCIES, FID_RENORM, FIDELITY, random_depth
from quantum_compilation.utils import is_redundant

@dataclass
class State(core.State):
    current_player: Array = jnp.int32(0) # single-player game
    observation: Array = jnp.eye(DIM_OBS, dtype=jnp.float32).ravel()
    rewards: Array = jnp.float32([0.0])
    terminated: Array = FALSE
    truncated: Array = FALSE
    legal_action_mask: Array = jnp.ones(LENGTH_GATES, dtype=jnp.bool_)
    _step_count: Array = jnp.int32(0)
    _circuit_unitary: Array = jnp.eye(DIM, dtype=jnp.complex64).ravel()
    _target_unitary: Array = jnp.eye(DIM_OBS, dtype=jnp.complex64).ravel()
    _circuit: Array = jnp.zeros(DEPTH, dtype=jnp.int32)
    _target_circuit: Array = jnp.int32([])
    _target_depth: Array = jnp.int32(0)

    @property
    def env_id(self) -> core.EnvId:
        return "quantum_compilation"

class QuantumCompilation(core.Env):
    def __init__(self):
        super().__init__()

    def _init(self, key: PRNGKey, m_target_depth=M_TARGET_DEPTH) -> State:
        return _init(key, m_target_depth=m_target_depth)

    def _step(self, state: core.State, action: Array, key) -> State:
        assert isinstance(state, State)
        return _step(state, action, key)
    
    def _observe(self, state: core.State, player_id: Array) -> Array:
        assert isinstance(state, State)
        return _observe(state, player_id)

    @property
    def id(self) -> core.EnvId:
        return "quantum_compilation"

    @property
    def version(self) -> str:
        return "v0.0.1"

    @property
    def num_players(self) -> int:
        return 1

def _init(rng: PRNGKey, m_target_depth=M_TARGET_DEPTH) -> State:
    rng1, rng2 = jax.random.split(rng)
    d = random_depth(rng1, m_target_depth=m_target_depth)
    # we generate MAX_TARGET_DEPTH gates, that's cause static array is needed by jit
    gates = rand_cir(d, rng2)
    # building target circuit
    v = jnp.eye(DIM, dtype=jnp.complex64)
    v = jax.lax.fori_loop(0, d, lambda i,v: jnp.matmul(GATES[gates[i]],v), v) 
    # This performs identity if N_ANCILLA == 0, else slice |0> in, |0> out on ancillaes
    v = jax.lax.slice(v, (0,0), (DIM,DIM), (TWO_ANCILLA,TWO_ANCILLA))
    # renormalize
    v = v/jnp.linalg.norm(v, ord=2) 
    return State(_target_unitary = v.conjugate().transpose(),
                 _target_circuit = gates,
                 _target_depth = d,
                 legal_action_mask = _legal_action_mask(gates,0)) # for ancilla case, not trivial 

def _init_u(u: Array, target_depth: int) -> State:
    # max target_depth before game stops
    d = jnp.array(target_depth, dtype=jnp.int32)
    # dummy circuit
    gates = jnp.zeros(target_depth, dtype=jnp.int32)
    return State(_target_unitary = u.conjugate().transpose(),
                 _target_circuit = gates,
                 _target_depth = d,
                 legal_action_mask = _legal_action_mask(gates,0)) # for ancilla case, not trivial 

def _step(state: State, action, key):
    # reshape unitary to matrix
    u = state._circuit_unitary.reshape(DIM,DIM)
    # apply new gate (action)
    u = jnp.matmul(GATES[action],u)
    # append action to circuit
    c = state._circuit.at[state._step_count-1].set(action)
    # generate new action mask
    legal_action_mask = _legal_action_mask(c, state._step_count)
    # compute the reward/terminated
    rewards, terminated = _reward(u,state._target_unitary)
    reached_target_depth = state._step_count >= state._target_depth
    reached_max_depth = state._step_count >= DEPTH
    #TODO: make sure this is compatible with penalty
    terminated = jnp.logical_or(terminated, reached_target_depth)
    terminated = jnp.logical_or(terminated, reached_max_depth)
    return state.replace(
            _circuit_unitary = u.ravel(),
            _circuit = c,
            rewards = rewards, 
            legal_action_mask = legal_action_mask,
            terminated = terminated)

def _legal_action_mask(circuit: Array, len_circuit):
    if HAS_ANCILLA:
        # Using ancilla init in 0, CX_(ancilla) and phase gates too
        def null_circuit(circuit):
            return ANCILLA_REDUDANCIES
    else:
        def null_circuit(circuit):
            return jnp.ones(LENGTH_GATES, dtype=jnp.bool_)

    def nonnull_circuit(circuit):
        f_is_red = jax.vmap(lambda x: is_redundant(x, circuit, len_circuit, COMMUTATIONS, REDUNDANCIES))
        legal = ~f_is_red(GATES_NUM)
        return legal

    return jax.lax.cond(len_circuit == 0,
                         null_circuit,
                         nonnull_circuit,
                         circuit)

if HAS_ANCILLA:
    def _reward(u: Array, vt: Array):
        #TODO: implement penalty for depth and/or certain gate type
        # slice for |0>, |0>
        u = jax.lax.slice(u, (0,0), (DIM,DIM), (TWO_ANCILLA,TWO_ANCILLA))
        # renormalization factor
        r = jnp.linalg.norm(u, ord=2)
        uvt = jnp.matmul(u,vt)
        fid = jnp.square(jnp.abs(uvt.trace()/r))/FID_RENORM
        f = fid > FIDELITY
        return jnp.float32([f]), f
else:
    def _reward(u: Array, vt: Array):
        # hacky but here rvt is just the normalization of the fidelity
        #TODO: implement penalty for depth and/or certain gate type
        uvt = jnp.matmul(u,vt)
        fid = jnp.square(jnp.abs(uvt.trace()))/FID_RENORM
        f = fid > FIDELITY
        return jnp.float32([f]), f

if HAS_ANCILLA: 
    def _observe(state: State, player_id) -> Array:
        u = state._circuit_unitary.reshape(DIM,DIM)
        u = jax.lax.slice(u, (0,0), (DIM,DIM), (TWO_ANCILLA,TWO_ANCILLA))
        uvt = jnp.matmul(u,state._target_unitary)
        obs = uvt/jnp.linalg.norm(u, ord=2)
        obs = jnp.append(obs.real.ravel(), obs.imag.ravel())
        return obs.reshape((2,DIM_OBS,DIM_OBS))
else:
    def _observe(state: State, player_id) -> Array:
        u = state._circuit_unitary.reshape(DIM,DIM)
        obs = jnp.matmul(u,state._target_unitary)
        obs = jnp.append(obs.real.ravel(), obs.imag.ravel())
        return obs.reshape((2,DIM_OBS,DIM_OBS))

@jax.jit
def random_gate(legal_gates, rng: PRNGKey):
    def cond_fn(state):
        gate, _ = state
        return gate == -1

    def body_fn(state):
        _, rng = state
        _, rng = jax.random.split(rng)
        gate = jax.random.choice(rng, legal_gates)
        return gate, rng

    gate, _ = jax.lax.while_loop(cond_fn, body_fn, (-1, rng))
    return gate

@jax.jit
def random_circuit(d, rng: PRNGKey) -> Array:
    circuit = jnp.zeros(DEPTH, dtype=jnp.int32) #could use max_target_depth, but hack to increase max_target_depth during learning

    def body_fn(i, state):
        circuit, rng = state
        _, rng = jax.random.split(rng)
        gates = _legal_action_mask(circuit, i)
        legal_gates = jnp.where(gates, size=LENGTH_GATES, fill_value=-1)[0]
        gate = random_gate(legal_gates, rng)
        circuit = circuit.at[i].set(gate)
        return (circuit, rng)
    
    circuit, _ = jax.lax.fori_loop(0, d, body_fn, (circuit, rng))
    return circuit

@jax.jit
def random_circuit_ancilla(d, rng: PRNGKey) -> Array:
    def cond_fn(state):
        circuit, _ = state
        u = jnp.eye(DIM, dtype=jnp.complex64)
        u = jax.lax.fori_loop(0, d, lambda i,u: jnp.matmul(GATES[circuit[i]],u), u)     
        u = jax.lax.slice(u, (0,0), (DIM,DIM), (TWO_ANCILLA,TWO_ANCILLA))
        # avoid trivial unitary
        is_trivial = jnp.allclose(u/jnp.linalg.norm(u, ord=2), EYE_DIM_OBS, atol=1e-3)
        uu = jnp.matmul(u.conjugate().transpose(),u)
        r = uu.trace()/DIM_OBS
        # make sure produced target is a unitary
        is_u = jnp.allclose(uu/r, EYE_DIM_OBS, atol=1e-3)
        return ~(is_u*~is_trivial)

    def body_fn(state):
        circuit, rng = state
        _, rng = jax.random.split(rng)
        circuit = random_circuit(d,rng)
        return (circuit, rng)

    circuit, _ = jax.lax.while_loop(cond_fn, body_fn, (random_circuit(d, rng), rng))
    return circuit

if HAS_ANCILLA:
    rand_cir = random_circuit_ancilla
else:
    rand_cir = random_circuit

def print_circuit(circuit: Array, len_circuit: int):
    " ; ".join([GATE_NAMES[g] for g in circuit.tolist()[:len_circuit]])

