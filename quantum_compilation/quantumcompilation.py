import jax
import jax.numpy as jnp

import pgx.core as core
from pgx._src.struct import dataclass
from pgx._src.types import Array, PRNGKey

from typing import NamedTuple, Optional

from quantum_compilation.gateset import generate_gate_all_to_all, generate_gate_with_ancilla
from quantum_compilation.gateset import commutations, redundancies, is_redundant, ancilla_first_redundant

from . import config

N_QUBITS = config.getint('environment','n_qubits')
N_ANCILLA = config.getint('environment','n_ancilla', fallback=0)
HAS_ANCILLA = N_ANCILLA > 0
TWO_ANCILLA = max(2*N_ANCILLA,1) #1 is a fallback for non ancillary slicing, a bithackysorry
DIM = 2**(N_QUBITS+N_ANCILLA) #total number of qubits, including ancilla
DIM_OBS = 2**N_QUBITS # observe dimension, i.e. final unitary space after measured ancilla
FID_RENORM = DIM**2
EYE_DIM_OBS = jnp.eye(DIM_OBS, dtype=jnp.complex64)
_gset = config['environment']['gateset'].split(',')
GATESET = [gate.strip() for gate in _gset]

if HAS_ANCILLA:
    GATE_NAMES, GATES = generate_gate_with_ancilla(GATESET, N_QUBITS, N_ANCILLA)
    ANCILLA_REDUDANCIES = ~ancilla_first_redundant(GATES, N_QUBITS, N_ANCILLA)
else:
    GATE_NAMES, GATES = generate_gate_all_to_all(GATESET, N_QUBITS)

COMMUTATIONS = commutations(GATES)
REDUNDANCIES = redundancies(GATES)
LENGTH_GATES = jnp.int32(len(GATES))
GATES_NUM = jnp.arange(LENGTH_GATES)
DEPTH = config.getint('environment', 'max_depth')
DD = jnp.int32(DEPTH)

MAX_TARGET_DEPTH = config.getint('environment', 'max_target_depth')
M_TARGET_DEPTH = MAX_TARGET_DEPTH
MIN_TARGET_DEPTH = config.getint('environment', 'min_target_depth')
FIDELTY = config.getfloat('environment', 'target_fidelity')

FALSE = jnp.bool_(False)
TRUE = jnp.bool_(True)
ZERO = jnp.int32(0)

if config.getboolean('environment', 'use_normal', fallback=False):
    MEAN_TARGET_DEPTH = config.getint('environment', 'mean_target_depth')+1
    M_TARGET_DEPTH = MEAN_TARGET_DEPTH
    STD_DEPTH = config.getint('environment', 'std_depth')
    def random_depth(key, m_target_depth=MEAN_TARGET_DEPTH):
        depth = jnp.minimum(jnp.maximum(jnp.int32(m_target_depth+STD_DEPTH*jax.random.normal(key)), MIN_TARGET_DEPTH), MAX_TARGET_DEPTH)
        return depth
else:
    def random_depth(key, m_target_depth=MAX_TARGET_DEPTH):
        d = jax.random.randint(key, (1,), MIN_TARGET_DEPTH, m_target_depth+1)[0]
        return d

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
    _renormalization: Array = jnp.float32(0.0)
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
    # we store the renormalization factor instead of v =/ r, to enhance precision
    r = jnp.linalg.norm(v) 
    return State(_target_unitary = v.conjugate().transpose(),
                 _target_circuit = gates,
                 _target_depth = d,
                 _renormalization = r,
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
    rewards, terminated = _reward(u,state._target_unitary,state._renormalization)
    reached_target_depth = state._step_count >= state._target_depth
    reached_max_depth = state._step_count >= DEPTH
    #TODO: make sure this is compatible with penalty
    terminated = jnp.logical_or(terminated, reached_target_depth)
    terminated = jnp.logical_or(terminated, reached_max_depth)
    return state.replace( #type: ignore
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
    def _reward(u: Array, vt: Array, rvt = 1.0):
        #TODO: implement penalty for depth and/or certain gate type
        # slice for |0>, |0>
        u = jax.lax.slice(u, (0,0), (DIM,DIM), (TWO_ANCILLA,TWO_ANCILLA))
        # renormalization factor
        r = jnp.linalg.norm(u)
        uvt = jnp.matmul(u,vt)
        fid = jnp.square(jnp.abs(uvt.trace()/(r*rvt)))
        r = fid > FIDELTY
        return jnp.float32([r]), r
else:
    def _reward(u: Array, vt: Array, rvt = 1.0):
        # hacky but here rvt is just the normalization of the fidelity
        #TODO: implement penalty for depth and/or certain gate type
        uvt = jnp.matmul(u,vt)
        fid = jnp.square(jnp.abs(uvt.trace()))/FID_RENORM
        r = fid > FIDELTY
        return jnp.float32([r]), r

if HAS_ANCILLA: 
    def _observe(state: State, player_id) -> Array:
        u = state._circuit_unitary.reshape(DIM,DIM)
        u = jax.lax.slice(u, (0,0), (DIM,DIM), (TWO_ANCILLA,TWO_ANCILLA))
        uvt = jnp.matmul(u,state._target_unitary)
        r = jnp.round(DIM_OBS/(jnp.linalg.norm(u)*state._renormalization),decimals=4)
        obs = uvt*r
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
    circuit = jnp.zeros(DEPTH, dtype=jnp.int32) #Could use MAX_TARGET_DEPTH, but hack to increase MAX_TARGET_DEPTH during learning

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
        uu = jnp.matmul(u.conjugate().transpose(),u)
        r = uu.trace()/DIM_OBS
        is_u = jnp.allclose(uu/r, EYE_DIM_OBS, atol=1e-3)
        return ~is_u

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

