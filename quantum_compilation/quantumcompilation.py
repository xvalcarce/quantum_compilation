import jax
import jax.numpy as jnp

import pgx.core as core
from pgx._src.struct import dataclass
from pgx._src.types import Array, PRNGKey

from typing import NamedTuple, Optional

from quantum_compilation.gateset import generate_gate_all_to_all, commutations, redundancies, is_redundant

from . import config

N_QUBITS = config.getint('environment','n_qubits')
N_ANCILLA = config.getint('environment','n_ancilla')
DIM = 2**N_QUBITS
_gset = config['environment']['gateset'].split(',')
GATESET = [gate.strip() for gate in _gset]
GATE_NAMES, GATES = generate_gate_all_to_all(GATESET, N_QUBITS)
COMMUTATIONS = commutations(GATES)
REDUNDANCIES = redundancies(GATES)
LENGTH_GATES = jnp.int32(len(GATES))
GATES_NUM = jnp.arange(LENGTH_GATES)
DEPTH = config.getint('environment', 'max_depth')
DD = jnp.int32(DEPTH)

MAX_TARGET_DEPTH = config.getint('environment', 'max_target_depth')
MIN_TARGET_DEPTH = config.getint('environment', 'min_target_depth')
FIDELTY = config.getfloat('environment', 'target_fidelity')

FALSE = jnp.bool_(False)
TRUE = jnp.bool_(True)
ZERO = jnp.int32(0)

@dataclass
class State(core.State):
    current_player: Array = jnp.int32(0) # single-player game
    observation: Array = jnp.eye(DIM, dtype=jnp.float32).ravel()
    rewards: Array = jnp.float32([0.0])
    terminated: Array = FALSE
    truncated: Array = FALSE
    legal_action_mask: Array = jnp.ones(LENGTH_GATES, dtype=jnp.bool_)
    _step_count: Array = jnp.int32(0)
    _composed_unitary: Array = jnp.eye(DIM, dtype=jnp.complex64).ravel() # U.V*, bad precision should be fine
    _circuit: Array = jnp.zeros(DEPTH, dtype=jnp.int32)
    _target_circuit: Array = jnp.int32([])
    _target_depth: Array = jnp.int32(0)

    @property
    def env_id(self) -> core.EnvId:
        return "quantum_compilation"

class QuantumCompilation(core.Env):
    def __init__(self):
        super().__init__()

    def _init(self, key: PRNGKey) -> State:
        return _init(key)

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

def _init(rng: PRNGKey) -> State:
    rng1, rng2 = jax.random.split(rng)
    d = jax.random.randint(rng1, (1,), MIN_TARGET_DEPTH, MAX_TARGET_DEPTH+1)[0]
    # we generate MAX_TARGET_DEPTH gates, that's cause static array is needed by jit
    gates = random_circuit(d, rng2)
    u = jnp.eye(DIM, dtype=jnp.complex64)
    u = jax.lax.fori_loop(0, d, lambda i,u: jnp.matmul(GATES[gates[i]],u), u) 
    return State(_composed_unitary = u.conjugate().transpose().ravel(),
                 _target_circuit = gates,
                 _target_depth = d,
                 legal_action_mask = jnp.ones(LENGTH_GATES, dtype=jnp.bool_))

def _step(state: State, action, key):
    u = state._composed_unitary.reshape(DIM,DIM)
    u = jnp.matmul(GATES[action],u)
    c = state._circuit.at[state._step_count-1].set(action)
    legal_action_mask = _legal_action_mask(c, state._step_count)
    rewards, terminated = _reward(u)
    reached_target_depth = state._step_count >= state._target_depth
    reached_max_depth = state._step_count >= DEPTH
    #TODO: make sure this is compatible with penalty
    terminated = jnp.logical_or(terminated, reached_target_depth)
    terminated = jnp.logical_or(terminated, reached_max_depth)
    return state.replace( #type: ignore
            _composed_unitary = u.ravel(),
            _circuit = c,
            rewards = rewards, 
            legal_action_mask = legal_action_mask,
            terminated = terminated)

def _legal_action_mask(circuit: Array, len_circuit):
    f_is_red = jax.vmap(lambda x: is_redundant(x, circuit, len_circuit, COMMUTATIONS, REDUNDANCIES))
    legal = ~f_is_red(GATES_NUM)
    return legal

def _reward(u: Array):
    #TODO: implement penalty for depth and/or certain gate type
    r = (u.trace()/DIM) > FIDELTY
    return jnp.float32([r]), r

def _observe(state: State, player_id) -> Array:
    obs = state._composed_unitary
    obs = jnp.append(obs.real.ravel(), obs.imag.ravel())
    return obs.reshape((2,DIM,DIM))

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

def random_circuit(d, rng: PRNGKey) -> Array:
    circuit = jnp.zeros(MAX_TARGET_DEPTH, dtype=jnp.int32)

    def body_fn(i, state):
        circuit, rng = state
        _, rng = jax.random.split(rng)
        gates = _legal_action_mask(circuit, i)
        legal_gates = jnp.where(gates, size=20, fill_value=-1)[0]
        gate = random_gate(legal_gates, rng)
        circuit = circuit.at[i].set(gate)
        return (circuit, rng)

    circuit, _ = jax.lax.fori_loop(0, d, body_fn, (circuit, rng))
    return circuit
