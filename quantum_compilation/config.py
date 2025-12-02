import jax
import qujax
import jax.numpy as jnp

from quantum_compilation.gateset import load_connectivity, generate_gateset
from quantum_compilation.utils import commutations, redundancies, ancilla_first_redundant

from . import config

def check_connectivity(connectivity, n_qubits, n_anilla):
    n_q = max(list(connectivity.keys()))+1
    assert n_q == n_qubits+n_anilla, "connectivity incompatible with provided number of qubits+ancilla"

N_QUBITS = config.getint('environment','n_qubits')
N_ANCILLA = config.getint('environment','n_ancilla', fallback=0)
N_ALL = N_QUBITS+N_ANCILLA
HAS_ANCILLA = N_ANCILLA > 0
TWO_ANCILLA = max(2*N_ANCILLA,1) #1 is a fallback for non ancillary slicing, a bithackysorry
ANCILLA_INIT = config.get('environment', 'ancilla_init', fallback='0')
# Total dim, including ancilla
DIM = 1<<N_ALL #total number of qubits, including ancilla
# Dim w/o ancilla
DIM_OBS = 1<<N_QUBITS # observe dimension, i.e. final unitary space after measured ancilla
# Renormalization factor for fidelity
FID_RENORM = DIM_OBS**2
# Identity matrix w/o ancilla
EYE_DIM_OBS = jnp.eye(DIM_OBS, dtype=jnp.complex64)
# Empty circuit
EYE = jnp.eye(DIM,dtype=jnp.complex64)
if ANCILLA_INIT in ['+','T']:
    for i in range(N_QUBITS,N_ALL):
        mat_ = qujax.get_params_to_unitarytensor_func(['H'],[[i]],[[]],N_ALL)().reshape(DIM,DIM)
        EYE = jnp.matmul(mat_,EYE)
        if ANCILLA_INIT == 'T':
            mat_ = qujax.get_params_to_unitarytensor_func(['T'],[[i]],[[]],N_ALL)().reshape(DIM,DIM)
            EYE = jnp.matmul(mat_,EYE)

# Gateset generation
_gset = config['environment']['gateset'].split(',')
GATESET = [gate.strip() for gate in _gset]
CONNECTIVITY = load_connectivity(config)
check_connectivity(CONNECTIVITY,N_QUBITS,N_ANCILLA)
GATE_NAMES, GATES = generate_gateset(GATESET, CONNECTIVITY)

# Utilities
COMMUTATIONS = commutations(GATES)
REDUNDANCIES = redundancies(GATES)
ANCILLA_REDUDANCIES = ~ancilla_first_redundant(EYE, GATES, N_QUBITS, N_ANCILLA)
LENGTH_GATES = jnp.int32(len(GATES))
GATES_NUM = jnp.arange(LENGTH_GATES)
DEPTH = config.getint('environment', 'max_depth')
DD = jnp.int32(DEPTH)

# Maximum depth of random target circuits
MAX_TARGET_DEPTH = config.getint('environment', 'max_target_depth')
M_TARGET_DEPTH = MAX_TARGET_DEPTH
# Minimum depth of random target circuits
MIN_TARGET_DEPTH = config.getint('environment', 'min_target_depth')
# Fidelity considered 1 above 1-ϵ threshold
FIDELITY = config.getfloat('environment', 'target_fidelity')

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
