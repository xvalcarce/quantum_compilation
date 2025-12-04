import qujax

import jax.numpy as jnp

from itertools import combinations

def load_connectivity(config):
    conn = {}
    for section in config.sections():
        if section.startswith("connectivity."):
            idx = int(section.split(".")[1])
    
            ctrl = config[section]["control"].split(",")
            tgt = config[section]["target"].split(",")
    
            conn[idx] = {
                "control": [int(c.strip()) for c in ctrl if c.strip()],
                "target": [t.strip() for t in tgt if t.strip()]
            }
    return conn

def generate_gateset(gateset, connectivity):
    n_qubits = max(list(connectivity.keys()))+1
    dim = 2**n_qubits
    gates = []
    gate_names = {}
    k = 0
    targets_per_qubit = {q: set(connectivity[q]["target"]) for q in connectivity}
    controls_per_qubit = {q: set(connectivity[q]["control"]) for q in connectivity}
    for gate in gateset:
        n_qubits_gate = gate.count("C") + 1
        if n_qubits_gate == 1:
            # Single-qubit gates
            for q in range(n_qubits):
                if gate in targets_per_qubit[q]:
                    mat = qujax.get_params_to_unitarytensor_func([gate], [[q]], [[]], n_qubits)
                    gates.append(mat().reshape(dim, dim).astype(jnp.complex64))
                    gate_names[k] = f"{gate}_{q}"
                    k += 1
        else:
            # Multi-qubit gates
            for target in range(n_qubits):
                if gate in targets_per_qubit[target]:
                    possible_controls = [q for q in range(n_qubits) if q != target and q in controls_per_qubit[target]]
                    if len(possible_controls) >= n_qubits_gate - 1:
                        # Generate all combinations of control qubits
                        for ctrl_combo in combinations(possible_controls, n_qubits_gate - 1):
                            qubits_ordered = list(ctrl_combo) + [target]
                            mat = qujax.get_params_to_unitarytensor_func([gate], [qubits_ordered], [[]], n_qubits)
                            gates.append(mat().reshape(dim, dim).astype(jnp.complex64))
                            gate_names[k] = f"{gate}_{tuple(ctrl_combo)}{target}"
                            k += 1
    return gate_names, jnp.stack(gates)
