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
    controls = [constraints['control'] for index, constraints in connectivity.items()]
    for gate in gateset:
        num_indices = gate.count('C') + 1
        if num_indices == 1:
            # Single-qubit operations
            for index, constraints in connectivity.items():
                if gate in constraints['target']:
                    mat = qujax.get_params_to_unitarytensor_func([gate],[[index]],[[]],n_qubits)
                    gates.append(mat().reshape(dim,dim).astype(jnp.complex64))
                    gate_names[k] = f"{gate}_{index}"
                    k+=1
        else:
            # Control operation
            for target_index, constraints in connectivity.items():
                if gate in constraints['target']:
                    i_control = [i for i, c in enumerate(controls) if target_index in c]
                    if len(i_control) >= num_indices-1:
                        for ctrls in combinations(i_control,num_indices-1):
                            mat = qujax.get_params_to_unitarytensor_func([gate],[list(ctrls)+[target_index]],[[]],n_qubits)
                            gates.append(mat().reshape(dim,dim).astype(jnp.complex64))
                            operation_name = f"{gate}_{ctrls}{target_index}"
                            gate_names[k] = operation_name
                            k+=1
    return gate_names, jnp.stack(gates)
