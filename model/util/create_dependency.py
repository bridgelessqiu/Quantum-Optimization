from qiskit import QuantumCircuit
import matplotlib.pyplot as plt
from typing import Optional
import json

"""
Date Created: 2024-11-03
Last Modified: 2024-11-03
"""
def create_dependency(input_path: str, edge_path: Optional[str] = None, attr_path: Optional[str] = None) -> None:
    r"""
    Creates the Dependency Graph of a Quantum Circuit.
    
    Args:
        input_path (required): Path to the input .qasm file
            Warning:
            The `QuantumCircuit` module does not support the 
            `ccz` gate directly. Ensure that `ccz` gates are
            decomposed into supported gates (e.g., `cx` and `h`).
            
        edge_path (optional): Path to the output .edgelist file
            Note:
            If not specified, the output file name will be derived
            from the `input_path` name.

        attr_path (optional): Path to the vertex attribute file.
            Note:
            If not specified, a file with a name associated with
            the provided `input_path` name will be created.

    Returns:
        None

    Example:
        >>> create_dependency('data/example.qasm')
        >>> # File 'example_dependency.edges' created
        >>> # File 'example_attr.txt' created

    Note:
        - Each 2-qubit gate corresponds to a vertex in the
          graph, labeled 0 to n-1. All 1-qubit gates are 
          omitted in a dependency graph.
          
        - The output graph is **directed**.
    """
    
    Q_circuit = QuantumCircuit.from_qasm_file(input_path)

    # -- Print some basic info --
    print(f"Number of qubits: {Q_circuit.num_qubits}")
    print(f"Number of gates: {Q_circuit.size()}")

    # -- Gather all 2-qubit gates --
    gates = []
    vertex_attr ={}
    for gate in Q_circuit.data:
        if gate.operation.num_qubits == 2:
            gate_type = gate.name
            qubit_indices = [Q_circuit.find_bit(qubit).index for qubit in gate.qubits]
            
            gates.append(tuple(qubit_indices))
            vertex_attr[len(gates)-1] = {
                'gate': gate_type,
                'qubit_1': qubit_indices[0],
                'qubit_2': qubit_indices[1],
            }

    # -- Create the dependency graph --
    n = len(gates)
    G = [[] for _ in range(n)]
    
    for i in range(n-1):
        for j in range(i+1, n):
            # target equals target
            if gates[i][1] == gates[j][1]:
                G[i].append(j)
                break
            # taraget equals control
            if gates[i][1] == gates[j][0]:
                G[i].append(j)

    # -- Output the graph ---
    file_anker = input_path.split('/')[-1].split('.')[0]
    if not edge_path:
        edge_path = file_anker + '_dependency.edges'
    if not attr_path:
        attr_path = file_anker + '_attr.json'
        
    # Edgelist
    with open(edge_path, 'w') as f:
        for u, neigh in enumerate(G):
            for v in neigh:
                f.write(f"{u} {v}\n")
    # JSON
    with open(attr_path, "w") as f:
        json.dump(vertex_attr, f, indent=4)