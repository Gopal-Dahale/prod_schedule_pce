from docplex.mp.model import Model
import numpy as np
from qiskit.quantum_info import SparsePauliOp
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter, ParameterVector
from qiskit.circuit.library import QAOAAnsatz


def get_qaoa_circuit(hamiltonian, reps=1, ry_init=False, measure=False):
    n = hamiltonian.num_qubits

    if ry_init:
        theta_symb = ParameterVector("θ", n)
        circuit = QuantumCircuit(n)
        for i in range(n):
            circuit.ry(theta_symb[i], i)
    else:
        circuit = QuantumCircuit(n)
        circuit.h(range(n))

    qaoa_circuit = QAOAAnsatz(
        cost_operator=hamiltonian,
        reps=reps,
        initial_state=circuit,
    )

    if measure:
        qaoa_circuit.measure_all()

    return qaoa_circuit


def transpile(pm, qc, ham, noise=False):
    isa_qc = pm.run(qc)
    isa_ham = ham.apply_layout(isa_qc.layout)
    return isa_qc, isa_ham


def makespan(durations, n_jobs, m_machines, seq):
    C = [[0] * m_machines for _ in range(n_jobs)]
    for i, j in enumerate(seq):
        for k in range(m_machines):
            C[i][k] = durations[j][k] + max(C[i - 1][k] if i else 0,
                                            C[i][k - 1] if k else 0)
    return C[-1][-1]


def evaluate_samples(durations, n_jobs, m_machines, samples):
    valid_samples = []
    valid_sequences = []
    energies = []
    probabilities = []
    for idx, bitstring in enumerate(samples):
        valid = True
        for i in range(n_jobs):
            if sum(bitstring[i * n_jobs:(i + 1) * n_jobs]) != 1:
                valid = False
                break
        for j in range(n_jobs):
            if sum(bitstring[j::n_jobs]) != 1:
                valid = False
                break
        if valid:
            valid_samples.append(bitstring)
            seq = seq = [
                next(j
                     for j in range(n_jobs)
                     if bitstring[i * n_jobs + j] > 0.5)
                for i in range(n_jobs)
            ]
            valid_sequences.append(seq)
            energies.append(int(makespan(durations, n_jobs, m_machines, seq)))

    valid_sequences = np.array(valid_sequences)

    # remove dupliate valid_sequences and then remove the corresponding valid_samples and energies
    unique_sequences, unique_indices = np.unique(valid_sequences,
                                                 axis=0,
                                                 return_index=True)
    valid_sequences = [valid_sequences[i] for i in unique_indices]
    valid_samples = [valid_samples[i] for i in unique_indices]
    energies = np.array([energies[i] for i in unique_indices])

    # sort the valid samples and sequences based on energies
    sorted_indices = np.argsort(energies)
    energies = energies[sorted_indices]
    valid_samples = [valid_samples[i] for i in sorted_indices]
    valid_sequences = [valid_sequences[i] for i in sorted_indices]

    # find probablity of each valid_sample in the original samples
    for sample in valid_samples:
        count = sum(np.array_equal(sample, s) for s in samples)
        probabilities.append(count / len(samples))

    return valid_samples, valid_sequences, energies, probabilities


to_int_list = lambda bs: list(map(int, list(bs)))


def circuit_sampling(circuit, params, sampler, durations, n_jobs, m_machines,
                     n_shots):
    pub = (circuit.assign_parameters(params),)
    job = sampler.run([pub], shots=n_shots)
    sampling_result = job.result()[0]
    memory = sampling_result.data.meas.get_bitstrings()
    memory = np.array([to_int_list(i)[::-1] for i in memory])
    valid_samples, valid_sequences, energies, probabilities = evaluate_samples(
        durations, n_jobs, m_machines, memory)
    return valid_samples, valid_sequences, energies, probabilities


def get_sim_and_pm(SEED):
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    from qiskit_aer import AerSimulator

    backend_options = {
        "precision": "single",
        "max_parallel_threads": 12,
        "max_parallel_experiments": 0,
        "max_parallel_shots": 0,
        "fusion_enable": True,
        "fusion_threshold": 14,
        "fusion_max_qubit": 4,
        "matrix_product_state_max_bond_dimension": 16,
        "matrix_product_state_truncation_threshold": 1e-8,
        "mps_sample_measure_algorithm": "mps_apply_measure",
        "mps_parallel_threshold": 12,
        "mps_omp_threads": 12,
        "mps_lapack": False,
        "seed_simulator": SEED,
    }

    aer_sim = AerSimulator(method="matrix_product_state", **backend_options)
    aer_sim.set_max_qubits(200)

    # Create pass manager for transpilation
    pm = generate_preset_pass_manager(backend=aer_sim,
                                      optimization_level=1,
                                      seed_transpiler=SEED)

    return aer_sim, pm


def read_taillard_instances(instance_path) -> array_like:
    """Read all Taillard instances from a file.

    Returns:
        A list of arrays, one per instance, each shaped
        (num_machines, num_jobs)
    """
    instances = []
    lower_bounds = []

    with open(instance_path, "r") as f:
        while True:
            # Find the next instance header
            header = f.readline()
            if not header:
                break

            # Skip the descriptive header line
            line = f.readline()
            if not line:
                break

            values = line.split()
            num_jobs = int(values[0])
            num_machines = int(values[1])
            lower_bound = int(values[4])

            lower_bounds.append(lower_bound)

            # Skip the "processing times :" line
            f.readline()

            # Read the processing-time matrix
            processing_times = []
            for _ in range(num_machines):
                line = f.readline()
                if not line:
                    raise ValueError("Unexpected end of file")

                times = list(map(int, line.split()))
                if len(times) != num_jobs:
                    raise ValueError("Invalid number of jobs in a row")

                processing_times.append(times)

            instances.append(np.array(processing_times))

    return instances, lower_bounds


def load_or_library_instances(instance_path) -> list[dict]:
    """Read the OR library formatted Flow Shop Schedule problems file.

    Args:
        instance_path: path to the flow shop instance file

    Returns:
        list[dict]: list containing the processing times as values and labels as keys
    """
    instances = {}
    label = None
    processing_times = []
    num_jobs = num_machines = None
    expect_problem = expect_label = False

    def _store_instance(processing_times):
        """Store processing_times as instance."""
        # stored as num_jobs x num_machines
        assert len(processing_times) == num_jobs
        assert len(processing_times[0]) == num_machines

        processing_times_array = np.array(processing_times).T
        instances[label] = processing_times_array
        processing_times.clear()

    with open(instance_path) as f:
        for line in f:
            if line.isspace():
                continue

            if "END OF DATA" in line:
                # scan done; store final instance and break
                _store_instance(processing_times)
                break

            elif line.strip() == "+++++++++++++++++++++++++++++":
                # if separator is found while expecting a problem,
                # it's the end of the problem and we store it
                if expect_problem:
                    _store_instance(processing_times)

                # if separator is found while expecting a label
                # it's the end of label and thus start of problem
                elif expect_label:
                    # skip name row and move on to size row
                    _ = next(f)

                    line = next(f)
                    num_jobs, num_machines = tuple(map(int, line.split()))

                expect_problem = expect_label
                expect_label = not expect_label

            elif expect_label and line[:9].strip() == "instance":
                label = line[10:].strip()

            elif expect_problem:
                # times are every second value (rest are machine number)
                times = list(map(int, line.split()))[1::2]
                processing_times.append(times)

        return instances


from pyjobshop.Solution import Solution, ScheduledTask


def solution_from_sequence(model, durations, job_sequence):
    data = model.data()
    n_machines = len(durations[0])
    machine_end = [0] * n_machines
    scheduled = [None] * len(data.tasks)

    for job in job_sequence:
        prev_end = 0
        for machine in range(n_machines):
            idx = job * n_machines + machine
            start = max(prev_end, machine_end[machine])
            end = start + durations[job][machine]

            scheduled[idx] = ScheduledTask(mode=idx,
                                           resources=[machine],
                                           start=start,
                                           end=end)
            machine_end[machine] = prev_end = end

    return Solution(data, scheduled)


import json


def read_base_instance(path):
    with open(path) as f:
        data = json.load(f)
    return np.array(
        [[op["duration"] for op in job["operations"]] for job in data["jobs"]])
