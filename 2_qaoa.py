# %%

import warnings

warnings.filterwarnings('ignore')

from docplex.mp.model import Model
import numpy as np
from qiskit_optimization.translators import from_docplex_mp
from qiskit_optimization.converters import QuadraticProgramToQubo
from qiskit_aer.primitives import SamplerV2 as Sampler
from utils import get_qaoa_circuit, transpile, circuit_sampling, get_sim_and_pm
from qiskit_algorithms.minimum_eigensolvers.diagonal_estimator import _DiagonalEstimator as DEstimator
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from utils import read_base_instance, solution_from_sequence
from plots import custom_plot_machine_gantt
from pyjobshop import Model as JobShopModel
from itertools import pairwise
import time

SEED = 10598
np.random.seed(SEED)
rng = np.random.default_rng(SEED)

from qiskit_algorithms.utils import algorithm_globals

algorithm_globals.random_seed = SEED

# %%

# base 2x2
DURATIONS = read_base_instance("input/base2_2.json")

# base 3x3
DURATIONS = read_base_instance("input/base3_3.json")

num_jobs, num_machines = len(DURATIONS), len(DURATIONS[0])
print(f"Problem size: {num_jobs} jobs, {num_machines} machines")

JOB_NAMES = [f"{i+1}" for i in range(num_jobs)]
MACHINE_NAMES = [f"Machine {i+1}" for i in range(num_machines)]

# %%

p = DURATIONS
A = 10
n, m = len(p), len(p[0])
mdl = Model("position_qubo")
x = {
    (i, j): mdl.binary_var(name=f"x_{i}_{j}") for i in range(n) for j in range(n)
}

onehot = [mdl.sum(x[i, j] for j in range(n)) - 1 for i in range(n)]
onehot += [mdl.sum(x[i, j] for i in range(n)) - 1 for j in range(n)]
Econstraints = mdl.sum(e * e for e in onehot)
Ecost = mdl.sum(p[j][k] * x[i + 1, j] - p[j][k + 1] * x[i, j]
                for k in range(m - 1)
                for i in range(n - 1)
                for j in range(n))

mdl.minimize(A * sum(map(sum, p)) * Econstraints + Ecost)
sol = mdl.solve(log_output=False)

if sol:
    seq = [
        next(j
             for j in range(n)
             if sol.get_value(x[i, j]) > 0.5)
        for i in range(n)
    ]
    seq = np.array(seq)
    print("Objective:", sol.objective_value)
    print("Sequence:", seq + 1)

    C = [[0] * m for _ in range(n)]
    for i, j in enumerate(seq):
        for k in range(m):
            C[i][k] = p[j][k] + max(C[i - 1][k] if i else 0,
                                    C[i][k - 1] if k else 0)
    print("Makespan:", C[-1][-1])

else:
    print("No solution:", mdl.solve_details.status)
# %%
qp = from_docplex_mp(mdl)
qp2qubo = QuadraticProgramToQubo()
qubo = qp2qubo.convert(qp)
qubit_op, offset = qubo.to_ising()
print("Offset:", offset)
print("Ising Hamiltonian:")
print(str(qubit_op))

# %%
print(mdl.prettyprint())
# %%

reps = 1

n_shots = 1024
aer_sim, pm = get_sim_and_pm(SEED)
sampler = Sampler(default_shots=n_shots, seed=SEED)
sampler._backend = aer_sim

circuit = get_qaoa_circuit(qubit_op, reps=reps, measure=True)
isa_qc, isa_ham = transpile(pm, circuit, qubit_op)

alpha = 0.3
estimator = DEstimator(sampler=sampler, aggregation=alpha)

init_rng = np.random.default_rng(SEED)
init_beta = init_rng.uniform(-0.1, 0.1, reps)
init_gamma = init_rng.uniform(0.0, 0.1, reps)
initial_gamma_beta = np.concatenate((init_beta, init_gamma))
print("Initial gamma and beta values:", initial_gamma_beta)

isa_qc.draw('mpl', fold=-1)
# %%

objective_func_vals = []  # Global variable


def cost_func_estimator(params, ansatz, hamiltonian, estimator):
    pub = (ansatz, hamiltonian, params)
    job = estimator.run([pub])
    results = job.result()[0]
    cost = results.data.evs
    objective_func_vals.append(cost)
    return cost


start_time = time.time()
result = minimize(
    cost_func_estimator,
    initial_gamma_beta,
    args=(isa_qc, isa_ham, estimator),
    method="COBYLA",
    options={
        "maxiter": 100,
        "rhobeg": 1.0
    },
    tol=1e-2,
)
end_time = time.time()
print("Optimization time:", end_time - start_time, "seconds")
# %%
plt.figure(figsize=(12, 6))
plt.plot(objective_func_vals)
plt.xlabel("Iteration")
plt.ylabel("Cost")
plt.show()

# %%
n_sampling_shots = 100
solutions, sequences, energies, probabilities = circuit_sampling(
    isa_qc, result.x, sampler, DURATIONS, n, m, n_sampling_shots)

# %%
for seq, energy, prob in zip(sequences, energies, probabilities):
    print(f"Sequence: {seq+1}, Makespan: {energy}, Probability: {prob:.4f}")

# %%
# Plot the Gantt chart for the best solution

job_shop_model = JobShopModel()
jobs = [job_shop_model.add_job(name=JOB_NAMES[i]) for i in range(num_jobs)]
machines = [
    job_shop_model.add_machine(name=MACHINE_NAMES[i])
    for i in range(num_machines)
]
tasks = {}  # store for later
for job_idx, job in enumerate(jobs):
    for machine_idx, machine in enumerate(machines):
        task = job_shop_model.add_task(job=job)
        tasks[job_idx, machine_idx] = task
        duration = DURATIONS[job_idx][machine_idx]
        job_shop_model.add_mode(task, machine, duration=duration)
for job_idx in range(num_jobs):
    for idx1, idx2 in pairwise(range(num_machines)):
        task1 = tasks[job_idx, idx1]
        task2 = tasks[job_idx, idx2]
        job_shop_model.add_end_before_start(task1, task2)
for machine1, machine2 in pairwise(machines):
    job_shop_model.add_same_sequence(machine1, machine2)

solution = solution_from_sequence(job_shop_model, DURATIONS, sequences[0])

fig, ax = plt.subplots(figsize=(12, 6))
custom_plot_machine_gantt(solution,
                          job_shop_model.data(),
                          plot_labels=True,
                          ax=ax)

# Control font sizes
ax.set_title("Machine Gantt Chart", fontsize=20)
ax.set_xlabel("Time", fontsize=20)

# Control tick-label font sizes
ax.tick_params(axis="x", labelsize=20)
ax.tick_params(axis="y", labelsize=20)

plt.tight_layout()
plt.show()
# %%

# %%
