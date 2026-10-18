# %%
import warnings

warnings.filterwarnings('ignore')

from docplex.mp.model import Model
import numpy as np
from qiskit_optimization.translators import from_docplex_mp
from qiskit_optimization.converters import QuadraticProgramToQubo
from qiskit_aer.primitives import EstimatorV2 as Estimator
from utils import (get_qaoa_circuit, transpile, circuit_sampling,
                   get_sim_and_pm, makespan, read_base_instance,
                   read_taillard_instances, load_or_library_instances,
                   solution_from_sequence)
from scipy.optimize import minimize
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import efficient_su2
from itertools import combinations, permutations
from plots import custom_plot_machine_gantt
import matplotlib.pyplot as plt
import networkx as nx
import time
from pyjobshop import Model as JobShopModel
from itertools import pairwise

SEED = 10598
np.random.seed(SEED)
rng = np.random.default_rng(SEED)

from qiskit_algorithms.utils import algorithm_globals

algorithm_globals.random_seed = SEED
# %%
# base 3x3
# DURATIONS = read_base_instance("input/base3_3.json")
# LB = 8

# Flowshop instances from OR-Tools
# instances = load_or_library_instances("input/flowshop1.txt")
# DURATIONS = instances['car1'].T
# LB = 7038

# Taillard instances
times, lbs = read_taillard_instances("input/tai20_5.txt")
DURATIONS = times[0].T
LB = lbs[0]

num_jobs, num_machines = len(DURATIONS), len(DURATIONS[0])
print(f"Problem size: {num_jobs} jobs, {num_machines} machines")
print(f"Lower bound: {LB}")

JOB_NAMES = [f"{i+1}" for i in range(num_jobs)]
MACHINE_NAMES = [f"Machine {i+1}" for i in range(num_machines)]

# %%

p = np.array(DURATIONS)
A = 10
n, m = len(p), len(p[0])
P = A * sum(map(sum, p))
N = n * n

print(f"Problem size: {n} jobs, {m} machines")

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

# Extract the objective as offset + z.T @ Q @ z.
vs = [x[i, j] for i in range(n) for j in range(n)]
vid = {v: i for i, v in enumerate(vs)}
obj = mdl.objective_expr
U = np.zeros((N, N))

for v, c in obj.iter_terms():
    U[vid[v], vid[v]] += c
for v1, v2, c in obj.iter_quad_triplets():
    i, j = vid[v1], vid[v2]
    U[min(i, j), max(i, j)] += c

Q = (U + U.T) / 2
offset = float(obj.get_constant())
assert np.allclose(Q, Q.T)


# %%
def build_pce(num_vars):
    nq = max(2, int(np.ceil((1 + np.sqrt(1 + 8 * num_vars / 3)) / 2)))
    while 3 * (nq * (nq - 1) // 2) < num_vars:
        nq += 1

    observables = []
    for axis in ("X", "Y", "Z"):
        for a, b in combinations(range(nq), 2):
            label = ["I"] * nq
            label[a] = label[b] = axis
            observables.append(
                SparsePauliOp.from_list([("".join(label[::-1]), 1.0)]))
            if len(observables) == num_vars:
                return nq, observables
    raise ValueError("Not enough PCE observables")


aer_sim, pm = get_sim_and_pm(SEED)

nq, observables = build_pce(N)
qc = efficient_su2(nq, su2_gates=["ry", "rz"], reps=1)
ansatz = pm.run(qc)
hamiltonian = [op.apply_layout(ansatz.layout) for op in observables]

params = list(ansatz.parameters)
ALPHA = float(nq)

estimator = Estimator()
estimator._backend = aer_sim

print(f"Number of qubits: {nq}")
print(f"Number of parameters in ansatz: {len(params)}")

# %%


def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -60, 60)))


def get_expectations(theta):
    pubs = [(ansatz, ham, theta) for ham in hamiltonian]
    job = estimator.run(pubs)
    results = job.result()
    evs = np.array([r.data.evs for r in results])
    return evs


def loss_fn(theta):
    e = get_expectations(theta)
    soft = sigmoid(-ALPHA * e)
    loss = float(soft @ Q @ soft)  # paper's QUBO-space loss
    if loss < best_train["loss"]:
        best_train.update(loss=loss, e=e.copy(), theta=theta.copy())

    objective_func_vals.append(loss)
    return loss


# %%
# QUBO Space Training

all_func_vals = []
objective_func_vals = []  # Global variable

TRAIN_RESTARTS, TRAIN_MAXITER = 1, 400
rng = np.random.default_rng(SEED)
best_train = {"loss": np.inf, "e": None, "theta": None}

for r in range(TRAIN_RESTARTS):
    theta0 = rng.uniform(-0.1, 0.1, len(params))

    start_time = time.time()
    result = minimize(loss_fn,
                      theta0,
                      method="COBYLA",
                      options={
                          "maxiter": TRAIN_MAXITER,
                          "rhobeg": 1.0
                      })
    end_time = time.time()
    all_func_vals.append(objective_func_vals.copy())
    objective_func_vals.clear()

    print(f"Training {r+1}: final loss={result.fun:.6f}, "
          f"evaluations={result.nfev}, time={end_time - start_time:.2f}s")

all_func_vals = np.array(all_func_vals)

e = best_train["e"]
print("Best training loss:", best_train["loss"])
print("Qubits:", nq, "| PCE variables:", N)

# %%
# plot the training loss for each restart
plt.figure(figsize=(12, 6))
for r in range(TRAIN_RESTARTS):
    plt.plot(all_func_vals[r], label=f"Restart {r+1}")
plt.title("Training Loss for Each Restart")
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel("Iteration", fontsize=14)
plt.ylabel("Cost", fontsize=14)
plt.grid(True)
plt.legend(fontsize=14)
plt.show()

# %%


# Problem-Aware Guided Decoder
def pagd_permutation(Q,
                     e,
                     n,
                     alpha,
                     beta=1.0,
                     K=100,
                     noise=0.05,
                     seed=123,
                     swap_refine=True):
    rng = np.random.default_rng(seed)
    N = n * n
    best_energy, best_bits = np.inf, None
    diag = np.diag(Q)

    def swap_descent(bits):
        bits = bits.copy()
        while True:
            perm = bits.reshape(n, n).argmax(axis=1)
            current = float(bits @ Q @ bits)
            next_energy, next_bits = current, None

            for i in range(n):
                for r in range(i + 1, n):
                    a, b = i * n + perm[i], r * n + perm[r]
                    c, d = i * n + perm[r], r * n + perm[i]
                    y = bits.copy()
                    y[a] = y[b] = 0
                    y[c] = y[d] = 1
                    energy = float(y @ Q @ y)
                    if energy < next_energy - 1e-10:
                        next_energy, next_bits = energy, y

            if next_bits is None:
                return bits
            bits = next_bits

    for _ in range(K):
        rho = sigmoid(-alpha * (e + rng.normal(0, noise, N)))
        bits = np.zeros(N)
        delta = diag.copy()
        rows, jobs = set(range(n)), set(range(n))

        while rows:
            cand = np.array(
                [i * n + j for i in sorted(rows) for j in sorted(jobs)])
            neg = cand[delta[cand] < -1e-10]

            if len(neg):
                score = -delta[neg] * rho[neg]**beta
                ties = neg[np.isclose(score,
                                      score.max(),
                                      rtol=1e-10,
                                      atol=1e-10)]
            else:
                ties = cand[np.isclose(delta[cand],
                                       delta[cand].min(),
                                       rtol=1e-10,
                                       atol=1e-10)]
                ties = ties[np.isclose(rho[ties],
                                       rho[ties].max(),
                                       rtol=1e-10,
                                       atol=1e-10)]

            a = int(rng.choice(ties))
            i, j = divmod(a, n)
            bits[a] = 1
            rows.remove(i)
            jobs.remove(j)
            delta += 2 * Q[:, a]

        if swap_refine:
            bits = swap_descent(bits)

        seq = seq = [
            next(j for j in range(n) if bits[i * n + j] > 0.5) for i in range(n)
        ]
        energy = makespan(p, n, m, seq)

        if energy < best_energy:
            best_energy, best_bits = energy, bits.copy()

    X = best_bits.reshape(n, n).astype(int)
    return best_energy, X.argmax(axis=1) + 1, X


Cmax, sequence, X = pagd_permutation(Q,
                                     e,
                                     n,
                                     ALPHA,
                                     beta=1.,
                                     K=100,
                                     noise=0.2,
                                     seed=SEED + 1,
                                     swap_refine=True)

# Evaluate actual schedule
print("Job sequence (1-based):", sequence.tolist())
print("Actual makespan:", Cmax)

if LB is not None:
    print("Lower bound:", LB)
    print("Gap to lower bound: {:.2f}%".format((Cmax - LB) / LB * 100))

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

solution = solution_from_sequence(job_shop_model, DURATIONS, sequence - 1)

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
'''
Problem size: 20 jobs, 5 machines

Number of qubits: 17
Number of parameters in ansatz: 102

Training 1: final loss=1077521.202364, evaluations=400, time=1182.55s
Best training loss: 1077521.202364472
Qubits: 17 | PCE variables: 400

Job sequence (1-based): [18, 6, 19, 4, 9, 17, 1, 12, 11, 2, 13, 5, 14, 16, 8, 15, 10, 7, 20, 3]
Actual makespan: 1418
Lower bound: 1232
Gap to lower bound: 15.10%
'''
'''
Problem size: 11 jobs, 5 machines

Number of qubits: 10
Number of parameters in ansatz: 60

Training 1: final loss=8995275.838111, evaluations=100, time=26.06s
Best training loss: 8995275.83811076
Qubits: 10 | PCE variables: 121

Job sequence (1-based): [10, 8, 4, 11, 9, 7, 2, 3, 6, 5, 1]
Actual makespan: 8847
Lower bound: 7038
Gap to lower bound: 25.70%
'''
'''
Problem size: 3 jobs, 3 machines

Number of qubits: 3
Number of parameters in ansatz: 18

Training 1: final loss=36.169483, evaluations=400, time=6.52s
Best training loss: 36.1694827993114
Qubits: 3 | PCE variables: 9

Job sequence (1-based): [2, 1, 3]
Actual makespan: 8
Lower bound: 8
Gap to lower bound: 0.00%
'''
# %%

# %%
