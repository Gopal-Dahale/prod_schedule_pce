# %%
from pyjobshop import Model
import numpy as np
from itertools import pairwise
from pyjobshop.plot import plot_machine_gantt
import matplotlib.pyplot as plt
from plots import custom_plot_machine_gantt
from utils import (load_or_library_instances, read_base_instance)

# %%

# 1. There is a set of n jobs that need to be processed on m machines.
# 2. All jobs follow the same routing: they are processed first on machine 1, then on machine 2, and so on until machine m.
# 3. All jobs must be processed in the same sequence on all machines (permutation constraint).
# 4. Each job has a processing time on each machine.
# 5. The objective is typically to minimize the makespan.

# Reference: Lan, L., and Berkhout, J. (2025). PyJobShop: Solving scheduling problems with constraint programming in Python. https://arxiv.org/abs/2502.13483

# %%

# base 2x2
DURATIONS = read_base_instance("input/base2_2.json")

# base 3x3
# DURATIONS = read_base_instance("input/base3_3.json")

# Flowshop instances from OR-Tools
# instances = load_or_library_instances("input/flowshop1.txt")
# DURATIONS = instances['car1'].T

# Taillard instances
times, lbs = read_taillard_instances("input/tai20_5.txt")
DURATIONS = times[0].T
LB = lbs[0]

num_jobs, num_machines = len(DURATIONS), len(DURATIONS[0])
print(f"Problem size: {num_jobs} jobs, {num_machines} machines")

JOB_NAMES = [f"{i+1}" for i in range(num_jobs)]
MACHINE_NAMES = [f"Machine {i+1}" for i in range(num_machines)]

# %%

# We start by adding all jobs and machines

model = Model()
jobs = [model.add_job(name=JOB_NAMES[i]) for i in range(num_jobs)]
machines = [
    model.add_machine(name=MACHINE_NAMES[i]) for i in range(num_machines)
]

# For each job and machine, we create one task and its corresponding processing mode.

tasks = {}  # store for later

for job_idx, job in enumerate(jobs):
    for machine_idx, machine in enumerate(machines):
        task = model.add_task(job=job)
        tasks[job_idx, machine_idx] = task
        duration = DURATIONS[job_idx][machine_idx]
        model.add_mode(task, machine, duration=duration)

# We have to make sure that a job actually “flows” through the machine environment. This we can enforce by setting precedence constraints

for job_idx in range(num_jobs):
    for idx1, idx2 in pairwise(range(num_machines)):
        task1 = tasks[job_idx, idx1]
        task2 = tasks[job_idx, idx2]
        model.add_end_before_start(task1, task2)

# Finally, we impose the same sequence constraint, which ensures that all machines process all tasks in the same order

for machine1, machine2 in pairwise(machines):
    model.add_same_sequence(machine1, machine2)
# %%
result = model.solve(display=False)
print(result)
# %%
fig, ax = plt.subplots(figsize=(12, 6))
custom_plot_machine_gantt(result.best, model.data(), plot_labels=True, ax=ax)

# Control font sizes
ax.set_title("Machine Gantt Chart", fontsize=20)
ax.set_xlabel("Time", fontsize=20)

# Control tick-label font sizes
ax.tick_params(axis="x", labelsize=20)
ax.tick_params(axis="y", labelsize=20)

plt.tight_layout()
plt.show()

# %%
