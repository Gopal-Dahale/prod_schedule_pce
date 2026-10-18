# Development Log

**Last updated:** 18 October 2026  
**Status:** Baselines and simulator experiments in progress

## Project objective and current structure

The project studies flow-shop scheduling, where every job is processed on the machines in the same order. The primary objective is to minimize the makespan. Results from quantum and classical methods are checked against the actual schedule, not only against the energy of the QUBO used to generate it.

The code is organized into three primary scripts:

- `1_cp_sat.py` uses PyJobShop / OR-Tools to solve baseline instances and benchmark flow shops, including an OR-Tools instance with 11 jobs and 5 machines and Taillard instances with 20 jobs and 5 machines.
- `2_qaoa.py` builds the position-based QUBO and applies QAOA with a Conditional Value at Risk (CVaR) objective.
- `3_pce.py` applies Pauli Correlation Encoding (PCE), the `efficient_su2` ansatz, QUBO-space training loss, and a permutation-aware Problem-Aware Guided Decoding (PAGD) post-processor. It covers the baseline and the larger reported flow-shop instances.

Each script produces a Gantt chart. Quantum circuit diagrams have also been saved. The experiments described here use simulators; no hardware result or quantum-advantage claim is made.

## 16 October 2026
### Kick-off and initial literature review

After attending the introductory meeting, we reviewed the project slides and the provided project brief. We then began an online literature review to understand QUBO formulations for flow-shop scheduling and qubit-efficient variational methods. The main references reviewed were:

- Farhani, Y., Saiem, M., Arbaoui, T., and Hnaien, F., *Enhanced QUBO Formulations for The Flow Shop Scheduling Problem* (GECCO '24 Companion, 2024, pp. 1942–1949). This paper compares a position-based QUBO with TSP-based approximations for permutation flow-shop scheduling. It motivated our initial study of a position-based job-permutation encoding. [1]
- Sciorilli, M., Borges, L., Patti, T. L., García-Martín, D., Camilo, G., Anandkumar, A., and Aolita, L., *Towards large-scale quantum optimization solvers with few qubits* (Nature Communications 16, 476, 2025). This paper introduces Pauli Correlation Encoding (PCE), which represents many binary optimization variables as Pauli correlators on a smaller qubit register. It motivated testing PCE for the larger scheduling instances. [2]
- Friedhoff, T., Metkar, M., Davis, W., Kumar, V., and Galda, A., *Pauli Correlation Encoding for mRNA Secondary Structure Prediction: Problem-Aware Decoding for Dense-Constraint QUBOs* (arXiv:2605.20163, 2026). This work proposes the Problem-Aware Guided Decoder (PAGD) and a QUBO-space sigmoid training loss. We considered these ideas for converting PCE expectation values into feasible job permutations. Its mRNA constraints differ from assignment constraints, so its decoder was modified by us for our problem. [3]
- IBM Quantum, *Pauli correlation encoding to reduce max-cut requirements* (Qiskit tutorial). We also used this implementation-oriented tutorial as a reference for PCE observables and simulator evaluation. [4]

The initial design direction was to establish classical reference solutions first, implement a compact position-based QUBO for the mandatory instances, and then compare QAOA and PCE-based approaches on a simulator. At this stage, no quantum-hardware performance or quantum-advantage claim was made.

## 17 October 2026
### Position-based QUBO formulation reviewed

The first QUBO under study uses binary assignment variables

$$
 x_{i,j}=1 \quad\text{if job }j\text{ occupies sequence position }i.
$$

The encoding uses $n^2$\)$ binary variables for $n$ jobs. A valid permutation is enforced by one-hot row and column constraints:

$$
\sum_{j=1}^{n}x_{i,j}=1\quad\forall i,\qquad
\sum_{i=1}^{n}x_{i,j}=1\quad\forall j.
$$

Each equality is converted into a squared penalty. The combined objective used in the prototype is

$$
E(x)=P E_{\mathrm{constraints}}(x)+E_{\mathrm{cost}}(x),
\qquad P=A\sum_{j=1}^{n}\sum_{k=1}^{m}p_{j,k}.
$$

The authors approximate the makespan objective with

$$
E_{\mathrm{cost}} = \sum_{k=1}^{m-1}\sum_{i=1}^{n-1} \left( \sum_{j=1}^{n} p_{j,k}x_{i+1,j} - \sum_{j=1}^{n} p_{j,k+1}x_{i,j}   \right).
$$

The assignment constraints are incorporated using squared penalties:
$$
E_{\mathrm{constraints}} ={} \sum_{i=1}^{n} \left(\sum_{j=1}^{n}x_{i,j}-1\right)^2\\ + \sum_{j=1}^{n} \left(\sum_{i=1}^{n}x_{i,j}-1\right)^2.
$$

The prototype uses $A=10$ for the small example. This was chosen as a scale-based penalty, not established as a universally sufficient value. 

The cost term follows the approximate position-based expression and preserves the authors' compact encoding, but it does not provide an exact makespan objective for arbitrary permutations. That distinction became important during validation below.

### Classical Solution with PyJobShop

We use the PyJobShop python library to quickly solve the base instances and generate gantt chart.

### Makespan evaluation separated from QUBO energy

A separate schedule evaluator was added to the workflow. For a sequence $\pi$, completion times are computed using

$$
C_{i,k}=p_{\pi_i,k}+\max(C_{i-1,k},C_{i,k-1}),
$$

The actual makespan is $C_{n,m}$. This calculation is independent of the approximate QUBO energy and is used to evaluate every decoded sequence.

### QUBO matrix and QAOA

The QUBO coefficients are extracted from the Docplex objective rather than reconstructed independently. This was then fed to QAOA to solve base instances using Conditional Value at Risk (CVaR) objective function.

### Initial QAOA baseline results recorded

The following QAOA results were supplied for the small instances

| Instance | Qubits | Ansatz parameters | Optimization time | Reported sequence | Makespan | Lower bound | Gap to lower bound |
|---|---:|---:|---:|---|---:|---:|---:|
| 2 jobs × 2 machines | 4 | 2 | 0.342 s | `[2, 1]` | 4 | 4 | 0% |
| 3 jobs × 3 machines | 9 | 2 | 0.846 s | `[1, 2, 3]` | 8 | 8 | 0% |

Both reported makespans match their stated lower bounds.

### PCE and PAGD adapted to permutation variables

The PCE approach reuses pairwise Pauli correlations of type $XX$, $YY$, and $ZZ$ to represent logical QUBO variables. The qubit count is chosen to satisfy

$$
3\binom{n_q}{2}\ge n^2.
$$

The variational ansatz is kept as `efficient_su2`. Its parameters are optimized on a simulator. The QUBO-space training loss follows the paper's soft-binary construction:

$$
\tilde{x}_i=\sigma(-\alpha e_i),\qquad
\mathcal L_{\mathrm{QUBO}}=\tilde{x}^{\mathsf T}Q\tilde{x},
$$

where $e_i$ are PCE expectation values. This trains against the penalized QUBO directly. It does not itself guarantee that a final binary assignment is feasible. PAGD was adapted for the permutation structure. For a symmetric QUBO, the marginal change for activating a currently unselected variable is

$$
\Delta_i=Q_{ii}+2\sum_{j:x_j=1}Q_{ij}.
$$

The decoder combines this value with the sigmoid prior, selects only assignments whose row and job column are unused, and removes all other candidates in that row and column. Unlike the independent-set decoder in the paper, it continues until every job and every position has been assigned. Randomized prior perturbations provide multiple decoding restarts. Pair-swap local search is used as an optional feasibility-preserving refinement.

**Reason for this adaptation:** the original PAGD commit-and-prune idea is useful, but the original stopping rule may return a partial independent set. A flow-shop solution must be a complete permutation. The adapted decoder guarantees permutation feasibility, but it remains a heuristic and does not guarantee the globally minimum QUBO energy or makespan.

## 18 October 2026 
### Reported PCE simulator results consolidated

The following results were supplied in project notes.

| Instance | Logical variables $n^2$ | PCE qubits | Ansatz parameters | Evaluations | Training time | Reported sequence | Makespan | Lower bound | Gap to lower bound |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|
| 3 jobs × 3 machines | 9 | 3 | 18 | 400 | 6.52 s | `[2, 1, 3]` | 8 | 8 | 0.00% |
| 11 jobs × 5 machines | 121 | 10 | 60 | 100 | 29.37 s | `[10, 8, 11, 9, 7, 4, 6, 3, 5, 2, 1]` | 8691 | 7038 | 23.49% |
| 20 jobs × 5 machines | 400 | 17 | 102 | 400 | 1182.55 s | `[18, 6, 19, 4, 9, 17, 1, 12, 11, 2, 13, 5, 14, 16, 8, 15, 10, 7, 20, 3]` | 1418 | 1232 | 15.10% |


PCE demonstrates a substantial reduction in physical qubit count relative to the number of encoded binary variables: 9 variables are represented with 3 qubits, 121 with 10 qubits, and 400 with 17 qubits. This reduction comes with the cost of measuring and optimizing many Pauli correlations.

 

## Design decisions and alternatives considered

### Retained for the current baseline

The position-based encoding was retained because it uses $n^2$ logical binary variables and encodes permutation feasibility directly. The same QUBO is used as the starting point for the CPLEX prototype and the PCE training pipeline, which makes it easier to compare their decoded results. QAOA is maintained as a separate solver in `2_qaoa.py`, with CVaR as its objective strategy. PCE is tested with a simulator and a classical PAGD post-processor.

### Not adopted as the current baseline

An exact timing-based encoding was not substituted for the position-based baseline. Time-indexed, binary-time, and other compact encodings remain possible extension directions, but they would change the variable set and require a separate correctness and variable-count analysis. The project brief gives 72 variables for a plain time-indexed 3×3 formulation at horizon $H=8$, and 39 after simple start-window pruning. These are comparison targets, not results produced by the current PCE code.

No hardware execution was used for the results in this log.

## Current limitations

The most important limitation is the mismatch between the approximate QUBO cost and the true makespan. A feasible minimum-energy assignment can still yield a suboptimal schedule. The PCE decoder solves a second problem as well: it must turn continuous expectation values into a complete feasible permutation. Its feasibility guarantee does not imply global optimality.

Variation across random seeds, sensitivity to the penalty multiplier $A$, and the training hyperparameters have not yet been documented. 


## References

[1] Y. Farhani, M. Saiem, T. Arbaoui, and F. Hnaien, “Enhanced QUBO Formulations for The Flow Shop Scheduling Problem,” *Proceedings of the Genetic and Evolutionary Computation Conference Companion (GECCO ’24 Companion)*, 2024, pp. 1942–1949. https://doi.org/10.1145/3638530.3664150

[2] M. Sciorilli, L. Borges, T. L. Patti, D. García-Martín, G. Camilo, A. Anandkumar, and L. Aolita, “Towards large-scale quantum optimization solvers with few qubits,” *Nature Communications*, vol. 16, article 476, 2025. https://doi.org/10.1038/s41467-024-55346-z

[3] T. Friedhoff, M. Metkar, W. Davis, V. Kumar, and A. Galda, “Pauli Correlation Encoding for mRNA Secondary Structure Prediction: Problem-Aware Decoding for Dense-Constraint QUBOs,” arXiv:2605.20163, 2026. https://arxiv.org/abs/2605.20163

[4] IBM Quantum, “Pauli correlation encoding to reduce max-cut requirements,” Qiskit documentation tutorial. https://qiskit.qotlabs.org/docs/tutorials/pauli-correlation-encoding-for-qaoa
