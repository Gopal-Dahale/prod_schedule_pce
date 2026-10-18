# Production Scheduling Optimization (ERP) using PCE

This project studies flow-shop production scheduling using classical optimization, QAOA, and Pauli Correlation Encoding (PCE) with PAGD decoding.

## Setup

Install [`uv`](https://docs.astral.sh/uv/getting-started/installation/), then clone the repository and enter its directory. Create a virtual environment and install dependencies:

```bash
uv venv
```

Activate it:

```bash
# macOS / Linux
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1
```

Install the requirements:

```bash
uv pip install -r requirements.txt
```

## Run

Select the instance inside the relevant Python file by commenting/uncommenting the instance definitions near the beginning, then run:

```bash
python 1_cp_sat.py
python 2_qaoa.py
python 3_pce.py
```

- **`1_cp_sat.py`** — Classical reference solutions using PyJobShop / OR-Tools, including the baselines, an OR-Tools flow-shop instance (11 jobs × 5 machines), and Taillard instances (20 jobs × 5 machines).
- **`2_qaoa.py`** — Position-based QUBO formulation solved with QAOA using a Conditional Value at Risk (CVaR) objective.
- **`3_pce.py`** — PCE with a `efficient_su2` ansatz and PAGD decoding, evaluated on the baselines, the 11 × 5 OR-Tools instance, and 20 × 5 Taillard instances.

Each script generates a Gantt chart for its selected instance. Quantum circuit diagrams are saved separately.

`output` holds output for base instances

## AI Usage

AI was primarily used for code generation, grammar correction, and improving sentence structure in the development log and project design documentation. GitHub Copilot was used for all these purposes to improve code quality, clarity, and readability.
