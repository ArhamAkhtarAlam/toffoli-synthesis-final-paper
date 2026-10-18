# From Truth Table to Quantum Circuit: K-Map-Guided Toffoli Synthesis

Reference code for the ICEE 2026 paper of the same title (A. Akhtar). Every number in the paper comes from these scripts.

- `code/disjoint_synth.py` - DisjointCover / GrowCube, benchmark truth tables
- `code/gatelevel.py` - gate emission, the X-X cancellation pass, classical-replay verifier
- `code/synth_full.py` - full pipeline: cover + derivation and sharing (chosen after cancellation) + cancellation
- `code/ordering.py` - term-ordering heuristic and exact ordering optimum (dynamic programming)
- `code/recompute.py` - gate counts, quantum cost, ordering results for all eleven functions
- `code/randcheck.py` - randomized validation over 2000 truth tables (about 4 minutes)
- `code/baselines.py` - runs tweedledum (PKRM, PPRM, transformation-based) on every function; writes `code/results/tweedledum_circuits.json`
- `code/device.py` - transpiles our and tweedledum's circuits to IBM's FakeTorino device model and checks each by statevector simulation; output in `code/results/device_torino.txt`

Every circuit is checked against its truth table on all inputs.

## Running

    cd code
    python3 recompute.py
    python3 randcheck.py

`recompute.py` and `randcheck.py` need only the Python standard library.

`baselines.py` needs tweedledum 1.1.1, which only has wheels for CPython 3.6-3.10:

    uv venv -p 3.10 .venv310
    uv pip install --python .venv310/bin/python tweedledum==1.1.1
    cd code && ../.venv310/bin/python baselines.py

`device.py` needs `pip install qiskit qiskit-ibm-runtime` (tested with qiskit 2.6.0 and qiskit-ibm-runtime 0.50.0) and takes about 8 minutes. It transpiles with `qubits_initially_zero=False`: with Qiskit's default, the transpiler may borrow untouched input wires as scratch qubits, which is wrong for circuits whose inputs are not |0>.
