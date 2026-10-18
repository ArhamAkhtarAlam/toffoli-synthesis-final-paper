# From Truth Table to Quantum Circuit: K-Map-Guided Toffoli Synthesis

ICEE 2026 paper (#1571351401) and the reference code behind its numbers.

- `paper/toffoli_synthesis.tex` - IEEEtran conference source (`pdflatex` twice; needs `IEEEtran.cls`, `quantikz`)
- `paper/toffoli_synthesis.pdf` - compiled PDF (4 pages, ICEE proceedings limit)
- `paper/toffoli_synthesis_extended.tex` - longer 7-page version with full proofs and discussion
- `code/disjoint_synth.py` - DisjointCover / GrowCube, benchmark truth tables
- `code/gatelevel.py` - gate emission and the X-X cancellation pass, classical-replay verifier
- `code/synth_full.py` - full pipeline: cover + sharing + derivation (chosen after cancellation) + cancellation
- `code/ordering.py` - term-ordering heuristic and exact ordering optimum
- `code/recompute.py` - regenerates every table in the paper except the tweedledum columns
- `code/table.py` - the main results table (Table I without the CZ columns), every circuit verified
- `code/draw.py` - saves a picture of every circuit (ours at each stage, tweedledum's) to `code/circuits/`
- `code/randcheck.py` - randomized validation behind every "over N random trials" figure
- `code/baselines.py` - runs tweedledum on every benchmark (tweedledum columns); writes `code/results/tweedledum_circuits.json`
- `code/device.py` - transpiles our and tweedledum's circuits to IBM's FakeTorino device model and checks each by statevector simulation (CZ columns); output in `code/results/device_torino.txt`

The hybrid (overlapping covers, selection by CZ count):

- `code/improve.py` - the hybrid's candidates: other disjoint covers (other seeds and variable orders) and overlapping XOR (ESOP) covers from every fixed-polarity Reed-Muller form, each run through the same pipeline and checked on all inputs; the disjoint pipeline's circuit is kept unless a candidate has no more gates and no higher quantum cost. Output in `code/results/improve.txt`
- `code/improve_device.py` - transpiles those candidates like `device.py` and keeps the one with the fewest CZ gates (the disjoint pipeline's circuit is a candidate, so never worse on the device); the Hyb. columns of Table I. Output in `code/results/device_torino_improved.txt`

Run `cd code && python3 recompute.py && python3 randcheck.py`.

`table.py`, `draw.py`, `randcheck.py` and `device.py` run in parallel worker processes, one per CPU core by default; pass a number to use fewer (e.g. `python3 randcheck.py 4`). Results do not depend on the number of workers. `draw.py` needs matplotlib.

`baselines.py` needs tweedledum 1.1.1, which only has wheels for CPython 3.6-3.10:

    uv venv -p 3.10 .venv310
    uv pip install --python .venv310/bin/python tweedledum==1.1.1
    cd code && ../.venv310/bin/python baselines.py

`device.py` needs `pip install qiskit qiskit-ibm-runtime` (tested with qiskit 2.6.0, qiskit-ibm-runtime 0.50.0) and checks every transpiled circuit by simulation on all inputs.
