"""Transpile this work's circuits and tweedledum's to an IBM device model and report
native two-qubit gate count and depth; every transpiled circuit is checked by
statevector simulation, including layout and routing.

Needs qiskit and qiskit-ibm-runtime (for the offline FakeTorino device snapshot), and
results/tweedledum_circuits.json written by baselines.py. Run: python3 device.py"""
import io, json, contextlib, random
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector
from qiskit_ibm_runtime.fake_provider import FakeTorino
from ordering import ordered
from gatelevel import cancel_xx, qcost

SEED = 0           # seed_transpiler: layout/routing are randomized, so fix it
MAX_EXHAUSTIVE = 16  # simulate every input if the reduced circuit has <= this many qubits
SAMPLES = 32       # otherwise this many random inputs (plus all-zero and all-one)

def to_qiskit(gates, width):
    qc = QuantumCircuit(width)
    for g in gates:
        if g[0] == 'X' or not g[1]:
            qc.x(g[1] if g[0] == 'X' else g[2])
        elif len(g[1]) == 1:
            qc.cx(g[1][0], g[2])
        else:
            qc.mcx(list(g[1]), g[2])
    return qc

def reduced(tqc, width):
    """Drop idle device qubits. Returns (circuit on active qubits, initial and final
    positions of each virtual qubit in it)."""
    init = tqc.layout.initial_index_layout(filter_ancillas=True)[:width]
    final = tqc.layout.final_index_layout(filter_ancillas=True)[:width]
    active = set(init) | set(final)
    for ins in tqc.data:
        active.update(tqc.find_bit(q).index for q in ins.qubits)
    order = sorted(active); pos = {p: i for i, p in enumerate(order)}
    rc = QuantumCircuit(len(order))
    for ins in tqc.data:
        if ins.operation.name in ('barrier', 'delay', 'measure'): continue
        rc.append(ins.operation, [pos[tqc.find_bit(q).index] for q in ins.qubits])
    return rc, [pos[p] for p in init], [pos[p] for p in final]

def check(tqc, outs, n, width):
    """Every simulated input must map, with probability > 0.999, to the basis state
    holding the inputs, the outputs, and every other active qubit back at 0."""
    rc, init, final = reduced(tqc, width)
    xs = list(np.ndindex(*([2] * n)))
    exhaustive = rc.num_qubits <= MAX_EXHAUSTIVE
    if not exhaustive:
        rng = random.Random(1)
        xs = [xs[0], xs[-1]] + rng.sample(xs[1:-1], min(SAMPLES, len(xs) - 2))
    for x in xs:
        bits = list(x) + [0] * (width - n)
        sv = Statevector.from_int(sum(b << init[i] for i, b in enumerate(bits)), 2 ** rc.num_qubits).evolve(rc)
        want = list(x) + [1 if tuple(x) in s else 0 for s in outs] + [0] * (width - n - len(outs))
        idx = sum(b << final[i] for i, b in enumerate(want))
        if abs(sv.data[idx]) ** 2 < 0.999:
            raise AssertionError(f"device circuit wrong on input {x}")
    return rc.num_qubits, len(xs), exhaustive

def device_stats(gates, outs, n, width, backend):
    # qubits_initially_zero=False: the inputs of an oracle or adder are not |0>, so the
    # transpiler must not borrow not-yet-touched input wires as clean scratch qubits
    # (with the default it does, and the ripple adder comes out wrong).
    tqc = transpile(to_qiskit(gates, width), backend=backend, optimization_level=3,
                    seed_transpiler=SEED, qubits_initially_zero=False)
    twoq = sum(1 for ins in tqc.data if ins.operation.num_qubits == 2)
    nq, ninputs, exhaustive = check(tqc, outs, n, width)
    return twoq, tqc.depth(), nq, ninputs, exhaustive

if __name__ == '__main__':
    with contextlib.redirect_stdout(io.StringIO()):
        import recompute
    tw = {(r['function'], r['method']): [(g[0], tuple(g[1]), g[2]) if g[0] == 'MCX' else tuple(g) for g in r['gates']]
          for r in json.load(open('results/tweedledum_circuits.json'))}
    backend = FakeTorino()
    print(f"backend {backend.name}, {backend.num_qubits} qubits, basis {sorted(backend.operation_names)}, "
          f"optimization_level=3, seed_transpiler={SEED}, qubits_initially_zero=False")
    print(f"{'function':14s} {'method':9s} {'gates':>5s} {'Q':>5s} {'2q':>6s} {'depth':>6s}  check")
    sign = lambda a, b: (a > b) - (a < b)
    pairs = agree_q = agree_g = 0
    for name, outs, n in recompute.functions():
        m = len(outs)
        g, nanc = ordered(outs, n)
        rows = [('this work', cancel_xx(g), n + m + nanc)] + [(k, tw[(name, k)], n + m) for k in ('pkrm', 'pprm')]
        res = {}
        for label, gates, width in rows:
            twoq, depth, nq, ninputs, ex = device_stats(gates, outs, n, width, backend)
            res[label] = (len(gates), sum(map(qcost, gates)), twoq)
            print(f"{name:14s} {label:9s} {len(gates):5d} {res[label][1]:5d} {twoq:6d} {depth:6d}  "
                  f"{ninputs} inputs {'(all)' if ex else '(sampled)'} on {nq} active qubits")
        for k in ('pkrm', 'pprm'):
            d = sign(res['this work'][2], res[k][2])
            pairs += 1; agree_g += d == sign(res['this work'][0], res[k][0]); agree_q += d == sign(res['this work'][1], res[k][1])
    print(f"this work vs each tweedledum circuit, {pairs} pairs: which needs fewer 2q gates on the device "
          f"matches which has lower Q-cost in {agree_q}, fewer MCT gates in {agree_g}")
