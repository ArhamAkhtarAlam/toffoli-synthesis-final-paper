"""Transpile this work's circuits and tweedledum's to an IBM device model and report
native two-qubit gate count and depth; every transpiled circuit is checked by
statevector simulation, including layout and routing.

Needs qiskit and qiskit-ibm-runtime (for the offline FakeTorino device snapshot), and
results/tweedledum_circuits.json written by baselines.py.
Run: python3 device.py [workers]   (default: all cores; results do not depend on it)"""
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

def simulate_batch(rc, columns):
    """Apply circuit rc to several basis states at once: returns a (2**q, K) array whose column k
    is rc applied to basis state columns[k] (Qiskit's little-endian qubit order)."""
    q, K = rc.num_qubits, len(columns)
    st = np.zeros((2 ** q, K), dtype=complex)
    st[columns, np.arange(K)] = 1
    st = st.reshape((2,) * q + (K,))
    cache = {}
    for ins in rc.data:
        op = ins.operation
        key = (op.name, tuple(float(x) for x in op.params))
        if key not in cache:
            k = op.num_qubits
            cache[key] = np.asarray(op.to_matrix()).reshape((2,) * (2 * k))
        M = cache[key]
        qs = [rc.find_bit(x).index for x in ins.qubits]
        axes = [q - 1 - j for j in reversed(qs)]  # matrix rows/cols are MSB-first over qs
        k = len(qs)
        st = np.moveaxis(np.tensordot(M, st, axes=(list(range(k, 2 * k)), axes)), list(range(k)), axes)
    return st.reshape(2 ** q, K)

def check(tqc, outs, n, width):
    """Every simulated input must map, with probability > 0.999, to the basis state
    holding the inputs, the outputs, and every other active qubit back at 0."""
    rc, init, final = reduced(tqc, width)
    xs = list(np.ndindex(*([2] * n)))
    exhaustive = rc.num_qubits <= MAX_EXHAUSTIVE
    if not exhaustive:
        rng = random.Random(1)
        xs = [xs[0], xs[-1]] + rng.sample(xs[1:-1], min(SAMPLES, len(xs) - 2))
    cols, want = [], []
    for x in xs:
        bits = list(x) + [0] * (width - n)
        cols.append(sum(b << init[i] for i, b in enumerate(bits)))
        w = list(x) + [1 if tuple(x) in s else 0 for s in outs] + [0] * (width - n - len(outs))
        want.append(sum(b << final[i] for i, b in enumerate(w)))
    out = simulate_batch(rc, cols)
    for k, x in enumerate(xs):
        if abs(out[want[k], k]) ** 2 < 0.999:
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

def load_tweedledum(path='results/tweedledum_circuits.json'):
    return {(r['function'], r['method']): [(g[0], tuple(g[1]), g[2]) if g[0] == 'MCX' else tuple(g) for g in r['gates']]
            for r in json.load(open(path))}

def job(args):
    """Transpile and check one circuit; runs in a worker process."""
    name, label = args
    with contextlib.redirect_stdout(io.StringIO()):
        import recompute
    outs, n = {f: (o, k) for f, o, k in recompute.functions()}[name]
    m = len(outs)
    if label == 'this work':
        g, nanc = ordered(outs, n)
        gates, width = cancel_xx(g), n + m + nanc
    else:
        gates, width = load_tweedledum()[(name, label)], n + m
    twoq, depth, nq, ninputs, ex = device_stats(gates, outs, n, width, FakeTorino())
    return len(gates), sum(map(qcost, gates)), twoq, depth, nq, ninputs, ex

def main(workers=None):
    with contextlib.redirect_stdout(io.StringIO()):
        import recompute
    names = [f for f, _, _ in recompute.functions()]
    labels = ('this work', 'pkrm', 'pprm')
    jobs = [(f, l) for f in names for l in labels]
    backend = FakeTorino()
    print(f"backend {backend.name}, {backend.num_qubits} qubits, basis {sorted(backend.operation_names)}, "
          f"optimization_level=3, seed_transpiler={SEED}, qubits_initially_zero=False")
    if workers == 1:
        out = list(map(job, jobs))
    else:
        from concurrent.futures import ProcessPoolExecutor
        from multiprocessing import get_context
        with ProcessPoolExecutor(workers, mp_context=get_context('spawn')) as ex:
            out = list(ex.map(job, jobs))
    res = dict(zip(jobs, out))
    print(f"{'function':14s} {'method':9s} {'gates':>5s} {'Q':>5s} {'2q':>6s} {'depth':>6s}  check")
    sign = lambda a, b: (a > b) - (a < b)
    pairs = agree_q = agree_g = 0
    for name in names:
        for label in labels:
            g, q, twoq, depth, nq, ninputs, ex = res[(name, label)]
            print(f"{name:14s} {label:9s} {g:5d} {q:5d} {twoq:6d} {depth:6d}  "
                  f"{ninputs} inputs {'(all)' if ex else '(sampled)'} on {nq} active qubits")
        ours = res[(name, 'this work')]
        for k in ('pkrm', 'pprm'):
            d = sign(ours[2], res[(name, k)][2])
            pairs += 1; agree_g += d == sign(ours[0], res[(name, k)][0]); agree_q += d == sign(ours[1], res[(name, k)][1])
    print(f"this work vs each tweedledum circuit, {pairs} pairs: which needs fewer 2q gates on the device "
          f"matches which has lower Q-cost in {agree_q}, fewer MCT gates in {agree_g}")

if __name__ == '__main__':
    import os, sys
    main(int(sys.argv[1]) if len(sys.argv) > 1 else os.cpu_count())
