"""Device-level numbers for improve.py's circuits, with the same transpiler settings and
statevector check as device.py; tweedledum's rows are read from results/device_torino.txt.
Two picks per function: improve()'s (fewest gates without raising quantum cost), and the
candidate with the fewest CZ gates after transpiling, which includes the paper's circuit,
so it is never worse on the device than the paper's.
Run: python3 improve_device.py [workers]   (needs qiskit and qiskit-ibm-runtime)"""
import io, os, re, sys, contextlib
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context  # spawn everywhere, as on macOS and Windows

def job(name):
    from qiskit_ibm_runtime.fake_provider import FakeTorino
    from device import device_stats
    from gatelevel import cancel_xx, qcost
    from improve import improve, candidates
    with contextlib.redirect_stdout(io.StringIO()):
        import recompute
    outs, n = {f: (o, k) for f, o, k in recompute.functions()}[name]
    backend, seen, res = FakeTorino(), {}, {}
    def stats(label, g, nanc):
        gates = cancel_xx(g)
        key = tuple(gates)
        if key not in seen:
            twoq, depth, nq, ninputs, ex = device_stats(gates, outs, n, n + len(outs) + nanc, backend)
            seen[key] = (len(gates), sum(map(qcost, gates)), twoq, depth, ninputs, ex)
        return seen[key] + (label,)
    info = {}
    g, nanc = improve(outs, n, info=info)
    res['improve'] = stats(info['used'], g, nanc)
    allc = [stats(c[0], c[1], c[2]) for c in candidates(outs, n)]
    res['cz'] = min(allc, key=lambda r: (r[2], r[0]))  # paper's circuit is first, so it wins ties
    return res

def paper_rows(path='results/device_torino.txt'):
    rows = {}
    for line in open(path):
        m = re.match(r'(.+?)\s+(this work|pkrm|pprm)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)', line)
        if m: rows[(m[1].strip(), m[2])] = tuple(map(int, m.groups()[2:]))
    return rows

def main(workers=None):
    with contextlib.redirect_stdout(io.StringIO()):
        import recompute
    names = [f for f, _, _ in recompute.functions()]
    with ProcessPoolExecutor(workers, mp_context=get_context('spawn')) as ex:
        res = dict(zip(names, ex.map(job, names)))
    old = paper_rows()
    print("CZ gates on FakeTorino (optimization_level=3, seed 0, qubits_initially_zero=False)")
    print(f"{'function':14s} {'paper':>6s} {'improve':>8s} {'CZ pick':>8s} {'PKRM':>6s} {'PPRM':>6s}  CZ pick: circuit, check")
    for f in names:
        a, b = res[f]['improve'], res[f]['cz']
        assert b[2] <= old[(f, 'this work')][2]
        print(f"{f:14s} {old[(f, 'this work')][2]:6d} {a[2]:8d} {b[2]:8d} {old[(f, 'pkrm')][2]:6d} {old[(f, 'pprm')][2]:6d}  "
              f"{b[6]}: {b[0]} gates, Q {b[1]}, depth {b[3]}; {b[4]} inputs {'(all)' if b[5] else '(sampled)'} simulated")
    return res

if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else os.cpu_count())
