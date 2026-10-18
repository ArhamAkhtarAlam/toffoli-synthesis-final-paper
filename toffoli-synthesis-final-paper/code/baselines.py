"""Run tweedledum's synthesizers on the paper's benchmark functions and verify every
circuit with this repository's classical replay.

Needs tweedledum 1.1.1, which ships wheels only for CPython 3.6-3.10, e.g.
    uv venv -p 3.10 .venv310 && uv pip install --python .venv310/bin/python tweedledum==1.1.1
    .venv310/bin/python baselines.py

Out-of-place: pkrm_synth and pprm_synth (ESOP-based) per output, on n+m lines,
directly comparable with our construction. In-place (bijections only):
transform_synth (transformation-based, Miller-Maslov-Dueck) on n lines.
(decomp_synth is not used: it emits single-target truth-table gates, not MCT.)
tweedledum emits mixed-polarity MCT gates; we report that native count and
the positive-control MCT count after rewriting each negative control as an
X-sandwich and applying our X-cancellation pass (the library our tables use)."""
import io, contextlib
from itertools import product
from tweedledum.classical import TruthTable, create_from_binary_string
from tweedledum.synthesis import pkrm_synth, pprm_synth, transform_synth
from gatelevel import simulate, cancel_xx, qcost
from synth_full import verify
from ordering import ordered

def tt_of(on_set, n):
    """tweedledum string: rightmost char is row 0; bit k of the row index is qubit k,
    our wire i (MSB-first tuple) is bit n-1-i."""
    rows = ['1' if tuple((r >> (n - 1 - i)) & 1 for i in range(n)) in on_set else '0' for r in range(2 ** n)]
    tt = TruthTable(n); create_from_binary_string(tt, ''.join(reversed(rows)))
    return tt

def convert(circ, wire):
    """tweedledum circuit -> (native mixed-polarity gate count, positive-control gate list)."""
    native, gates = 0, []
    for ins in circ:
        assert ins.kind() == 'std.x', ins.kind()
        qs = ins.qubits(); native += 1
        ctrls = [(wire[q.uid()], q.polarity() == q.Polarity.positive) for q in qs[:-1]]
        neg = [w for w, pos in ctrls if not pos]
        gates += [('X', w) for w in neg]
        gates.append(('X', wire[qs[-1].uid()]) if not ctrls else ('MCX', tuple(w for w, _ in ctrls), wire[qs[-1].uid()]))
        gates += [('X', w) for w in neg]
    return native, gates

EXPORT = {}  # (function, method) -> positive-control gate list after X-cancellation, for device.py

def out_of_place(synth, outs, n, key=None):
    m, native, gates = len(outs), 0, []
    for j, s in enumerate(outs):
        wire = {k: n - 1 - k for k in range(n)}; wire[n] = n + j
        nat, g = convert(synth(tt_of(s, n)), wire)
        native += nat; gates += g
    c = cancel_xx(gates)
    assert verify(gates, n, outs) and verify(c, n, outs), "VERIFY FAILED"
    if key: EXPORT[key] = c
    return native, len(c), sum(map(qcost, c))

def permutation(outs, n):
    """outs as on-sets of a bijection -> list perm[r] in tweedledum's bit order."""
    perm = []
    for r in range(2 ** n):
        x = tuple((r >> (n - 1 - i)) & 1 for i in range(n))
        y = [1 if x in s else 0 for s in outs]
        perm.append(sum(b << (n - 1 - i) for i, b in enumerate(y)))
    return perm if sorted(perm) == list(range(2 ** n)) else None

def in_place(synth, outs, n):
    native, gates = convert(synth(permutation(outs, n)), {k: n - 1 - k for k in range(n)})
    c = cancel_xx(gates)
    for g in (gates, c):
        for x in product([0, 1], repeat=n):
            assert tuple(simulate(g, list(x))) == tuple(1 if x in s else 0 for s in outs), "VERIFY FAILED"
    return native, len(c), sum(map(qcost, c))

def functions():
    with contextlib.redirect_stdout(io.StringIO()):
        import recompute
    return recompute.functions()

if __name__ == '__main__':
    from importlib.metadata import version
    print("tweedledum", version('tweedledum'))
    print(f"{'function':14s} {'method':10s} lines native posMCT+X Qcost")
    for name, outs, n in functions():
        m = len(outs)
        g, nanc = ordered(outs, n)
        c = cancel_xx(g)
        assert verify(c, n, outs, nanc)
        print(f"{name:14s} {'this work':10s} {n+m+nanc:5d} {'-':>6s} {len(c):8d} {sum(map(qcost, c)):5d}")
        for label, f in (('pkrm', pkrm_synth), ('pprm', pprm_synth)):
            print(f"{name:14s} {label:10s} {n+m:5d} {'%6d %8d %5d' % out_of_place(f, outs, n, (name, label))}")
        if permutation(outs, n):
            for label, f in (('transform', transform_synth),):
                print(f"{name:14s} {label:10s} {n:5d} {'%6d %8d %5d' % in_place(f, outs, n)}")
    import json, os
    os.makedirs('results', exist_ok=True)
    with open('results/tweedledum_circuits.json', 'w') as fh:
        json.dump([{'function': k[0], 'method': k[1], 'gates': v} for k, v in EXPORT.items()], fh)
