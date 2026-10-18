"""Table I of the paper without the CZ columns (those come from device.py): gate counts at each
stage, exact best term order, tweedledum's gate counts, and quantum cost. Every circuit is checked
against its truth table on all inputs. Needs results/tweedledum_circuits.json (from baselines.py).
Run: python3 table.py [workers]"""
import io, os, sys, json, contextlib
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context  # spawn everywhere, as on macOS and Windows
from synth_full import synthesize, cancelled_len, verify
from gatelevel import cancel_xx, qcost
from ordering import ordered

COLUMNS = ['n', 'm', 'Raw', 'Cover+X', 'Full+X', 'Ordered', 'Opt.', 'PKRM gates', 'PPRM gates',
           'Q ours', 'Q PKRM', 'Q PPRM']

def load_tweedledum(path='results/tweedledum_circuits.json'):
    return {(r['function'], r['method']): [(g[0], tuple(g[1]), g[2]) if g[0] == 'MCX' else tuple(g) for g in r['gates']]
            for r in json.load(open(path))}

def row(name):
    with contextlib.redirect_stdout(io.StringIO()):
        import recompute
    outs, n = {f: (o, k) for f, o, k in recompute.functions()}[name]
    tw = load_tweedledum()
    cover, _ = synthesize(outs, n, share=False, derive=False)
    full, nf = synthesize(outs, n)
    ours, na = ordered(outs, n)
    ours = cancel_xx(ours)
    for g, k in ((cover, 0), (full, nf), (ours, na)):
        assert verify(g, n, outs, k), name
    for k in ('pkrm', 'pprm'):
        assert verify(tw[(name, k)], n, outs), (name, k)
    opt = recompute.best_exact(outs, n)
    return [n, len(outs), len(recompute.raw(outs, n)), cancelled_len(cover), cancelled_len(full), len(ours),
            '-' if opt is None else opt, len(tw[(name, 'pkrm')]), len(tw[(name, 'pprm')]),
            sum(map(qcost, ours)), sum(map(qcost, tw[(name, 'pkrm')])), sum(map(qcost, tw[(name, 'pprm')]))]

def main(workers=None):
    """Returns {function name: row}; rows follow COLUMNS."""
    with contextlib.redirect_stdout(io.StringIO()):
        import recompute
    names = [f for f, _, _ in recompute.functions()]
    with ProcessPoolExecutor(workers, mp_context=get_context('spawn')) as ex:
        return dict(zip(names, ex.map(row, names)))

if __name__ == '__main__':
    rows = main(int(sys.argv[1]) if len(sys.argv) > 1 else os.cpu_count())
    print(f"{'Function':14s}" + ''.join(f"{c:>11s}" for c in COLUMNS))
    for name, r in rows.items():
        print(f"{name:14s}" + ''.join(f"{str(v):>11s}" for v in r))
    print("every circuit verified on all inputs")
