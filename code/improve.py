"""The hybrid of the paper (Section III, "Overlapping covers"; Section IV, "Selection"), built on the
disjoint pipeline (ordering.ordered), which the paper also reports on its own.

Builds extra candidate circuits and keeps one only if it is no worse than the disjoint pipeline's circuit
(ordering.ordered) in both gate count and quantum cost after X-cancellation, so the result can
never be worse than the pipeline's. Every candidate is replayed on all 2^n inputs before it is
compared. Candidate covers, each fed through the same derivation, sharing, cancellation and
term ordering (each output's cover is picked on its own, so a bigger candidate set can give
a worse whole circuit; each variant is therefore kept as a separate candidate):
  multistart  disjoint covers grown from other seeds and variable orders (Theorems 1-2 still apply)
  esop        overlapping XOR covers: every fixed-polarity Reed-Muller form, plus the disjoint
              covers, each also tried after merging cube pairs that differ in one position
              (correct by replay; the disjoint-cover proof does not cover them)
  esopm       the same, keeping only the merged version of each cover
Run: python3 improve.py [workers]   (results do not depend on the worker count)"""
import os, sys, io, contextlib
from functools import lru_cache
from itertools import permutations, product
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context  # spawn everywhere, as on macOS and Windows
from disjoint_synth import cube_cells
from gatelevel import emit_term, qcost, cancel_xx
from synth_full import cancelled_len, verify
from ordering import ordered

def term_cost(terms, n):
    """(gates, quantum cost) of one output's terms after X-cancellation."""
    return score([x for t in terms for x in emit_term(t, n)])

def grow(seed, remaining, order):
    p = list(seed)
    for i in order:
        trial = p[:]; trial[i] = -1
        if all(c in remaining for c in cube_cells(trial)): p = trial
    return tuple(p)

def disjoint(S, order, pick):
    """Algorithm 2 with a chosen variable order and seed rule; still a disjoint cover."""
    remaining, out = set(S), []
    while remaining:
        p = grow(pick(remaining), remaining, order)
        out.append(p)
        remaining -= set(cube_cells(p))
    return out

def orders(n):
    if n <= 4: return list(permutations(range(n)))
    base = list(range(n))
    return [tuple(base[k:] + base[:k]) for k in range(n)] + [tuple(reversed(base[k:] + base[:k])) for k in range(n)]

def disjoint_covers(S, n):
    return [disjoint(S, o, pick) for o in orders(n) for pick in (min, max)]

def fprm_covers(S, n):
    """Every fixed-polarity Reed-Muller form of S as a cube list (XOR of the cubes = S)."""
    out = []
    for pol in product([0, 1], repeat=n):  # pol[i] = 1: variable i appears complemented
        g = [int(tuple(b ^ q for b, q in zip(x, pol)) in S) for x in product([0, 1], repeat=n)]
        for i in range(n):  # Reed-Muller butterfly, MSB-first index
            step = 1 << (n - 1 - i)
            for k in range(len(g)):
                if k & step: g[k] ^= g[k ^ step]
        terms = []
        for k, c in enumerate(g):
            if c:
                terms.append(tuple((0 if pol[i] else 1) if k >> (n - 1 - i) & 1 else -1 for i in range(n)))
        out.append(terms)
    return out

def merge(terms):
    """c ^ c = 0, and two cubes that differ in one position XOR to one cube
    ({0,1} -> -, {-,1} -> 0, {-,0} -> 1). Each step removes at least one term."""
    third = {frozenset((0, 1)): -1, frozenset((-1, 1)): 0, frozenset((-1, 0)): 1}
    ts = list(terms)
    changed = True
    while changed:
        changed = False
        for a in range(len(ts)):
            for b in range(a + 1, len(ts)):
                d = [i for i in range(len(ts[a])) if ts[a][i] != ts[b][i]]
                if len(d) > 1: continue
                if not d:
                    del ts[b]; del ts[a]
                else:
                    i = d[0]; t = list(ts[a]); t[i] = third[frozenset((ts[a][i], ts[b][i]))]
                    del ts[b]; ts[a] = tuple(t)
                changed = True
                break
            if changed: break
    return ts

def make_cover(kind, key):
    @lru_cache(maxsize=None)
    def best(S, n):
        cands = disjoint_covers(S, n)
        if kind != 'multistart':
            cands += fprm_covers(S, n)
            # merging cuts terms but can add X gates: 'esop' keeps both versions, 'esopm' only merged
            cands = [merge(c) for c in cands] + (cands if kind == 'esop' else [])
        return min(cands, key=lambda t: (term_cost(t, n)[::-1] if key == 'qcost' else term_cost(t, n)))
    return lambda S, n: best(frozenset(S), n)

COVERS = {f'{k}-{key}': make_cover(k, key) for k in ('multistart', 'esop', 'esopm') for key in ('gates', 'qcost')}
GROUP = {'multistart': 'multistart', 'esop': 'esop', 'esopm': 'esop'}

def score(g):
    g = cancel_xx(g)
    return len(g), sum(map(qcost, g))

def candidates(outs, n, kinds=('multistart', 'esop')):
    """[(name, gates, nanc)]: the disjoint pipeline's circuit first, then one per cover variant,
    each checked on all inputs."""
    out = [('disjoint',) + tuple(ordered(outs, n))]
    for name, cover in COVERS.items():
        if GROUP[name.split('-')[0]] not in kinds: continue
        g, na = ordered(outs, n, cover=cover)
        assert verify(g, n, outs, na), name
        out.append((name, g, na))
    return out

def improve(outs, n, kinds=('multistart', 'esop'), info=None):
    """The disjoint pipeline's circuit, or a verified candidate with no more gates and no higher quantum cost
    (fewest gates, then lowest cost, among those). Returns (gates, nanc), uncancelled like ordered()."""
    cands = candidates(outs, n, kinds)
    p = score(cands[0][1])
    ok = [c for c in cands if score(c[1])[0] <= p[0] and score(c[1])[1] <= p[1]]
    name, g, na = min(ok, key=lambda c: score(c[1]))  # min keeps the first on ties: the disjoint pipeline's
    if info is not None: info['used'] = name
    return g, na

def job(args):
    name, outs, n = args
    p = score(ordered(outs, n)[0])
    r = {'name': name, 'disjoint': p}
    for kinds in (('multistart',), ('multistart', 'esop')):
        info = {}
        g, na = improve(outs, n, kinds, info)
        assert verify(g, n, outs, na)
        r[kinds[-1]] = (score(g), info['used'])
    if name is not None:  # ablation: the gate-count ESOP cover with and without derivation
        c = COVERS['esop-gates']
        r['esop_noderiv'] = score(ordered(outs, n, cover=c, share=False, derive=False)[0])
        r['esop_deriv'] = score(ordered(outs, n, cover=c)[0])
    return r

def main(workers=None):
    with contextlib.redirect_stdout(io.StringIO()):
        import recompute
    from randcheck import trials
    try:
        from table import load_tweedledum
        tw = load_tweedledum()
    except Exception:
        tw = {}
    fs = recompute.functions()
    rnd = [(None, outs, n) for n, outs, _, _ in trials()]
    with ProcessPoolExecutor(workers, mp_context=get_context('spawn')) as ex:
        res = list(ex.map(job, [(f, o, n) for f, o, n in fs] + rnd, chunksize=8))
    bench, rand = res[:len(fs)], res[len(fs):]
    print("== Benchmarks: gates / quantum cost after X-cancellation (never worse than the disjoint pipeline's) ==")
    print("(ESOP no der. / + der.: the gate-count ESOP cover alone, without and with derivation)")
    print(f"{'function':15s} {'disjoint':>12s} {'+multistart':>12s} {'+ESOP':>12s}  {'PKRM':>10s} {'PPRM':>10s}  "
          f"{'ESOP no der.':>12s} {'ESOP + der.':>12s}  ESOP pick")
    for r in bench:
        t = lambda k: (lambda g: f"{len(g)}/{sum(map(qcost, g))}")(cancel_xx(tw[(r['name'], k)])) if (r['name'], k) in tw else '-'
        f = lambda s: f"{s[0]}/{s[1]}"
        print(f"{r['name']:15s} {f(r['disjoint']):>12s} {f(r['multistart'][0]):>12s} {f(r['esop'][0]):>12s}  "
              f"{t('pkrm'):>10s} {t('pprm'):>10s}  {f(r['esop_noderiv']):>12s} {f(r['esop_deriv']):>12s}  {r['esop'][1]}")
    live = [r for r in rand if r['disjoint'][0]]
    for k, label in (('multistart', 'multistart only (disjoint, proof intact)'), ('esop', 'multistart + ESOP')):
        better = [r for r in live if r[k][0] < r['disjoint']]
        red = [1 - r[k][0][0] / r['disjoint'][0] for r in live]
        qred = [1 - r[k][0][1] / r['disjoint'][1] for r in live]
        assert all(r[k][0][0] <= r['disjoint'][0] and r[k][0][1] <= r['disjoint'][1] for r in rand)
        print(f"\n== {len(rand)} random truth tables (randcheck.py's set), {label} ==")
        print(f"improved {len(better)} of {len(live)} non-empty tables ({100 * len(better) / len(live):.1f}%); "
              f"mean gate reduction {100 * sum(red) / len(red):.1f}%, mean quantum-cost reduction "
              f"{100 * sum(qred) / len(qred):.1f}%; worse on none (checked)")
        if k == 'esop':
            used = sum(r[k][1].startswith('esop') for r in live)
            print(f"the pick uses an ESOP cover in {used} of {len(live)} ({100 * used / len(live):.1f}%)")
    return bench, rand

if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else os.cpu_count())
