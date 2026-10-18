"""Randomized validation behind every 'over N random trials' statement in the paper.
Run: python3 randcheck.py [workers]   (default: all cores; results do not depend on it)"""
import os, sys, random
from itertools import product
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context  # spawn everywhere, as on macOS and Windows
from synth_full import run, synthesize, cancelled_len, verify
from ordering import ordered, exact
from disjoint_synth import random_validation

T = 2000

def pct(a, b): return 100 * a / b if b else 0.0
def mean(xs): return 100 * sum(xs) / len(xs) if xs else 0.0

def trials():
    """All T truth tables, drawn in the same order as a sequential run."""
    rng = random.Random(7)
    vr = random.Random(11)  # separate stream, so the trial set matches seed 7 alone
    out = []
    for _ in range(T):
        n = rng.randint(2, 4); m = rng.randint(1, 3)
        uni = list(product([0, 1], repeat=n))
        outs = [set(rng.sample(uni, rng.randint(0, len(uni)))) for _ in range(m)]
        out.append((n, outs, vr.choice(uni), vr.randrange(m)))
    return out

def one_trial(args):
    n, outs, x, j = args
    m = len(outs)
    r = {}
    c0, c1, _ = run(outs, n, share=False, derive=False)        # run() asserts exhaustive replay
    o0, o1, oa = run(outs, n, aware=False)
    a0, a1, aa = run(outs, n, fallback=False)
    f0, f1, fa = run(outs, n)
    g, na = ordered(outs, n)
    # the verifier must reject the same circuit against a truth table with one bit flipped
    bad = [set(s) for s in outs]; bad[j] ^= {x}
    gf, nf = synthesize(outs, n)
    r['verify_ok'] = verify(g, n, outs, na) and verify(gf, n, outs, nf)
    r['verify_reject'] = not verify(gf, n, bad, nf)
    r['case'] = c0[0] != 0
    if not r['case']: return r
    r['red_x'] = 1 - c1[0] / c0[0]; r['changed_x'] = c1[0] < c0[0]
    r['share_orig'] = oa > 0; r['share_aware'] = aa > 0
    r['multi'] = m >= 2; r['deriv_red'] = None
    if m >= 2:
        do0, _, _ = run(outs, n, share=False, derive=True, aware=False)
        if do0[0] < c0[0]: r['deriv_red'] = 1 - do0[0] / c0[0]
    r['orig_worse'] = o1[0] > c1[0]; r['aware_worse'] = a1[0] > c1[0]
    r['final_worse_cover'] = f1[0] > c1[0]; r['final_worse_orig'] = f1[0] > o1[0]
    lo = cancelled_len(g); r['ord_red'] = 1 - lo / f1[0] if f1[0] else 0
    blocks = [synthesize(outs, n, blocks=True, **kw)[0] for kw in (dict(fallback=False), dict(share=False, derive=False), dict(aware=False))]
    e = [exact(b, n) for b in blocks]
    r['ord_exact'] = None not in e
    r['ord_opt'] = r['ord_exact'] and lo == min(e)
    return r

def main(workers=None):
    t, v = random_validation(trials=2000)
    print(f"DisjointCover exactness/disjointness: {t} on-sets (n<=6), {v} violations")
    jobs = trials()
    if workers == 1:
        res = list(map(one_trial, jobs))
    else:
        with ProcessPoolExecutor(workers, mp_context=get_context('spawn')) as ex:
            res = list(ex.map(one_trial, jobs, chunksize=20))
    cs = [r for r in res if r['case']]
    count = lambda k, rs=cs: sum(bool(r[k]) for r in rs)
    cases, multi = len(cs), count('multi')
    deriv = [r['deriv_red'] for r in cs if r['deriv_red'] is not None]
    print(f"random truth tables: {T} (2<=n<=4, 1<=m<=3, seed 7), {cases} with a non-empty circuit; every circuit replay-verified")
    print(f"verifier: accepted all {count('verify_ok', res)}/{T} synthesized circuits, rejected {count('verify_reject', res)}/{T} against a one-bit-flipped table")
    print(f"X-cancel on cover circuits: changed {pct(count('changed_x'), cases):.1f}% of circuits, mean reduction {mean([r['red_x'] for r in cs]):.1f}%")
    print(f"sharing used: original rule {pct(count('share_orig'), cases):.1f}%, cancellation-aware rule {pct(count('share_aware'), cases):.1f}%")
    print(f"derivation (original rule, before X-cancel), {multi} tables with m>=2: chosen in {pct(len(deriv), multi):.1f}%, "
          f"mean reduction when chosen {mean(deriv):.1f}%")
    print(f"worse than cover+X: original pipeline {pct(count('orig_worse'), cases):.1f}%, cancellation-aware without fallback "
          f"{pct(count('aware_worse'), cases):.1f}%, final pipeline {count('final_worse_cover')} cases; final worse than original: {count('final_worse_orig')} cases")
    ord_exact, ord_opt = count('ord_exact'), count('ord_opt')
    print(f"term ordering on top of full+X: mean reduction {mean([r['ord_red'] for r in cs]):.1f}%; reached the exact best order in "
          f"{ord_opt}/{ord_exact} ({pct(ord_opt, ord_exact):.1f}%) of cases where the exact optimum was computed")

if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else os.cpu_count())
