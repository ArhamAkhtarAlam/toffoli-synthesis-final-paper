"""Randomized validation behind every 'over N random trials' statement in the paper."""
import random
from itertools import product
from synth_full import run, synthesize, cancelled_len, verify
from ordering import ordered, exact
from disjoint_synth import random_validation

def pct(a, b): return 100 * a / b if b else 0.0
def mean(xs): return 100 * sum(xs) / len(xs) if xs else 0.0

t, v = random_validation(trials=2000)
print(f"DisjointCover exactness/disjointness: {t} on-sets (n<=6), {v} violations")

rng = random.Random(7)
vr = random.Random(11)  # separate stream, so the trial set matches seed 7 alone
T = 2000; cases = 0; multi = 0
red_x, changed_x = [], 0
share_orig = share_aware = 0
deriv_trig = 0; deriv_red = []
orig_worse = aware_worse = final_worse_cover = final_worse_orig = 0
ord_red = []; ord_opt = ord_exact = 0
verify_ok = verify_reject = 0
for _ in range(T):
    n = rng.randint(2, 4); m = rng.randint(1, 3)
    uni = list(product([0, 1], repeat=n))
    outs = [set(rng.sample(uni, rng.randint(0, len(uni)))) for _ in range(m)]
    c0, c1, _ = run(outs, n, share=False, derive=False)        # run() asserts exhaustive replay
    o0, o1, oa = run(outs, n, aware=False)
    a0, a1, aa = run(outs, n, fallback=False)
    f0, f1, fa = run(outs, n)
    g, na = ordered(outs, n)
    # the verifier must reject the same circuit against a truth table with one bit flipped
    x = vr.choice(uni); j = vr.randrange(m)
    bad = [set(s) for s in outs]; bad[j] ^= {x}
    gf, nf = synthesize(outs, n)
    verify_ok += verify(g, n, outs, na) and verify(gf, n, outs, nf)
    verify_reject += not verify(gf, n, bad, nf)
    if c0[0] == 0: continue
    cases += 1
    red_x.append(1 - c1[0] / c0[0]); changed_x += c1[0] < c0[0]
    share_orig += oa > 0; share_aware += aa > 0
    if m >= 2:
        multi += 1
        do0, _, _ = run(outs, n, share=False, derive=True, aware=False)
        if do0[0] < c0[0]: deriv_trig += 1; deriv_red.append(1 - do0[0] / c0[0])
    orig_worse += o1[0] > c1[0]; aware_worse += a1[0] > c1[0]
    final_worse_cover += f1[0] > c1[0]; final_worse_orig += f1[0] > o1[0]
    lo = cancelled_len(g); ord_red.append(1 - lo / f1[0] if f1[0] else 0)
    blocks = [synthesize(outs, n, blocks=True, **kw)[0] for kw in (dict(fallback=False), dict(share=False, derive=False), dict(aware=False))]
    e = [exact(b, n) for b in blocks]
    if None not in e:
        ord_exact += 1; ord_opt += lo == min(e)

print(f"random truth tables: {T} (2<=n<=4, 1<=m<=3, seed 7), {cases} with a non-empty circuit; every circuit replay-verified")
print(f"verifier: accepted all {verify_ok}/{T} synthesized circuits, rejected {verify_reject}/{T} against a one-bit-flipped table")
print(f"X-cancel on cover circuits: changed {pct(changed_x, cases):.1f}% of circuits, mean reduction {mean(red_x):.1f}%")
print(f"sharing used: original rule {pct(share_orig, cases):.1f}%, cancellation-aware rule {pct(share_aware, cases):.1f}%")
print(f"derivation (original rule, before X-cancel), {multi} tables with m>=2: chosen in {pct(deriv_trig, multi):.1f}%, "
      f"mean reduction when chosen {mean(deriv_red):.1f}%")
print(f"worse than cover+X: original pipeline {pct(orig_worse, cases):.1f}%, cancellation-aware without fallback "
      f"{pct(aware_worse, cases):.1f}%, final pipeline {final_worse_cover} cases; final worse than original: {final_worse_orig} cases")
print(f"term ordering on top of full+X: mean reduction {mean(ord_red):.1f}%; reached the exact best order in "
      f"{ord_opt}/{ord_exact} ({pct(ord_opt, ord_exact):.1f}%) of cases where the exact optimum was computed")
