"""Regenerates every table in the paper. Run: python3 recompute.py"""
from itertools import product, permutations
from synth_full import run, synthesize, cancelled_len, verify, cover_cost
from gatelevel import emit_term, cancel_xx
from ordering import heuristic, exact, ordered
from disjoint_synth import ripple_carry_adder_2bit, bcd_to_excess3, four_bit_comparator_eq, three_bit_multiplier

def mk(fn, n, m):
    o = [set() for _ in range(m)]
    for row in product([0, 1], repeat=n):
        r = fn(*row); r = r if isinstance(r, tuple) else (r,)
        for i, v in enumerate(r):
            if v: o[i].add(row)
    return o

def raw(outs, n):
    g = []
    for j, s in enumerate(outs):
        for row in sorted(s): g += emit_term(row, n + j)
    return g

PRESETS = {'AND': (lambda a, b: (a & b,), 2, 1), 'OR': (lambda a, b: (a | b,), 2, 1), 'XOR': (lambda a, b: (a ^ b,), 2, 1),
           'NAND': (lambda a, b: (1 - (a & b),), 2, 1), 'Half adder': (lambda a, b: (a ^ b, a & b), 2, 2),
           'Full adder': (lambda a, b, c: (a ^ b ^ c, (a & b) | (b & c) | (a & c)), 3, 2)}

# 3_17 as a permutation of {0..7}, MSB-first rows and outputs
PLA_3_17 = {(0,0,0):(1,1,1),(0,0,1):(0,0,0),(0,1,0):(0,0,1),(0,1,1):(0,1,1),(1,0,0):(1,0,0),(1,0,1):(0,1,0),(1,1,0):(1,1,0),(1,1,1):(1,0,1)}

def functions():
    """(name, on-sets, n) for every function in the paper. rd32 on RevLib is the full adder."""
    fs = [(k, mk(fn, n, m), n) for k, (fn, n, m) in PRESETS.items()]
    n, o = ripple_carry_adder_2bit(); fs.append(('2b ripple', [o[k] for k in ['S0', 'C0', 'S1', 'Cout']], n))
    n, o = bcd_to_excess3(); fs.append(('BCD-XS3', [o[k] for k in sorted(o)], n))
    n, o = three_bit_multiplier(); fs.append(('3x3 mult', [o[k] for k in sorted(o)], n))
    n, o = four_bit_comparator_eq(); fs.append(('4b comparator', [o['EQ']], n))
    fs.append(('3_17', [{x for x, y in PLA_3_17.items() if y[j]} for j in range(3)], 3))
    return fs

def best_exact(outs, n, max_blocks=16):
    """Exact best block order over the three candidate plans of ordering.ordered; None if any is too large."""
    best = None
    for kw in (dict(fallback=False), dict(share=False, derive=False), dict(aware=False)):
        blocks, _ = synthesize(outs, n, blocks=True, **kw)
        e = exact(blocks, n) if len(blocks) <= max_blocks else None
        if e is None: return None
        best = e if best is None else min(best, e)
    return best

if __name__ == '__main__':
    F = {name: (outs, n) for name, outs, n in functions()}
    print("== Table: gate counts, all functions (before -> after X-cancellation) ==")
    for name, outs, n in functions():
        rg = raw(outs, n)
        (c0, c1, _) = run(outs, n, share=False, derive=False)
        (f0, f1, _) = run(outs, n)
        print(f"{name:14s} n={n} m={len(outs)} | raw {len(rg):4d}->{len(cancel_xx(rg)):4d} | cover {c0[0]:4d}->{c1[0]:4d} (Q {c0[3]}->{c1[3]}) "
              f"| full {f0[0]:4d}->{f1[0]:4d} (Q {f0[3]}->{f1[3]})")

    print("== Table: 2-bit ripple, per-output cost before X-cancellation ==")
    outs, n = F['2b ripple']; names = ['S0', 'C0', 'S1', 'Cout']
    for label, kw in (('gate-count rule', dict(aware=False)), ('cancellation-aware rule', {})):
        info = {}; g, nanc = synthesize(outs, n, info=info, **kw)
        assert verify(g, n, outs, nanc)
        cells = []
        for j, b in info['base'].items():
            S = outs[j] if b is None else outs[j] ^ outs[b]
            cells.append(f"{names[j]}={(0 if b is None else 1) + cover_cost(S, n)} (from {names[b] if b is not None else '-'})")
        print(f"{label:24s} {' '.join(cells)} | total {len(g)} -> {cancelled_len(g)} after X")
    print("direct per output:", ' '.join(f"{nm}={cover_cost(s, n)}" for nm, s in zip(names, outs)))

    print("== Table: broader functions and head-to-head (gates, Q-cost) ==")
    for k in ['Full adder', '2b ripple', 'BCD-XS3', '3x3 mult', '4b comparator', '3_17']:
        outs, n = F[k]; rg = raw(outs, n)
        (c0, c1, _) = run(outs, n, share=False, derive=False)
        (o0, o1, _) = run(outs, n, aware=False)
        info = {}; synthesize(outs, n, info=info)
        (f0, f1, na) = run(outs, n)
        print(f"{k:14s} raw {len(rg):4d} | cover {c0[0]:4d}->{c1[0]:4d} (Q {c1[3]}) | original full {o0[0]:4d}->{o1[0]:4d} (Q {o1[3]}) "
              f"| full {f0[0]:4d}->{f1[0]:4d} (Q {f1[3]}, anc={na}, used={info['used']})")

    print("== Table: term ordering (gate count after X-cancellation) ==")
    for name, outs, n in functions():
        cb, _ = synthesize(outs, n, share=False, derive=False, blocks=True)
        ch = [g for b in heuristic(cb, n) for g in b]
        assert verify(ch, n, outs)
        fg, _ = synthesize(outs, n)
        og, na = ordered(outs, n)
        assert verify(og, n, outs, na)
        e = best_exact(outs, n)
        if name == 'Full adder':
            costs = [cancelled_len([g for i in p for g in cb[i]]) for p in permutations(range(len(cb)))]
            print(f"  full adder cover, all {len(costs)} term orders: best {min(costs)}, reached by {costs.count(min(costs))}")
        print(f"{name:14s} cover+X {cancelled_len([g for b in cb for g in b]):4d} -> ordered {cancelled_len(ch):4d} "
              f"| full+X {cancelled_len(fg):4d} -> ordered {cancelled_len(og):4d} (Q {sum(map(__import__('gatelevel').qcost, cancel_xx(og)))}) "
              f"| exact best order {e if e is not None else '-'}")
