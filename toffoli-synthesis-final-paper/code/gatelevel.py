"""Gate-level emission + X-X cancellation peephole pass for the disjoint-cover synthesizer."""
from itertools import product
from disjoint_synth import (disjoint_cover, full_adder, ripple_carry_adder_2bit, bcd_to_excess3,
                            four_bit_comparator_eq, three_bit_multiplier)

# gate = ('X', wire) | ('MCX', (controls...), target)

def emit_term(pattern, target):
    ctrls = [i for i, v in enumerate(pattern) if v != -1]
    comp = [i for i in ctrls if pattern[i] == 0]
    g = [('X', w) for w in comp]
    g.append(('MCX', tuple(ctrls), target))
    g += [('X', w) for w in comp]
    return g

def emit_output(on_set, n, target):
    g = []
    for t in disjoint_cover(on_set, n):
        g += emit_term(t, target)
    return g

def touches(gate, w):
    return gate[1] == w if gate[0] == 'X' else (w in gate[1] or w == gate[2])

def commutes_with_x(gate, w):
    """An X on wire w commutes with gate if gate doesn't touch w, or w is gate's TARGET (X commutes with a flip)."""
    if gate[0] == 'X':
        return True  # X's commute with each other (different wire) or cancel (same wire)
    return (w not in gate[1])  # not a control (target or untouched)

def cancel_xx(gates):
    """Repeatedly cancel pairs of X on the same wire separated only by gates the X commutes with."""
    gates = list(gates)
    changed = True
    while changed:
        changed = False
        for i, g in enumerate(gates):
            if g[0] != 'X':
                continue
            w = g[1]
            for j in range(i + 1, len(gates)):
                h = gates[j]
                if h[0] == 'X' and h[1] == w:
                    del gates[j]; del gates[i]
                    changed = True
                    break
                if not commutes_with_x(h, w):
                    break
            if changed:
                break
    return gates

def simulate(gates, bits):
    s = list(bits)
    for g in gates:
        if g[0] == 'X':
            s[g[1]] ^= 1
        elif all(s[c] for c in g[1]):
            s[g[2]] ^= 1
    return s

def verify(gates, n, outs):
    m = len(outs)
    for x in product([0, 1], repeat=n):
        res = simulate(gates, list(x) + [0] * m)
        if tuple(res[:n]) != x: return False
        for j, s in enumerate(outs):
            if res[n + j] != (1 if x in s else 0): return False
    return True

def build(outs, n):
    g = []
    for j, s in enumerate(outs):
        g += emit_output(s, n, n + j)
    return g

def qcost(g):
    if g[0] == 'X': return 1
    k = len(g[1])
    return 1 if k <= 1 else 2 ** (k + 1) - 3

def report(name, n, outs):
    outs = list(outs)
    g = build(outs, n)
    c = cancel_xx(g)
    ok = verify(g, n, outs) and verify(c, n, outs)
    nx = lambda gl: sum(1 for x in gl if x[0] == 'X')
    mcx = lambda gl: sum(1 for x in gl if x[0] != 'X')
    print(f"{name:14s} before={len(g):4d} (X={nx(g)}, MCX={mcx(g)})  after={len(c):4d} (X={nx(c)})  "
          f"saved={len(g)-len(c)} ({100*(len(g)-len(c))/len(g):.0f}%)  Qcost {sum(map(qcost,g))}->{sum(map(qcost,c))}  verified={ok}")
    return g, c

if __name__ == '__main__':
    # Outputs are emitted in natural order here; recompute.py (the paper's numbers) emits
    # them in increasing-cost order, so some totals after cancellation differ.
    from itertools import product as P
    def mk(fn, n, m):
        o = [set() for _ in range(m)]
        for row in P([0, 1], repeat=n):
            r = fn(*row); r = r if isinstance(r, tuple) else (r,)
            for i, v in enumerate(r):
                if v: o[i].add(row)
        return o
    pre = {'AND': (lambda a,b:(a&b,),2,1), 'OR': (lambda a,b:(a|b,),2,1), 'XOR': (lambda a,b:(a^b,),2,1),
           'NAND': (lambda a,b:(1-(a&b),),2,1), 'Half adder': (lambda a,b:(a^b,a&b),2,2),
           'Full adder': (lambda a,b,c:(a^b^c,(a&b)|(b&c)|(a&c)),3,2)}
    for k,(fn,n,m) in pre.items(): report(k, n, mk(fn,n,m))
    n,o = ripple_carry_adder_2bit(); report('2b ripple (direct)', n, [o[k] for k in ['S0','C0','S1','Cout']])
    n,o = bcd_to_excess3(); report('BCD-XS3', n, [o[k] for k in sorted(o)])
    n,o = three_bit_multiplier(); report('3x3 mult', n, [o[k] for k in sorted(o)])
    n,o = four_bit_comparator_eq(); report('4b comparator', n, [o['EQ']])
    pla={(0,0,0):(1,1,1),(0,0,1):(0,0,0),(0,1,0):(0,0,1),(0,1,1):(0,1,1),(1,0,0):(1,0,0),(1,0,1):(0,1,0),(1,1,0):(1,1,0),(1,1,1):(1,0,1)}
    o=[set(),set(),set()]
    for i,v in pla.items():
        for j in range(3):
            if v[j]: o[j].add(i)
    report('3_17', 3, o)
