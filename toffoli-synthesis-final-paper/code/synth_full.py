"""Full gate-level synthesizer: disjoint cover + (optional) sharing + (optional) derivation + X-X cancellation."""
from itertools import product
from collections import defaultdict
from disjoint_synth import disjoint_cover
from gatelevel import emit_term, cancel_xx, simulate, qcost

def term_len(t): return len(emit_term(t, 0))
def cover_cost(S, n): return sum(term_len(t) for t in disjoint_cover(S, n))

def cancelled_len(gates):
    """Gate count after cancel_xx, computed in one pass. X gates only ever sit on
    input wires, so on each wire the X's between two consecutive control uses
    of that wire reduce to their parity (Lemma 3); every MCX survives."""
    par, out = {}, 0
    for g in gates:
        if g[0] == 'X':
            par[g[1]] = par.get(g[1], 0) ^ 1
        else:
            out += 1
            for c in g[1]:
                out += par.pop(c, 0)
    return out + sum(par.values())

def emit_blocks(plan, order, n, m, shared):
    """plan: j -> (base or None, term list). shared: term -> output list.
    Returns (blocks, nanc): each block is one term's gates, one derivation CNOT,
    or one shared term's compute/copy/uncompute sequence."""
    blocks, nanc, done = [], 0, set()
    for j in order:
        tg = n + j
        if plan[j][0] is not None:
            blocks.append([('MCX', (n + plan[j][0],), tg)])
        for t in plan[j][1]:
            if t in shared:
                if t in done: continue
                done.add(t)
                a = n + m + nanc; nanc += 1
                blocks.append(emit_term(t, a) + [('MCX', (a,), n + jj) for jj in shared[t]] + emit_term(t, a))
            else:
                blocks.append(emit_term(t, tg))
    return blocks, nanc

def emit(plan, order, n, m, shared):
    blocks, nanc = emit_blocks(plan, order, n, m, shared)
    return [g for b in blocks for g in b], nanc

def share_candidates(plan, m):
    use = defaultdict(list)
    for j in range(m):
        for t in plan[j][1]:
            if sum(v != -1 for v in t) >= 2: use[t].append(j)
    return {t: js for t, js in sorted(use.items()) if len(js) >= 2}

def synthesize(outs, n, share=True, derive=True, aware=True, fallback=True, info=None, blocks=False):
    """aware=False reproduces the original pipeline, which picks derivation bases and
    shared terms by gate count before X-cancellation. aware=True picks each by the
    gate count after X-cancellation of the circuit built so far, and (fallback=True)
    returns the plain cover circuit or the original pipeline's circuit instead if
    either is smaller after cancellation. info, if a dict, receives the derivation
    bases ('base': j -> base or None) and which circuit was returned ('used').
    blocks=True returns the circuit as a block list (see emit_blocks)."""
    m = len(outs)
    direct = [cover_cost(s, n) for s in outs]
    order = sorted(range(m), key=lambda j: direct[j])
    plan = {}
    if not aware:
        for idx, j in enumerate(order):
            best, base, S = direct[j], None, outs[j]
            if derive:
                for i in order[:idx]:
                    d = outs[j] ^ outs[i]
                    c = 1 + cover_cost(d, n)
                    if c < best: best, base, S = c, i, d
            plan[j] = (base, disjoint_cover(S, n))
        shared = {}
        if share:
            for t, js in share_candidates(plan, m).items():
                if 2 * term_len(t) + len(js) < len(js) * term_len(t):
                    shared[t] = js
        if info is not None: info.update(base={j: plan[j][0] for j in plan}, used='original')
        return (emit_blocks if blocks else emit)(plan, order, n, m, shared)
    prefix = []
    for idx, j in enumerate(order):
        cands = [(None, outs[j])]
        if derive: cands += [(i, outs[j] ^ outs[i]) for i in order[:idx]]
        best = None
        for base, S in cands:
            terms = disjoint_cover(S, n)
            block = ([('MCX', (n + base,), n + j)] if base is not None else [])
            block += [g for t in terms for g in emit_term(t, n + j)]
            c = cancelled_len(prefix + block)
            if best is None or c < best[0]: best = (c, base, terms, block)
        plan[j] = (best[1], best[2])
        prefix += best[3]
    shared = {}
    if share:
        cur = cancelled_len(emit(plan, order, n, m, shared)[0])
        for t, js in share_candidates(plan, m).items():
            trial = dict(shared); trial[t] = js
            c = cancelled_len(emit(plan, order, n, m, trial)[0])
            if c < cur: shared, cur = trial, c
    gates, nanc = (emit_blocks if blocks else emit)(plan, order, n, m, shared)
    if info is not None: info.update(base={j: plan[j][0] for j in plan}, used='aware')
    flat = (lambda b: [g for x in b for g in x]) if blocks else (lambda g: g)
    if fallback and (share or derive):
        for alt, kw in (('cover', dict(share=False, derive=False)),
                        ('original', dict(share=share, derive=derive, aware=False))):
            sub = {}
            g, na = synthesize(outs, n, info=sub, blocks=blocks, **kw)
            if cancelled_len(flat(g)) < cancelled_len(flat(gates)):
                gates, nanc = g, na
                if info is not None: info.update(sub, used=alt)
    return gates, nanc

def verify(gates, n, outs, nanc=0):
    m = len(outs)
    for x in product([0, 1], repeat=n):
        res = simulate(gates, list(x) + [0] * (m + nanc))
        if tuple(res[:n]) != x or any(res[n + m + k] for k in range(nanc)): return False
        for j, s in enumerate(outs):
            if res[n + j] != (1 if x in s else 0): return False
    return True

def stats(gates):
    nx = sum(g[0] == 'X' for g in gates)
    return len(gates), nx, len(gates) - nx, sum(map(qcost, gates))

def run(outs, n, **kw):
    g, nanc = synthesize(outs, n, **kw)
    c = cancel_xx(g)
    assert len(c) == cancelled_len(g)
    assert verify(g, n, outs, nanc) and verify(c, n, outs, nanc), "VERIFY FAILED"
    return stats(g), stats(c), nanc
