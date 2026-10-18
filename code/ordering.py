"""Term-ordering for X-cancellation: greedy construction + insertion local search,
and an exact optimum (dynamic programming) for circuits with few blocks.

A block (synth_full.emit_blocks) is moved as a unit. Blocks commute except that a
block reading wire r as a control must follow every block that writes r (a
derivation CNOT reads its base output; nothing else reads a non-input wire outside
its own block), so every order kept here computes the same function."""
from synth_full import cancelled_len, synthesize

def wires(block):
    rd, wr = set(), set()
    for g in block:
        if g[0] == 'MCX':
            rd.update(g[1]); wr.add(g[2])
    return rd - wr, wr  # ancilla reads inside a shared block are internal

def precedence(blocks, n):
    """pred[b] = blocks that must come before b."""
    io = [wires(b) for b in blocks]
    return [{a for a in range(len(blocks)) if a != b and io[a][1] & {r for r in io[b][0] if r >= n}}
            for b in range(len(blocks))]

def valid(seq, pred):
    pos = {b: i for i, b in enumerate(seq)}
    return all(pos[a] < pos[b] for b in seq for a in pred[b])

def cost(seq, blocks):
    return cancelled_len([g for b in seq for g in blocks[b]])

def greedy(blocks, pred):
    seq, left = [], set(range(len(blocks)))
    while left:
        ready = sorted(b for b in left if pred[b] <= set(seq))
        b = min(ready, key=lambda b: (cost(seq + [b], blocks), b))
        seq.append(b); left.discard(b)
    return seq

def local_search(seq, blocks, pred):
    """First-improvement insertion moves until no single move lowers the cost."""
    seq, best = list(seq), cost(seq, blocks)
    improved = True
    while improved:
        improved = False
        for i in range(len(seq)):
            for k in range(len(seq)):
                if k == i: continue
                s = seq[:i] + seq[i + 1:]
                s.insert(k, seq[i])
                if not valid(s, pred): continue
                c = cost(s, blocks)
                if c < best:
                    seq, best, improved = s, c, True
                    break
            if improved: break
    return seq

def heuristic(blocks, n):
    """Best of local search started from the emitted order and from the greedy order."""
    pred = precedence(blocks, n)
    seqs = [local_search(s, blocks, pred) for s in (list(range(len(blocks))), greedy(blocks, pred))]
    seq = min(seqs, key=lambda s: cost(s, blocks))
    assert valid(seq, pred)
    return [blocks[b] for b in seq]

def exact(blocks, n, limit=2_000_000):
    """Minimum post-cancellation gate count over all valid block orders, by DP over
    (set of placed blocks, X-parity of every input wire since its last control use).
    Returns None if the state space exceeds `limit`."""
    pred = precedence(blocks, n)
    pm = [sum(1 << a for a in p) for p in pred]
    def step(par, block):
        par, c = list(par), 0
        for g in block:
            if g[0] == 'X':
                par[g[1]] ^= 1
            else:
                c += 1
                for w in g[1]:
                    if w < n: c += par[w]; par[w] = 0
        return tuple(par), c
    full = (1 << len(blocks)) - 1
    layer = {(0, (0,) * n): 0}
    seen = 0
    for _ in blocks:
        nxt = {}
        for (mask, par), c in layer.items():
            for b in range(len(blocks)):
                if mask >> b & 1 or pm[b] & ~mask: continue
                p2, dc = step(par, blocks[b])
                key = (mask | 1 << b, p2)
                if c + dc < nxt.get(key, 1 << 30): nxt[key] = c + dc
        seen += len(nxt)
        if seen > limit: return None
        layer = nxt
    return min(c + sum(par) for (mask, par), c in layer.items() if mask == full)

def ordered(outs, n, **kw):
    """Full pipeline with ordering: reorder each candidate circuit (cancellation-aware,
    plain cover, original rule) and keep the smallest. Returns (gates, nanc)."""
    best = None
    cov = {k: kw[k] for k in ('cover',) if k in kw}
    for c in ([dict(cov, share=False, derive=False)] if not (kw.get('share', True) or kw.get('derive', True))
              else [dict(kw, fallback=False), dict(cov, share=False, derive=False), dict(kw, aware=False)]):
        blocks, nanc = synthesize(outs, n, blocks=True, **c)
        g = [x for b in heuristic(blocks, n) for x in b]
        if best is None or cancelled_len(g) < cancelled_len(best[0]): best = (g, nanc)
    return best
