"""Save a picture of every circuit: this work's circuit after cover + X-cancellation, after the
full pipeline, and after term ordering, plus tweedledum's PKRM and PPRM circuits (all after
X-cancellation, every one checked against its truth table first).
Needs matplotlib and results/tweedledum_circuits.json (from baselines.py).
Run: python3 draw.py [workers]   (writes circuits/*.png)"""
import io, os, re, sys, contextlib
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context  # spawn everywhere, as on macOS and Windows
from synth_full import synthesize, verify
from gatelevel import cancel_xx, qcost
from ordering import ordered

# wire labels, inputs MSB-first then outputs in the order recompute.functions() uses
LABELS = {
    'AND': (['A', 'B'], ['Out']), 'OR': (['A', 'B'], ['Out']), 'XOR': (['A', 'B'], ['Out']),
    'NAND': (['A', 'B'], ['Out']), 'Half adder': (['A', 'B'], ['Sum', 'Carry']),
    'Full adder': (['A', 'B', 'Cin'], ['Sum', 'Carry']),
    '2b ripple': (['A1', 'A0', 'B1', 'B0', 'Cin'], ['S0', 'C0', 'S1', 'Cout']),
    'BCD-XS3': (['B3', 'B2', 'B1', 'B0'], ['E3', 'E2', 'E1', 'E0']),  # BCD in, Excess-3 out, MSB first
    '3x3 mult': (['X2', 'X1', 'X0', 'Y2', 'Y1', 'Y0'], ['P5', 'P4', 'P3', 'P2', 'P1', 'P0']),
    '4b comparator': (['A3', 'A2', 'A1', 'A0', 'B3', 'B2', 'B1', 'B0'], ['EQ']),
    '3_17': (['a', 'b', 'c'], ['f1', 'f2', 'f3']),
}
STAGES = [('cover', 'cover + X-cancellation'), ('full', 'full pipeline'), ('ordered', 'full pipeline + term ordering'),
          ('pkrm', 'tweedledum PKRM'), ('pprm', 'tweedledum PPRM')]

def columns(gates):
    """Place each gate in the earliest column after every earlier gate that overlaps its wire span."""
    last, cols = {}, []
    for g in gates:
        ws = [g[1]] if g[0] == 'X' else list(g[1]) + [g[2]]
        lo, hi = min(ws), max(ws)
        c = max([last.get(w, -1) for w in range(lo, hi + 1)], default=-1) + 1
        for w in range(lo, hi + 1): last[w] = c
        cols.append(c)
    return cols

def draw(gates, labels, title, path, dpi=150):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    cols = columns(gates)
    W, H = (max(cols) + 1 if cols else 1), len(labels)
    fig, ax = plt.subplots(figsize=(max(4, 1.2 + 0.42 * W), 0.6 + 0.42 * H))
    y = lambda w: H - 1 - w
    for w, lab in enumerate(labels):
        ax.plot([-0.6, W - 0.4], [y(w), y(w)], color='black', lw=0.8, zorder=0)
        ax.text(-0.75, y(w), lab, ha='right', va='center', fontsize=9)
    for g, c in zip(gates, cols):
        if g[0] == 'X' or not g[1]:
            w = g[1] if g[0] == 'X' else g[2]
            ax.add_patch(plt.Rectangle((c - 0.22, y(w) - 0.22), 0.44, 0.44, fc='white', ec='black', lw=0.8, zorder=2))
            ax.text(c, y(w), 'X', ha='center', va='center', fontsize=8, zorder=3)
            continue
        ws = list(g[1]) + [g[2]]
        ax.plot([c, c], [y(min(ws)), y(max(ws))], color='black', lw=0.8, zorder=1)
        for w in g[1]:
            ax.plot(c, y(w), 'o', color='black', ms=5, zorder=2)
        ax.add_patch(plt.Circle((c, y(g[2])), 0.17, fc='white', ec='black', lw=0.8, zorder=2))
        ax.plot([c - 0.17, c + 0.17], [y(g[2])] * 2, color='black', lw=0.8, zorder=3)
        ax.plot([c, c], [y(g[2]) - 0.17, y(g[2]) + 0.17], color='black', lw=0.8, zorder=3)
    ax.set_xlim(-1.6, W - 0.2); ax.set_ylim(-0.6, H - 0.4)
    ax.set_aspect('equal'); ax.axis('off')
    ax.set_title(title, fontsize=10, loc='left')
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)

def circuit(name, stage, outs, n):
    if stage == 'cover': g, a = synthesize(outs, n, share=False, derive=False)
    elif stage == 'full': g, a = synthesize(outs, n)
    elif stage == 'ordered': g, a = ordered(outs, n)
    else:
        from table import load_tweedledum  # no qiskit needed
        return load_tweedledum()[(name, stage)], 0
    return cancel_xx(g), a

def job(args):
    name, stage, desc, outdir = args
    with contextlib.redirect_stdout(io.StringIO()):
        import recompute
    outs, n = {f: (o, k) for f, o, k in recompute.functions()}[name]
    gates, nanc = circuit(name, stage, outs, n)
    assert verify(gates, n, outs, nanc), (name, stage)
    ins, outl = LABELS.get(name, ([f'x{i}' for i in range(n)], [f'f{j}' for j in range(len(outs))]))
    labels = ins + outl + [f'anc{k}' for k in range(nanc)]
    path = os.path.join(outdir, re.sub(r'[^A-Za-z0-9]+', '_', name).strip('_') + f'_{stage}.png')
    draw(gates, labels, f"{name}, {desc}: {len(gates)} gates, quantum cost {sum(map(qcost, gates))}", path)
    return path

def main(workers=None, outdir='circuits'):
    with contextlib.redirect_stdout(io.StringIO()):
        import recompute
    os.makedirs(outdir, exist_ok=True)
    jobs = [(f, s, d, outdir) for f, _, _ in recompute.functions() for s, d in STAGES]
    with ProcessPoolExecutor(workers, mp_context=get_context('spawn')) as ex:
        paths = list(ex.map(job, jobs))
    print(f"saved {len(paths)} circuit pictures to {outdir}/ (every circuit verified on all inputs first)")
    return paths

if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else os.cpu_count())
