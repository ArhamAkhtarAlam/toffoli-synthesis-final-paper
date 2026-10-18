"""
Reconstruction of the GrowCube / DisjointCover algorithm and the sharing /
derivation optimizations described in:

  "From Truth Table to Quantum Circuit: K-Map-Guided Toffoli Synthesis
   with Disjoint Cube Covers"

This is an independent re-implementation from the paper's pseudocode
(Algorithms 1-2, Definitions, Lemmas) so we can run it against standard
benchmark functions and get real, reproducible gate counts for the
extended paper's experimental section.

Pattern representation: a tuple of length n over {0,1,-1} where -1 means
"don't care" (dash). We use -1 instead of '-' for easy arithmetic.
"""

from itertools import product
import random

# ---------------------------------------------------------------------
# Core cube / cover machinery
# ---------------------------------------------------------------------

def cube_cells(pattern):
    """All points in {0,1}^n covered by a pattern (tuple w/ -1 = dash)."""
    free_positions = [i for i, v in enumerate(pattern) if v == -1]
    base = list(pattern)
    cells = []
    for bits in product([0, 1], repeat=len(free_positions)):
        p = base[:]
        for pos, b in zip(free_positions, bits):
            p[pos] = b
        cells.append(tuple(p))
    return cells


def grow_cube(seed, remaining_set, n):
    """Algorithm 1: GrowCube. Greedily dash out positions while the
    resulting cube stays a subset of `remaining_set`."""
    p = list(seed)
    for i in range(n):
        if p[i] == -1:
            continue
        trial = p[:]
        trial[i] = -1
        if all(c in remaining_set for c in cube_cells(trial)):
            p = trial
    return tuple(p)


def disjoint_cover(on_set, n):
    """Algorithm 2: DisjointCover. on_set: set of tuples in {0,1}^n."""
    remaining = set(on_set)
    selected = []
    while remaining:
        seed = min(remaining)  # lexicographically smallest, as in the paper
        p = grow_cube(seed, remaining, n)
        selected.append(p)
        for c in cube_cells(p):
            remaining.discard(c)
    return selected


# ---------------------------------------------------------------------
# Gate costing (matches Section "From Terms to Circuits")
# ---------------------------------------------------------------------

def term_gate_count(pattern):
    """Gate count contributed by a single term, per the paper's table:
    1 (no vars) -> 1 unconditional X
    1 var       -> 1 CNOT
    k>=2 vars, all uncomplemented -> 1 k-controlled Toffoli
    k>=2 vars, some complemented  -> 2*(#complemented) extra X's + 1 Toffoli
    """
    controls = [(i, v) for i, v in enumerate(pattern) if v != -1]
    k = len(controls)
    if k == 0:
        return 1
    if k == 1:
        return 1
    complemented = sum(1 for _, v in controls if v == 0)
    return 1 + 2 * complemented  # the k-controlled Toffoli itself + X-sandwich


def synthesize_output_direct(on_set, n):
    """Disjoint-cover synthesis cost for one output, no cross-output opts."""
    terms = disjoint_cover(on_set, n)
    cost = sum(term_gate_count(t) for t in terms)
    return terms, cost


def synthesize_raw(on_set):
    """'Raw' baseline: one multi-controlled gate per minterm, no
    minimization at all (the naive one-Toffoli-per-row approach)."""
    cost = 0
    for row in on_set:
        k = len(row)
        if k == 0:
            cost += 1
        elif k == 1:
            cost += 1
        else:
            complemented = sum(1 for v in row if v == 0)
            cost += 1 + 2 * complemented
    return cost


# ---------------------------------------------------------------------
# Sharing optimization (Lemma: Sharing) -- share identical >=2-control
# terms across multiple outputs via one scratch ancilla.
# ---------------------------------------------------------------------

def synthesize_with_sharing(on_sets, n):
    """on_sets: list of on-sets, one per output.
    Returns (total_cost, per_output_terms, shared_term_count)."""
    per_output_terms = [disjoint_cover(s, n) for s in on_sets]

    # Count which >=2-control terms are shared across >=2 outputs
    from collections import defaultdict
    term_to_outputs = defaultdict(set)
    for j, terms in enumerate(per_output_terms):
        for t in terms:
            k = sum(1 for v in t if v != -1)
            if k >= 2:
                term_to_outputs[t].add(j)

    shared_terms = {t: outs for t, outs in term_to_outputs.items() if len(outs) >= 2}

    total = 0
    accounted = set()  # (output_idx, term) pairs already paid for via sharing
    for t, outs in shared_terms.items():
        # compute once (term_gate_count), uncompute once (same cost again),
        # then one CNOT per consuming output
        compute_cost = term_gate_count(t)
        total += 2 * compute_cost + len(outs)  # compute + uncompute + CNOTs
        for j in outs:
            accounted.add((j, t))

    for j, terms in enumerate(per_output_terms):
        for t in terms:
            if (j, t) not in accounted:
                total += term_gate_count(t)

    return total, per_output_terms, len(shared_terms)


# ---------------------------------------------------------------------
# Derivation optimization (Lemma: Derivation)
# ---------------------------------------------------------------------

def synthesize_with_derivation(on_sets, n):
    """Greedy: process outputs in increasing direct-cost order; each later
    output may instead be derived from the cheapest earlier output j via
    CNOT + disjoint-cover(f_k XOR f_j)."""
    m = len(on_sets)
    direct_costs = []
    for s in on_sets:
        _, c = synthesize_output_direct(s, n)
        direct_costs.append(c)

    order = sorted(range(m), key=lambda j: direct_costs[j])
    built_cost = [None] * m
    built_from = [None] * m

    for idx, j in enumerate(order):
        best_cost = direct_costs[j]
        best_from = None
        for i in order[:idx]:
            # difference set: symmetric difference of the two on-sets
            diff = on_sets[j].symmetric_difference(on_sets[i])
            _, cdiff = synthesize_output_direct(diff, n)
            cand_cost = 1 + cdiff  # 1 CNOT + disjoint-cover(diff)
            if cand_cost < best_cost:
                best_cost = cand_cost
                best_from = i
        built_cost[j] = best_cost
        built_from[j] = best_from

    total = sum(built_cost)
    return total, built_cost, built_from, direct_costs


# ---------------------------------------------------------------------
# Benchmark truth-table builders
# ---------------------------------------------------------------------

def full_adder():
    n = 3
    sum_set, carry_set = set(), set()
    for a, b, cin in product([0, 1], repeat=3):
        s = a ^ b ^ cin
        c = (a & b) | (b & cin) | (a & cin)
        if s: sum_set.add((a, b, cin))
        if c: carry_set.add((a, b, cin))
    return n, [sum_set, carry_set]


def ripple_carry_adder_2bit():
    n = 5  # A1 A0 B1 B0 Cin
    outs = {"S0": set(), "C0": set(), "S1": set(), "Cout": set()}
    for a1, a0, b1, b0, cin in product([0, 1], repeat=5):
        s0 = a0 ^ b0 ^ cin
        c0 = (a0 & b0) | (b0 & cin) | (a0 & cin)
        s1 = a1 ^ b1 ^ c0
        cout = (a1 & b1) | (b1 & c0) | (a1 & c0)
        row = (a1, a0, b1, b0, cin)
        if s0: outs["S0"].add(row)
        if c0: outs["C0"].add(row)
        if s1: outs["S1"].add(row)
        if cout: outs["Cout"].add(row)
    return n, outs


def bcd_to_excess3():
    """Classic reversible-logic benchmark: BCD-to-Excess-3 code converter,
    4 inputs -> 4 outputs (standard RevLib-style benchmark)."""
    n = 4
    outs = {f"Y{i}": set() for i in range(4)}
    for bits in product([0, 1], repeat=4):
        val = int("".join(map(str, bits)), 2)
        if val > 9:
            continue  # BCD only defines 0-9; others are don't-cares (skip)
        out_val = val + 3
        out_bits = tuple(int(b) for b in format(out_val, "04b"))
        for i in range(4):
            if out_bits[i]:
                outs[f"Y{i}"].add(bits)
    return n, outs


def four_bit_comparator_eq():
    """4-bit equality comparator: single output, A==B over 8 input bits."""
    n = 8  # a3 a2 a1 a0 b3 b2 b1 b0
    out = set()
    for bits in product([0, 1], repeat=8):
        a = bits[:4]
        b = bits[4:]
        if a == b:
            out.add(bits)
    return n, {"EQ": out}


def three_bit_multiplier():
    """3x3 -> 6-bit multiplier (classic RevLib-style benchmark, 'mult').
    n=6 inputs, m=6 outputs (upper product bits beyond 6 are impossible
    since max product of two 3-bit numbers (7*7=49) fits in 6 bits)."""
    n = 6
    outs = {f"P{i}": set() for i in range(6)}
    for bits in product([0, 1], repeat=6):
        x = bits[0] * 4 + bits[1] * 2 + bits[2]
        y = bits[3] * 4 + bits[4] * 2 + bits[5]
        prod = x * y
        prod_bits = tuple(int(b) for b in format(prod, "06b"))
        for i in range(6):
            if prod_bits[i]:
                outs[f"P{i}"].add(bits)
    return n, outs


def miller_3_17():
    """RevLib's 3_17: 3-input, 3-output bijection. Same table as
    recompute.PLA_3_17, which every number in the paper uses."""
    n = 3
    # (f1, f2, f3) for input rows 000..111, MSB first
    table = [
        (1, 1, 1),
        (0, 0, 0),
        (0, 0, 1),
        (0, 1, 1),
        (1, 0, 0),
        (0, 1, 0),
        (1, 1, 0),
        (1, 0, 1),
    ]
    outs = {"f1": set(), "f2": set(), "f3": set()}
    for row, (f1, f2, f3) in zip(product([0, 1], repeat=3), table):
        if f1: outs["f1"].add(row)
        if f2: outs["f2"].add(row)
        if f3: outs["f3"].add(row)
    return n, outs


# ---------------------------------------------------------------------
# Randomized validation (matches paper's "validated over N random trials")
# ---------------------------------------------------------------------

def random_validation(trials=2000, max_n=6, seed=0):
    rng = random.Random(seed)
    violations = 0
    for _ in range(trials):
        n = rng.randint(1, max_n)
        universe = list(product([0, 1], repeat=n))
        k = rng.randint(0, len(universe))
        on_set = set(rng.sample(universe, k))
        terms = disjoint_cover(on_set, n)
        covered = set()
        for t in terms:
            cells = cube_cells(t)
            for c in cells:
                if c in covered:
                    violations += 1  # overlap!
                covered.add(c)
        if covered != on_set:
            violations += 1
    return trials, violations


if __name__ == "__main__":
    print("=== Randomized exactness/disjointness validation ===")
    trials, violations = random_validation(trials=2000)
    print(f"{trials} trials, {violations} violations")
    print()

    print("=== Full adder (paper: 16 + 7 = 23) ===")
    n, outs = full_adder()
    total = 0
    for name, s in zip(["Sum", "Carry"], outs):
        terms, cost = synthesize_output_direct(s, n)
        print(f"  {name}: {len(terms)} terms, {cost} gates")
        total += cost
    print(f"  TOTAL: {total}")
    print()

    print("=== 2-bit ripple-carry adder (paper: 126 -> 64) ===")
    n, outs = ripple_carry_adder_2bit()
    names = list(outs.keys())
    sets = [outs[nm] for nm in names]
    direct_total = sum(synthesize_output_direct(s, n)[1] for s in sets)
    deriv_total, built_cost, built_from, direct_costs = synthesize_with_derivation(sets, n)
    print(f"  Direct total: {direct_total}")
    print(f"  With derivation: {deriv_total}")
    for j, name in enumerate(names):
        src = names[built_from[j]] if built_from[j] is not None else "-"
        print(f"    {name}: direct={direct_costs[j]}, used={built_cost[j]}, from={src}")
    print()

    print("=== BCD to Excess-3 (4 in, 4 out) ===")
    n, outs = bcd_to_excess3()
    names = list(outs.keys())
    sets = [outs[nm] for nm in names]
    raw_total = sum(synthesize_raw(s) for s in sets)
    direct_total = sum(synthesize_output_direct(s, n)[1] for s in sets)
    share_total, _, nshared = synthesize_with_sharing(sets, n)
    deriv_total, built_cost, built_from, direct_costs = synthesize_with_derivation(sets, n)
    print(f"  Raw (one gate/minterm): {raw_total}")
    print(f"  Disjoint-cover only:    {direct_total}")
    print(f"  + sharing ({nshared} shared terms): {share_total}")
    print(f"  + derivation:           {deriv_total}")
    print()

    print("=== 4-bit equality comparator (8 in, 1 out) ===")
    n, outs = four_bit_comparator_eq()
    s = outs["EQ"]
    raw = synthesize_raw(s)
    terms, direct = synthesize_output_direct(s, n)
    print(f"  Raw: {raw}, Disjoint-cover: {direct}, #terms: {len(terms)}")
    print()

    print("=== 3x3 multiplier (6 in, 6 out) ===")
    n, outs = three_bit_multiplier()
    names = list(outs.keys())
    sets = [outs[nm] for nm in names]
    raw_total = sum(synthesize_raw(s) for s in sets)
    direct_total = sum(synthesize_output_direct(s, n)[1] for s in sets)
    share_total, _, nshared = synthesize_with_sharing(sets, n)
    deriv_total, built_cost, built_from, direct_costs = synthesize_with_derivation(sets, n)
    print(f"  Raw: {raw_total}")
    print(f"  Disjoint-cover only: {direct_total}")
    print(f"  + sharing ({nshared} shared terms): {share_total}")
    print(f"  + derivation: {deriv_total}")
    for j, name in enumerate(names):
        src = names[built_from[j]] if built_from[j] is not None else "-"
        print(f"    {name}: direct={direct_costs[j]}, used={built_cost[j]}, from={src}")
    print()

    print("=== Miller 3_17 benchmark (3 in, 3 out) ===")
    n, outs = miller_3_17()
    names = list(outs.keys())
    sets = [outs[nm] for nm in names]
    raw_total = sum(synthesize_raw(s) for s in sets)
    direct_total = sum(synthesize_output_direct(s, n)[1] for s in sets)
    share_total, _, nshared = synthesize_with_sharing(sets, n)
    deriv_total, built_cost, built_from, direct_costs = synthesize_with_derivation(sets, n)
    print(f"  Raw: {raw_total}")
    print(f"  Disjoint-cover only: {direct_total}")
    print(f"  + sharing ({nshared} shared terms): {share_total}")
    print(f"  + derivation: {deriv_total}")
