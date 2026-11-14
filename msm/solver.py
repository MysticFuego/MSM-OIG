"""
The solver. Three steps, each locking in the result of the one before:

1. Pack     the most income (income mode) or fill squares (fill mode). Looks
            at the whole island and only at sizes, so it is fast and proven.
2. Mirror   the most structures mirrored across the in-game axis; a structure
            counts when one of the same size sits in its mirror spot. It first
            tries "every structure mirrored" (fast, and optimal when it works),
            then maximises over every spot on the island. Named types are then
            handed out over those spots so the same type sits opposite itself
            wherever possible.
3. Arrange  back to front: structures, extras, then the collection spread by
            size (4x4 at the back of its area, 1x1 at the front). Cosmetic, so
            it stops once it stops improving, within 1% of its bound, or at
            its time limit.

Steps 2 and 3 start from the previous step's complete layout and keep every
piece count fixed, so they can never lower the income / fill of step 1, and
running out of time only means a less tidy arrangement.
"""

from __future__ import annotations

import os
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
from ortools.sat.python import cp_model

from .config import SIZES, Island, Kind, Lock
from .grid import Axis, Height, Pos, find_axis, fits, load_grid, square

ARRANGE_GAP = 0.01
ARRANGE_PATIENCE = 20       # arrange stops after this many seconds without a real improvement ...
ARRANGE_MIN_GAIN = 0.0005   # ... "real" = at least 0.05% better
PACK_MIN_SECS = 5           # packing always gets at least this long, even with a short time limit
PACK_WORKERS = 16           # packing uses at least this many workers if the machine has them
# workers=None: this computer's physical cores, up to this many. Measured on a
# 16-core i7-13700HX (8 fast + 8 efficiency cores), full solves of 5 islands:
# 8 workers 292s, 16 workers 402s, 24 workers 420s, with 8 also giving the
# tidiest arrangements. Packing is the exception (see PACK_WORKERS).
AUTO_MAX_WORKERS = 8
SYMMETRIC_PASS_MAX = 60     # cap on each "every structure mirrored" attempt
# Structures aim for the back part of the island: the "every structure
# mirrored" attempt and the arrange step try these spots first (speed). The
# spots step 1 and the mirror step used, and their mirrors, are always added.
# The mirror step's maximisation uses every spot, so "proven best" is exact.
STRUCTURE_BAND = {4: 0.40, 3: 0.50, 2: 0.60, 1: 1.0}
# Collection: where each size aims, as a fraction of the area in front of
# the structures and extras (0 = its back edge, 1 = the island's front).
COLLECTION_TARGET = {4: 0.10, 3: 0.32, 2: 0.66, 1: 0.92}
SIZE_WEIGHT = {4: 6, 3: 4, 2: 2, 1: 1}


@dataclass(frozen=True)
class Placement:
    x: int
    y: int
    s: int
    kind: str
    group: str
    locked: bool = False


@dataclass
class Layout:
    island: Island
    grid: np.ndarray
    axis: Axis
    height: Height
    kinds: list[Kind]
    placements: list[Placement]
    income: int | None = None          # income mode
    income_bound: int | None = None
    fill: int | None = None            # fill mode
    fill_bound: int | None = None
    packing_optimal: bool = False      # income / fill proven best
    mirrored: int | None = None        # structures with a same-size one in the mirror spot; None = symmetry off
    same_type: int | None = None       # ... of those, with the same type there
    mirrored_bound: int | None = None
    symmetry_optimal: bool = False
    arrange_gap: float | None = None   # distance from the best arrangement
    seconds: dict = field(default_factory=dict)

    @property
    def mix(self) -> dict[str, int]:
        return {k.name: k.count for k in self.kinds if k.group == "extra"}

    @property
    def beds_used(self) -> int:
        return sum(k.beds * k.count for k in self.kinds if k.group == "extra")

    @property
    def structures_total(self) -> int:
        return sum(k.count for k in self.kinds if k.group == "structure")

    @property
    def free_cells(self) -> int:
        return int(self.grid.sum()) - sum(p.s * p.s for p in self.placements)


# --------------------------------------------------------------------------- #
def solve(island: Island, *, time_limit: float | None = None, arrange_time_limit: float | None = 120,
          symmetry: bool = True, workers: int | None = None, log: bool = False) -> Layout:
    """
    Solve one island. time_limit=None runs packing and symmetry until proven
    best; arrange_time_limit caps only the cosmetic last step. workers=None
    picks the number for this computer (see auto_workers); a number overrides it.
    Raises ValueError for bad settings and RuntimeError if nothing fits.
    """
    for label, t in (("time_limit", time_limit), ("arrange_time_limit", arrange_time_limit)):
        if t is not None and t <= 0:
            raise ValueError(f"{label} must be more than 0 seconds, or None for no limit; got {t}.")
    if workers is None:
        workers = auto_workers()
    elif int(workers) < 1:
        raise ValueError(f"workers must be at least 1, or None for automatic; got {workers}.")
    t0 = time.time()
    say = (lambda m: print(f"[{datetime.now():%H:%M:%S}] {island.name}: {m}", flush=True)) if log else (lambda m: None)

    def left(share=1.0):
        return None if time_limit is None else max(1.0, share * (time_limit - (time.time() - t0)))

    grid = load_grid(island.csv)
    n_open = int(grid.sum())
    kinds = island.kinds(n_open)
    locks = island.locks(kinds)
    axis = find_axis(grid)
    height = Height(grid, axis, island.flip_up)
    spots = {s: fits(grid, s) for s in SIZES}
    _check_locks(island.name, grid, kinds, locks)
    say(f"{n_open} open tiles, mirror axis {axis.kind} c={axis.c} ({axis.matched}/{axis.total} tiles mirror),"
        f" {workers} workers ({_pack_workers(workers)} for packing)")

    # ---- 1. pack ---------------------------------------------------------- #
    pack_limit = None if time_limit is None else max(PACK_MIN_SECS, time_limit * (0.4 if symmetry else 0.6))
    layout, pack = _pack(grid, kinds, locks, spots, island, height, pack_limit, _pack_workers(workers))
    out = Layout(island, grid, axis, height, kinds, layout, packing_optimal=pack["optimal"])
    if island.income_mode:
        out.income, out.income_bound = pack["value"], pack["bound"]
        say(f"income {out.income:,} (bound {out.income_bound:,}) {out.mix} | {pack['seconds']:.1f}s")
    else:
        out.fill, out.fill_bound = pack["value"], pack["bound"]
        say(f"fill {out.fill} (bound {out.fill_bound}) | {pack['seconds']:.1f}s")
    out.seconds["pack"] = pack["seconds"]

    # Steps 2-3 work on shapes: structures of one size are one interchangeable
    # group ("structures 4x4"); their names are handed out at the end, putting
    # the same type opposite each other wherever the shape allows.
    pools = _pools(kinds, spots, height, axis, layout, locks)
    skinds, s_of = _merge(kinds, monsters=False)
    spools = _merged_pools(skinds, s_of, pools)
    slocks = _rename_locks(locks, s_of)
    layout = _rename(layout, s_of)
    structures = [k for k in skinds if k.group == "structure" and k.count]

    # ---- 2. mirror -------------------------------------------------------- #
    if symmetry and structures:
        res = _mirror_step(grid, height, skinds, spools, slocks, structures, axis, spots, layout, left, workers, say)
        out.seconds["mirror"] = res["seconds"]
        if res["layout"] is not None:
            layout = res["layout"]
            out.mirrored, out.mirrored_bound, out.symmetry_optimal = res["value"], res["bound"], res["optimal"]
            for k in structures:  # arrange may use every spot the mirror step chose, and its mirror
                used = {(p.x, p.y) for p in layout if p.kind == k.name}
                used |= {axis.mirror_square(*q, k.size) for q in used} & set(spots[k.size])
                spools[k.name] = sorted(set(spools[k.name]) | used)
            say(f"mirrored {out.mirrored}/{out.structures_total} (bound {out.mirrored_bound}) | {res['seconds']:.1f}s")
        else:
            say(f"mirror step found nothing in time; keeping step 1's layout | {res['seconds']:.1f}s")

    # ---- 3. arrange ------------------------------------------------------- #
    behind = sum(k.size ** 2 * k.count for k in kinds if k.back)
    front = height.depth_after(grid, behind)
    m, v = _kind_model(grid, skinds, spools, slocks)
    if out.mirrored:
        m.Add(sum(_mirror_terms(m, v, structures, axis, hint=layout)) >= out.mirrored)
    by_name = {k.name: k for k in skinds}
    m.Maximize(sum(_preference(by_name[name], height, front, x, y) * var
                   for name, vm in v.items() for (x, y), var in vm.items()))
    _hint(m, v, layout)
    limit = left()
    if arrange_time_limit:
        limit = arrange_time_limit if limit is None else min(limit, arrange_time_limit)
    solver, status, secs = _run(m, limit, workers, gap=ARRANGE_GAP, patience=ARRANGE_PATIENCE,
                                min_gain=ARRANGE_MIN_GAIN)
    out.seconds["arrange"] = secs
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        layout = _extract(solver, v, skinds, slocks)
        obj, bound = solver.ObjectiveValue(), solver.BestObjectiveBound()
        out.arrange_gap = max(0.0, (bound - obj) / max(1.0, abs(obj)))
    say(f"arranged ({out.arrange_gap or 0:.1%} from the solver's bound) | {secs:.1f}s")

    layout, out.same_type = _name_structures(layout, kinds, locks, axis, workers)
    out.placements = layout
    if symmetry and structures:  # count on the final layout (arrange may have mirrored more)
        out.mirrored = count_mirrored(layout, axis)
        bound = out.structures_total if out.mirrored_bound is None else out.mirrored_bound
        out.mirrored_bound = max(out.mirrored, bound)
        out.symmetry_optimal = out.symmetry_optimal or out.mirrored == out.structures_total
    out.seconds["total"] = time.time() - t0
    problems = check(out)
    if problems:  # should never happen; refuse to hand back a broken layout
        raise RuntimeError(f"{island.name}: invalid layout: " + "; ".join(problems))
    return out


# --------------------------------------------------------------------------- #
# 1. packing: sizes only, counts of extras / fill squares chosen by the solver
# --------------------------------------------------------------------------- #
def auto_workers() -> int:
    """
    Workers for this computer: its physical cores, at most AUTO_MAX_WORKERS.
    CP-SAT runs a different strategy on each worker. Fewer cores than that
    means fewer workers; beyond it the extra strategies didn't pay off for
    these models, and hyper-threads (two threads sharing one core) only slow
    each strategy down. Without psutil the logical CPU count is used.
    """
    return max(1, min(AUTO_MAX_WORKERS, physical_cores()))


def physical_cores() -> int:
    """Physical CPU cores (psutil), or logical CPUs if psutil is missing."""
    try:
        import psutil
        cores = psutil.cpu_count(logical=False)
    except Exception:  # psutil missing or unsupported here
        cores = None
    return cores or os.cpu_count() or 1


def _pack_workers(workers: int) -> int:
    """
    Packing uses at least PACK_WORKERS when the machine has them: with 16
    workers CP-SAT adds bound-proving strategies that 8 lack (Ethereal: proven
    in ~80s instead of ~450s). Usually packing takes seconds either way.
    """
    return max(int(workers), min(PACK_WORKERS, os.cpu_count() or 1))


def _pack(grid, kinds, locks, spots, island, height, time_limit, workers):
    m = cp_model.CpModel()
    n_open = int(grid.sum())
    n = {k.name: m.NewIntVar(k.min, k.max, k.name) for k in kinds if k.group == "extra"}
    fill = next((k for k in kinds if k.group == "fill"), None)
    if fill:
        n[fill.name] = m.NewIntVar(0, n_open // fill.size ** 2, fill.name)

    def amount(k):
        return n[k.name] if k.name in n else k.count

    def most(k):
        return k.max if k.group == "extra" else (n_open if k.group == "fill" else k.count)

    # Packing only looks at sizes, so every piece of one size shares one set
    # of position variables (no duplicate variables per spot: proofs go much
    # faster). _assign then gives structures the spots furthest back.
    def cls(k):
        return "piece"

    one_cells = {(l.x, l.y) for l in locks if _size_of(kinds, l.kind) == 1}  # keep free for locked 1x1s
    classes = defaultdict(list)
    for k in kinds:
        classes[(cls(k), k.size)].append(k)

    v, cover = {}, defaultdict(list)
    for (c, s), ks in classes.items():
        if s == 1 or sum(most(k) for k in ks) == 0:
            continue
        pool = [p for p in spots[s] if not any(q in one_cells for q in square(*p, s))]
        v[(c, s)] = {p: m.NewBoolVar("") for p in pool}
        m.Add(sum(v[(c, s)].values()) == sum(amount(k) for k in ks))
        for p, var in v[(c, s)].items():
            for q in square(*p, s):
                cover[q].append(var)
    for vars_ in cover.values():
        if len(vars_) > 1:
            m.AddAtMostOne(vars_)
    for l in locks:
        k = next(k for k in kinds if k.name == l.kind)
        if k.size > 1:
            m.Add(v[(cls(k), k.size)][(l.x, l.y)] == 1)
    # 1x1 pieces fit in any free cell, so for them only the tile count matters.
    m.Add(sum(k.size ** 2 * amount(k) for k in kinds) <= n_open)

    if island.income_mode:
        if island.beds is not None:
            m.Add(sum(k.beds * n[k.name] for k in kinds if k.group == "extra") == int(island.beds))
        scale = _income_scale([k.income for k in kinds if k.group == "extra"])
        m.Maximize(sum(round(k.income * scale) * n[k.name] for k in kinds if k.group == "extra"))
    else:
        scale = 1
        m.Maximize(n[fill.name])

    solver, status, secs = _run(m, time_limit, workers)
    if status == cp_model.UNKNOWN:
        raise RuntimeError(f"{island.name}: ran out of time after {secs:.1f}s before finding any layout"
                           f" (this does not mean nothing fits). Give it more time: a bigger time_limit"
                           f" (MAX_SOLVE_SECS), or no limit (RUN_TO_OPTIMAL = True).")
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        why = f"{island.name}: no layout fits ({solver.StatusName(status)})."
        if island.income_mode and island.beds is not None and status == cp_model.INFEASIBLE:
            why += (f" Often no mix of extras uses exactly {island.beds} beds and still fits:"
                    f" check bed costs, min/max, or free space.")
        raise RuntimeError(why)

    for k in kinds:
        if k.name in n:
            k.count = int(solver.Value(n[k.name]))
    chosen = {key: [p for p, var in vm.items() if solver.Value(var)] for key, vm in v.items()}
    layout = _assign(grid, height, kinds, locks, chosen, cls)
    if island.income_mode:  # exact income from the counts (the model used scaled whole numbers)
        value = _tidy(sum(k.income * k.count for k in kinds if k.group == "extra"))
    else:
        value = int(round(solver.ObjectiveValue()))
    bound = _tidy(int(solver.BestObjectiveBound() + 1e-6) / scale)
    info = {"value": value, "bound": bound, "optimal": status == cp_model.OPTIMAL, "seconds": secs}
    return layout, info


def _income_scale(incomes) -> int:
    """Smallest power of ten that makes every income a whole number (CP-SAT needs integers)."""
    for scale in (1, 10, 100, 1000, 10_000, 100_000, 1_000_000):
        if all(abs(i * scale - round(i * scale)) < 1e-6 for i in incomes):
            return scale
    return 1_000_000


def _tidy(x):
    x = round(float(x), 6)
    return int(x) if x.is_integer() else x


# --------------------------------------------------------------------------- #
# 2. mirror
# --------------------------------------------------------------------------- #
def _mirror_step(grid, height, skinds, spools, slocks, structures, axis, spots, layout, left, workers, say):
    """
    The most structures with a same-size structure in their mirror spot.
    1. "Every structure mirrored" as a yes/no question, on the back-band spots
       (then on every spot if the band is proven impossible). If yes, that is
       optimal at once (usually seconds).
    2. Otherwise maximise the mirrored count on the back band (quick), then on
       every spot starting from that result; only this last search gives the
       bound, so "proven best" means best over the whole island.
    1x1 collection / extras / fill stay out of the model (any free cell will
    do; packing already made room), keeping their locked cells free.
    Returns dict(layout, value, bound, optimal, seconds); layout None = nothing found in time.
    """
    t0 = time.time()
    total = sum(k.count for k in structures)
    merged, m_of = _merge(skinds, monsters=True)
    mlocks = _rename_locks(slocks, m_of)
    ones = {(l.x, l.y) for l in mlocks if l.kind == "monsters 1x1"}
    mlocks = [l for l in mlocks if l.kind != "monsters 1x1"]
    kinds_m = [k for k in merged if k.group == "structure" or k.size > 1]

    def clear(pool, s):
        return [p for p in pool if not any(q in ones for q in square(*p, s))]

    full = {k.name: clear(spots[k.size], k.size) for k in kinds_m}
    band = {k.name: clear(spools[k.name], k.size) if k.group == "structure" else full[k.name] for k in kinds_m}
    hint = _rename(layout, m_of)

    def model(pools, symmetric, hint):
        m, v = _kind_model(grid, kinds_m, pools, mlocks)
        if symmetric:
            for k in structures:
                vm = v[k.name]
                for p, var in vm.items():
                    mp = axis.mirror_square(*p, k.size)
                    if mp == p:
                        continue
                    if mp in vm:
                        if p < mp:
                            m.Add(var == vm[mp])
                    else:
                        m.Add(var == 0)
        _hint(m, v, hint)
        return m, v

    def possible(pools):  # an odd count needs a spot on the axis
        return all(k.count % 2 == 0 or any(axis.mirror_square(*p, k.size) == p for p in pools[k.name])
                   for k in structures)

    def result(solver, v, value, bound, optimal):
        fixed = [p for p in _extract(solver, v, kinds_m, mlocks) if p.group == "structure"]
        chosen = {("monster", k.size): [p for p, var in v[k.name].items() if solver.Value(var)]
                  for k in kinds_m if k.group != "structure" and k.count}
        lay = _assign(grid, height, skinds, slocks, chosen, lambda k: "monster", preset=fixed)
        return dict(layout=lay, value=value, bound=bound, optimal=optimal, seconds=time.time() - t0)

    budget_end = None if left() is None else time.time() + left(0.5)  # this step's share of the time

    def share(f):
        return None if budget_end is None else max(1.0, f * (budget_end - time.time()))

    restricted = any(set(full[k.name]) != set(band[k.name]) for k in structures)
    stages = [("back-band", band)] + ([("all", full)] if restricted else [])

    # 1. "every structure mirrored?" (the "all" try only if the back band is proven impossible)
    for label, pools in stages:
        if not possible(pools):
            continue
        cap = SYMMETRIC_PASS_MAX if budget_end is None else min(SYMMETRIC_PASS_MAX, share(0.6))
        m, v = model(pools, True, hint)
        solver, status, secs = _run(m, cap, workers)
        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            say(f"every structure can be mirrored ({label} spots) | {secs:.1f}s")
            return result(solver, v, total, total, True)
        say(f"not every structure mirrored on {label} spots ({solver.StatusName(status).lower()}) | {secs:.1f}s")
        if status != cp_model.INFEASIBLE:
            break  # out of time for yes/no questions; maximise instead

    # 2. maximise: on the back band (quick), then on every spot starting from there
    progress = lambda obj, bound, t: say(f"  mirror: {int(round(obj))}/{total} so far"
                                         f" (bound {min(total, int(bound + 1e-6))}) at {t:.0f}s")
    best, bound = None, total
    for i, (label, pools) in enumerate(stages):
        last = i == len(stages) - 1
        m, v = model(pools, False, hint)
        m.Maximize(sum(_mirror_terms(m, v, structures, axis, hint=hint)))
        solver, status, secs = _run(m, share(1.0 if last else 0.5), workers, on_improve=progress)
        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            break
        value = int(round(solver.ObjectiveValue()))
        if last:  # only a search over every spot gives a true bound
            bound = int(solver.BestObjectiveBound() + 1e-6)
        best = (solver, v, value, last and status == cp_model.OPTIMAL)
        say(f"mirrored {value}/{total} on {label} spots | {secs:.1f}s")
        if value == total:
            break
        hint = _extract(solver, v, kinds_m, mlocks)
    if best is None:
        return dict(layout=None, seconds=time.time() - t0)
    solver, v, value, optimal = best
    return result(solver, v, value, max(value, bound), optimal or value == total)


def count_mirrored(placements, axis) -> int:
    """Structures with a same-size structure in their mirror spot (on the axis: itself)."""
    at = {(p.x, p.y, p.s) for p in placements if p.group == "structure"}
    return sum((*axis.mirror_square(x, y, s), s) in at for x, y, s in at)


def _merge(kinds, monsters: bool):
    """
    Structures of one size as a single kind; with monsters=True also all
    other kinds of one size (for the mirror step). Returns the new kinds and
    old name -> new name for the merged ones.
    """
    merged, new_of = [], {}
    for k in kinds:
        is_monster = k.group != "structure"
        if is_monster and not monsters:
            merged.append(k)
            continue
        name = f"monsters {k.size}x{k.size}" if is_monster else f"structures {k.size}x{k.size}"
        new_of[k.name] = name
        old = next((m for m in merged if m.name == name), None)
        if old is None:
            merged.append(Kind(name, "collection" if is_monster else "structure", k.size, k.count, back=not is_monster))
        else:
            old.count += k.count
    return merged, new_of


def _merged_pools(merged, new_of, pools):
    out = {k.name: set(pools[k.name]) for k in merged if k.name in pools}
    for old, new in new_of.items():
        out.setdefault(new, set()).update(pools[old])
    return {name: sorted(p) for name, p in out.items()}


def _rename(layout, new_of):
    return [Placement(p.x, p.y, p.s, new_of.get(p.kind, p.kind), p.group, p.locked) for p in layout]


def _rename_locks(locks, new_of):
    return [Lock(new_of.get(l.kind, l.kind), l.x, l.y) for l in locks]


def _name_structures(layout, kinds, locks, axis, workers):
    """
    Hand the structure names out over the structure spots (sizes fixed), so as
    many as possible have the same type in their mirror spot. Locked ones keep
    their name. Returns the named layout and how many structures have a
    same-type partner (one on the axis counts as its own partner).
    """
    out = [p for p in layout if p.group != "structure"]
    locked = {(l.x, l.y): l.kind for l in locks}
    same_type = 0
    for s in SIZES:
        spots = [(p.x, p.y) for p in layout if p.group == "structure" and p.s == s]
        types = [k for k in kinds if k.group == "structure" and k.size == s and k.count]
        if not spots:
            continue
        m = cp_model.CpModel()
        a = {(k.name, p): m.NewBoolVar("") for k in types for p in spots}
        for p in spots:
            m.AddExactlyOne(a[k.name, p] for k in types)
            if p in locked and any(k.name == locked[p] for k in types):
                m.Add(a[locked[p], p] == 1)
        for k in types:
            m.Add(sum(a[k.name, p] for p in spots) == k.count)
        terms, at = [], set(spots)
        for p in spots:
            mp = axis.mirror_square(*p, s)
            if mp == p:
                terms.append(1)
            elif mp in at and p < mp:
                for k in types:
                    both = m.NewBoolVar("")
                    m.AddImplication(both, a[k.name, p])
                    m.AddImplication(both, a[k.name, mp])
                    terms.append(2 * both)
        m.Maximize(sum(terms))
        solver, status, _ = _run(m, 30, workers)
        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            raise RuntimeError(f"could not name the {s}x{s} structures")
        same_type += int(round(solver.ObjectiveValue()))
        for (name, p), var in a.items():
            if solver.Value(var):
                out.append(Placement(p[0], p[1], s, name, "structure", locked.get(p) == name))
    return out, same_type


def _assign(grid, height, kinds, locks, chosen, cls, preset=()) -> list[Placement]:
    """
    Turn positions per size (`chosen`, keyed by (class, size)) into a complete
    layout with kinds: `preset` pieces as given, locked pieces next, then
    back-pulled kinds get the spots furthest back. Any 1x1 pieces not in
    `chosen` go into the free cells the same way.
    """
    locked_at = {(l.x, l.y): l.kind for l in locks}
    by_name = {k.name: k for k in kinds}
    left = {k.name: k.count for k in kinds}
    order = sorted(kinds, key=lambda k: (k.group != "structure", not k.back, k.group == "collection"))
    out = []

    def place(p, k):
        out.append(Placement(p[0], p[1], k.size, k.name, k.group, locked_at.get(p) == k.name))
        left[k.name] -= 1

    for p in preset:
        out.append(p)
        left[p.kind] -= 1
    for (c, s), positions in chosen.items():
        positions = sorted(positions, key=lambda p: height.depth2(p[0], p[1], s))
        free = []
        for p in positions:
            if p in locked_at and by_name[locked_at[p]].size == s and cls(by_name[locked_at[p]]) == c:
                place(p, by_name[locked_at[p]])
            else:
                free.append(p)
        for p in free:
            k = next(k for k in order if k.size == s and cls(k) == c and left[k.name] > 0)
            place(p, k)

    taken = np.zeros_like(grid)
    for p in out:
        for q in square(p.x, p.y, p.s):
            taken[q[1], q[0]] = True
    ys, xs = np.nonzero(grid & ~taken)
    cells = sorted(zip(xs.tolist(), ys.tolist()), key=lambda p: height.depth2(p[0], p[1], 1))
    for p in [p for p in cells if p in locked_at]:
        place(p, by_name[locked_at[p]])
    # Back-pulled 1x1s take the free cells furthest back, the rest (the
    # collection) the ones furthest forward, close to where arrange wants them.
    cells = deque(p for p in cells if p not in locked_at)
    for k in order:
        if k.size == 1:
            for _ in range(left[k.name]):
                place(cells.popleft() if k.back else cells.pop(), k)
    return out


# --------------------------------------------------------------------------- #
# 2-3. model with one set of placement variables per kind
# --------------------------------------------------------------------------- #
def _pools(kinds, spots, height, axis, layout, locks) -> dict[str, list[Pos]]:
    """Spots each kind may use in steps 2-3 (always includes step 1's layout)."""
    used = defaultdict(set)  # structure spots per size, so same-size types can swap
    for p in layout:
        if p.group == "structure":
            used[p.s].add((p.x, p.y))
    pools = {}
    for k in kinds:
        s, base = k.size, spots[k.size]
        if k.group == "structure":
            ok = set(base)
            pool = {p for p in base if height.frac(*p, s) <= STRUCTURE_BAND[s]} | used[s]
            pool |= {axis.mirror_square(*p, s) for p in used[s]} & ok
        else:
            pool = set(base)
        pool |= {(l.x, l.y) for l in locks if l.kind == k.name}
        pools[k.name] = sorted(pool)
    return pools


def _kind_model(grid, kinds, pools, locks):
    m = cp_model.CpModel()
    v = {k.name: {p: m.NewBoolVar("") for p in pools[k.name]} for k in kinds if k.count > 0}
    cover = defaultdict(list)
    for k in kinds:
        if k.count == 0:
            continue
        m.Add(sum(v[k.name].values()) == k.count)
        for p, var in v[k.name].items():
            for q in square(*p, k.size):
                cover[q].append(var)
    for vars_ in cover.values():
        if len(vars_) > 1:
            m.AddAtMostOne(vars_)
    for l in locks:
        m.Add(v[l.kind][(l.x, l.y)] == 1)
    return m, v


def _mirror_terms(m, v, structures, axis, hint=None):
    """
    Terms summing to the number of structures with a same-type partner in
    their mirror spot. hint: the layout the model is hinted with, so the
    helpers get matching hints and the solver starts from its full value.
    """
    placed = None if hint is None else {(p.kind, p.x, p.y) for p in hint}
    terms = []
    for k in structures:
        vm = v[k.name]
        for p, var in vm.items():
            mp = axis.mirror_square(*p, k.size)
            if mp == p:
                terms.append(var)  # sits on the axis: mirrors itself
            elif mp in vm and p < mp:  # one helper per pair, worth both structures
                both = m.NewBoolVar("")
                m.AddImplication(both, var)
                m.AddImplication(both, vm[mp])
                if placed is not None:
                    m.AddHint(both, (k.name, *p) in placed and (k.name, *mp) in placed)
                terms.append(2 * both)
    return terms


def _preference(k: Kind, height: Height, front: float, x: int, y: int) -> int:
    """
    How much a piece likes sitting at (x, y); the arrange step maximises the sum.
    Back-pulled pieces aim for the very back, structures and bigger pieces
    pulling hardest, so the order comes out structures > extras > collection.
    The collection spreads by size over the area in front of them.
    """
    s = k.size
    if k.back:
        weight, target = ((6 + s) * s * s if k.group == "structure" else s ** 3), 0.0
    else:
        weight, target = SIZE_WEIGHT[s], front + COLLECTION_TARGET[s] * (1.0 - front)
    return weight * (height.span2 - abs(height.depth2(x, y, s) - round(target * height.span2)))


def _hint(m, v, layout):
    placed = {(p.kind, p.x, p.y) for p in layout}
    for name, vm in v.items():
        for (x, y), var in vm.items():
            m.AddHint(var, (name, x, y) in placed)


def _extract(solver, v, kinds, locks) -> list[Placement]:
    by_name = {k.name: k for k in kinds}
    locked = {(l.kind, l.x, l.y) for l in locks}
    return [Placement(x, y, by_name[name].size, name, by_name[name].group, (name, x, y) in locked)
            for name, vm in v.items() for (x, y), var in vm.items() if solver.Value(var)]


class _Progress(cp_model.CpSolverSolutionCallback):
    """Remembers when the objective last improved by at least min_gain (relative)."""

    def __init__(self, min_gain=0.0, on_improve=None):
        super().__init__()
        self.best, self.last, self.min_gain, self.on_improve = None, time.time(), min_gain, on_improve

    def on_solution_callback(self):
        obj = self.ObjectiveValue()
        if self.best is None or obj > self.best + self.min_gain * max(1.0, abs(self.best)):
            self.best, self.last = obj, time.time()
            if self.on_improve:
                self.on_improve(obj, self.BestObjectiveBound(), self.WallTime())


def _run(m, time_limit, workers, gap=None, patience=None, min_gain=0.0, on_improve=None):
    """
    Solve a (maximisation) model. patience: stop once the objective has not
    improved by min_gain for that many seconds. on_improve(obj, bound, secs)
    is called on every improvement.
    """
    solver = cp_model.CpSolver()
    solver.parameters.num_workers = int(workers)
    if time_limit is not None:
        solver.parameters.max_time_in_seconds = float(time_limit)
    if gap:
        solver.parameters.relative_gap_limit = gap
    if patience is None and on_improve is None:
        status = solver.Solve(m)
        return solver, status, solver.WallTime()

    watch, done = _Progress(min_gain, on_improve), threading.Event()

    def watchdog():
        while not done.wait(0.25):
            if time.time() - watch.last > patience:
                solver.StopSearch()
                return

    if patience is not None:
        threading.Thread(target=watchdog, daemon=True).start()
    try:
        status = solver.Solve(m, watch)
    finally:
        done.set()
    return solver, status, solver.WallTime()


# --------------------------------------------------------------------------- #
# checks
# --------------------------------------------------------------------------- #
def _size_of(kinds, name) -> int:
    return next(k.size for k in kinds if k.name == name)


def _check_locks(island, grid, kinds, locks: list[Lock]):
    H, W = grid.shape
    taken = {}
    for l in locks:
        k = next(k for k in kinds if k.name == l.kind)
        cells = list(square(l.x, l.y, k.size))
        if any(not (0 <= x < W and 0 <= y < H and grid[y, x]) for x, y in cells):
            raise ValueError(f"{island}: locked {l.kind} doesn't fit on open tiles there.")
        for q in cells:
            if q in taken:
                raise ValueError(f"{island}: locked {l.kind} overlaps locked {taken[q]}.")
            taken[q] = l.kind
    for k in kinds:
        n = sum(l.kind == k.name for l in locks)
        if k.group == "extra":
            if n > k.max:
                raise ValueError(f"{island}: {n} {k.name} locked but at most {k.max} allowed.")
            k.min = max(k.min, n)
        elif n > k.count:
            raise ValueError(f"{island}: {n} {k.name} locked but the island only has {k.count}.")


def check(layout: Layout) -> list[str]:
    """Problems with a layout (empty list = sound): bounds, overlaps, sizes, counts, beds, locks."""
    grid, problems = layout.grid, []
    H, W = grid.shape
    seen = np.zeros_like(grid, dtype=int)
    count = defaultdict(int)
    for p in layout.placements:
        if not (0 <= p.x and 0 <= p.y and p.x + p.s <= W and p.y + p.s <= H):
            problems.append(f"{p.kind} outside the grid at {p.x},{p.y}")
            continue
        if not grid[p.y:p.y + p.s, p.x:p.x + p.s].all():
            problems.append(f"{p.kind} on a blocked tile at {p.x},{p.y}")
        seen[p.y:p.y + p.s, p.x:p.x + p.s] += 1
        count[p.kind] += 1
    if (seen > 1).any():
        problems.append(f"{int((seen > 1).sum())} tile(s) covered twice")
    known = {k.name: k for k in layout.kinds}
    for p in layout.placements:
        if p.kind not in known:
            problems.append(f"{p.kind} at {p.x},{p.y} is not a piece of this island")
        elif p.s != known[p.kind].size:
            problems.append(f"{p.kind} at {p.x},{p.y} is {p.s}x{p.s}, should be {known[p.kind].size}x{known[p.kind].size}")
    for k in layout.kinds:
        if count[k.name] != k.count:
            problems.append(f"{k.name}: {count[k.name]} placed, expected {k.count}")
        if k.group == "extra" and not k.min <= k.count <= k.max:
            problems.append(f"{k.name}: {k.count} placed, allowed {k.min}-{k.max}")
    isl = layout.island
    if isl.income_mode and isl.beds is not None and layout.beds_used != int(isl.beds):
        problems.append(f"extras use {layout.beds_used} beds, expected exactly {isl.beds}")
    placed = {(p.kind, p.x, p.y) for p in layout.placements}
    for l in layout.island.locks(layout.kinds):
        if (l.kind, l.x, l.y) not in placed:
            problems.append(f"locked {l.kind} not at its cell")
    return problems
