"""
MSM island layout optimizer.

Packs a fixed collection of structures and monsters (1x1 .. 4x4) onto an island
grid read from a CSV mask, then fills the leftover space with as many NxN
"fill" squares as possible, while pulling each piece toward its preferred
vertical band (big pieces high, small pieces low; structures biased to the top).

Solved lexicographically with CP-SAT:
    Phase 1  maximise the number of fill squares.
    (1b)     optional: freeze that count, maximise mirrored structure pairs.
    Phase 2  freeze the above, then maximise the vertical-gradient score.

This replaces the old single blended objective, whose primary weight
(~1e8) x score (~2e5) produced objective values near 1e12. Those coefficients
wrecked CP-SAT's LP relaxation and made the reported optimality gap
meaningless. The two-phase form is faster and its gap is a real bound on the
number of fill squares.
"""

from __future__ import annotations

import csv
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime

import numpy as np
from ortools.sat.python import cp_model

__all__ = [
    "Placement",
    "DISPLAY_COLORS",
    "load_avail_from_csv",
    "find_mirror_axis",
    "mirror_top_left",
    "solve_full_collection_with_fill",
    "validate_solution",
    "draw_solution",
    "extra_colors",
    "export_solution_to_excel",
]

SIZES = (1, 2, 3, 4)
# "back" = income-mode extras flagged back=True (e.g. Wubbox): placed like
# monsters, but pulled hard to the in-game back so they hide nothing.
GROUPS = ("structure", "monster", "back")

DISPLAY_COLORS = {
    "blocked": "#2B2B2B",
    "empty": "#FFFFFF",
    "structure": "#4C78A8",
    "monster": "#E45756",
    "extra": "#54A24B",
}

# Where each piece "wants" to sit, as a fraction of the island's height
# (0 = in-game back, 1 = front). Structures and extras go to the very back;
# the collection is spread out by size.
TARGET_RATIOS = {
    "structure": {4: 0.0, 3: 0.0, 2: 0.0, 1: 0.0},
    "monster": {4: 0.10, 3: 0.32, 2: 0.66, 1: 0.92},
    "extra": {4: 0.10, 3: 0.32, 2: 0.66, 1: 0.92},
    "back": {4: 0.0, 3: 0.0, 2: 0.0, 1: 0.0},
}
SIZE_WEIGHT = {4: 6, 3: 4, 2: 2, 1: 1}


def piece_weight(group: str, s: int) -> int:
    """
    How hard a piece is pulled toward its target height.

    Back-to-front order: structures, then "back" extras, then the collection.
    Two pieces trade places when that lowers the weighted depth, so what
    decides the order is the weight *per tile*:
      structure  7..10 per tile (bigger = deeper)
      back       1..4  per tile (bigger = deeper, so nothing small hides
                                 behind a Wubbox)
      collection <= 1  per tile (spread over the island by TARGET_RATIOS)
    """
    if group == "structure":
        return (6 + s) * s * s
    if group == "back":
        return s * s * s
    return SIZE_WEIGHT[s]


# Hard vertical bands used only to shrink the candidate pool for 2x2..4x4.
# They are a speed heuristic, not a requirement: if the model comes back
# infeasible the bands are widened automatically and finally dropped.
# 1x1 pieces are deliberately NOT banded (see module docstring / README notes).
BAND_DEFAULTS = {
    "structure": {4: (0.00, 0.40), 3: (0.00, 0.50), 2: (0.00, 0.60)},
    "monster": {4: (0.00, 0.75), 3: (0.00, 1.00), 2: (0.00, 1.00)},
}


# --------------------------------------------------------------------------- #
# logging
# --------------------------------------------------------------------------- #
def log_print(enabled: bool, msg: str) -> None:
    if enabled:
        print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


@dataclass(frozen=True, slots=True)
class Placement:
    x: int
    y: int
    s: int
    kind: str
    group: str


# --------------------------------------------------------------------------- #
# CSV loading
# --------------------------------------------------------------------------- #
_TRUE = {".", "1", "true", "t", "yes", "y", "o", "avail", "available", "open"}
_FALSE = {"#", "0", "false", "f", "no", "n", "", "-", "x", "na", "n/a", "blocked"}


def load_avail_from_csv(path: str, log: bool = False) -> np.ndarray:
    """Read a grid mask. Returns a bool array, True = usable cell."""
    t0 = time.time()
    rows: list[list[str]] = []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        for raw in csv.reader(f):
            cells = [str(v).strip() for v in raw]
            if any(cells):  # skip blank lines
                rows.append(cells)
    if not rows:
        raise ValueError(f"{path}: CSV is empty.")

    W = max(len(r) for r in rows)
    H = len(rows)
    avail = np.zeros((H, W), dtype=bool)
    for y, row in enumerate(rows):
        for x, value in enumerate(row):
            v = value.lower()
            if v in _TRUE:
                avail[y, x] = True
            elif v not in _FALSE:
                # Report *where*, so a stray character is findable in a 33x33 grid.
                raise ValueError(
                    f"{path}: unrecognised cell {value!r} at row {y + 1}, column {x + 1}. "
                    f"Use '.' for open and '#' for blocked."
                )
        # short rows pad as blocked, which np.zeros already did

    log_print(log, f"Loaded {W}x{H} from {path} | available={int(avail.sum())} | {time.time() - t0:.2f}s")
    return avail


# --------------------------------------------------------------------------- #
# geometry helpers
# --------------------------------------------------------------------------- #
def valid_top_left(avail: np.ndarray, s: int) -> list[tuple[int, int]]:
    """Every (x, y) where an s x s square fits entirely on available cells."""
    H, W = avail.shape
    if s < 1 or H < s or W < s:
        return []
    if s == 1:
        ys, xs = np.nonzero(avail)
    else:
        blocked = (~avail).astype(np.int32)
        sat = np.pad(blocked, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
        bad = sat[s:, s:] - sat[:-s, s:] - sat[s:, :-s] + sat[:-s, :-s]
        ys, xs = np.nonzero(bad == 0)
    return list(zip(xs.tolist(), ys.tolist()))


# --------------------------------------------------------------------------- #
# mirror symmetry (the in-game left/right mirror of the island)
# --------------------------------------------------------------------------- #
# MSM draws the grid as a diamond, so the on-screen vertical axis of the island
# is a *diagonal* of the CSV grid, not its middle column. Mirroring left/right
# on screen is a reflection across that diagonal:
#     "anti": axis x + y = c   ->  cell (x, y) maps to (c - y, c - x)
#     "main": axis x - y = c   ->  cell (x, y) maps to (y + c, x - c)
# Which diagonal (and which c) depends on how the CSV was drawn, so it is
# detected from the island mask: the axis that mirrors the most open cells
# onto open cells wins.
def _mirror_cell(x: int, y: int, axis: dict) -> tuple[int, int]:
    c = axis["c"]
    if axis["kind"] == "anti":
        return c - y, c - x
    return y + c, x - c


def mirror_top_left(x: int, y: int, s: int, axis: dict) -> tuple[int, int]:
    """Top-left corner of the mirror image of the s x s square at (x, y)."""
    ax, ay = _mirror_cell(x, y, axis)
    bx, by = _mirror_cell(x + s - 1, y + s - 1, axis)
    return min(ax, bx), min(ay, by)


def find_mirror_axis(avail: np.ndarray) -> dict:
    """
    Best-fitting in-game mirror axis for this island mask.
    Returns {"kind": "anti"|"main", "c": int, "matched": int, "total": int}.
    """
    H, W = avail.shape
    ys, xs = np.nonzero(avail)
    cells = set(zip(xs.tolist(), ys.tolist()))
    best = None
    candidates = [("anti", c) for c in range(0, H + W)] + [("main", c) for c in range(-H, W + 1)]
    for kind, c in candidates:
        axis = {"kind": kind, "c": c}
        matched = sum(_mirror_cell(x, y, axis) in cells for x, y in cells)
        if best is None or matched > best["matched"]:
            best = {"kind": kind, "c": c, "matched": matched, "total": len(cells)}
    return best


class Height:
    """
    How "high" a piece sits, used by the arrangement step and the bands.

    axis=None   -> grid rows: row 0 is the top (the original behaviour).
    axis=dict   -> in-game height: distance along the mirror axis, the
                   diagonal that is vertical on screen. A piece and its mirror
                   image get the same height, so arranging by height never
                   fights the structure symmetry.
                   By default "up" is the end of the axis nearer grid row 0;
                   flip=True uses the other end.

    Everything is integer, in half-cell units (centres of even-sized pieces
    land on half cells, so doubling keeps the arithmetic exact).
    """

    def __init__(self, avail: np.ndarray, axis: dict | None = None, flip: bool = False):
        H, _W = avail.shape
        self.axis, self.flip = axis, flip
        sign = -1 if flip else 1
        if axis is None:
            self._raw = lambda x, y, s: sign * (2 * y + (s - 1))
            lo, hi = sorted((0, sign * 2 * (H - 1)))
        else:
            if axis["kind"] == "anti":
                # axis runs along x + y = c; height changes with x - y, and the
                # end nearer row 0 is the one with larger x - y.
                self._raw = lambda x, y, s: sign * 2 * (y - x)
            else:
                # axis runs along x - y = c; height changes with x + y.
                self._raw = lambda x, y, s: sign * (2 * (x + y) + 2 * (s - 1))
            ys, xs = np.nonzero(avail)
            vals = [self._raw(x, y, 1) for x, y in zip(xs.tolist(), ys.tolist())]
            lo, hi = (min(vals), max(vals)) if vals else (0, 1)
        self.lo = lo
        self.span2 = max(1, hi - lo)

    def depth2(self, x: int, y: int, s: int) -> int:
        """0 at the top, span2 at the bottom."""
        return self._raw(x, y, s) - self.lo

    def frac(self, x: int, y: int, s: int) -> float:
        return self.depth2(x, y, s) / self.span2

    # Where the collection's own area starts (fraction of the height). The
    # back of the island goes to structures and extras, so the collection's
    # size-spread targets are squeezed into [front_start, 1]. 0 = whole island.
    front_start = 0.0

    def set_front_start(self, avail: np.ndarray, cells_behind: int) -> None:
        """Collection area starts after the first `cells_behind` tiles from the back."""
        if cells_behind <= 0:
            self.front_start = 0.0
            return
        ys, xs = np.nonzero(avail)
        d = sorted(self.depth2(x, y, 1) for x, y in zip(xs.tolist(), ys.tolist()))
        self.front_start = d[min(len(d) - 1, cells_behind)] / self.span2

    def score(self, group: str, s: int, x: int, y: int) -> int:
        """Preference for an s x s piece of `group` at (x, y); higher is better."""
        ratio = TARGET_RATIOS[group][s]
        if group in ("monster", "extra"):
            ratio = self.front_start + ratio * (1.0 - self.front_start)
        target2 = int(round(ratio * self.span2))
        return piece_weight(group, s) * (self.span2 - abs(self.depth2(x, y, s) - target2))

    def top_cell(self, avail: np.ndarray) -> tuple[int, int]:
        """The open cell at the very top (for drawing the 'up' arrow)."""
        ys, xs = np.nonzero(avail)
        cells = list(zip(xs.tolist(), ys.tolist()))
        best = min(self.depth2(x, y, 1) for x, y in cells)
        top = [p for p in cells if self.depth2(p[0], p[1], 1) == best]
        if self.axis is not None:  # the one closest to the axis line
            c = self.axis["c"]
            off = (lambda p: abs(p[0] + p[1] - c)) if self.axis["kind"] == "anti" else (lambda p: abs(p[0] - p[1] - c))
            return min(top, key=off)
        return top[len(top) // 2]

    def up_vector(self) -> tuple[float, float]:
        """Unit direction (dx, dy) on the grid that points in-game up."""
        if self.axis is None:
            v = (0.0, -1.0)
        elif self.axis["kind"] == "anti":
            v = (0.7071, -0.7071)
        else:
            v = (-0.7071, -0.7071)
        return (-v[0], -v[1]) if self.flip else v


def _greedy_disjoint_count(coords, s: int, H: int, W: int, cap: int) -> int:
    """
    How many pairwise-disjoint s x s squares can be taken from `coords`
    (greedy row-major, so a lower bound on the true maximum). Stops at `cap`.

    Used to reject a band that *contains* enough candidate positions but cannot
    actually hold enough non-overlapping pieces -- the cause of the old
    spurious "INFEASIBLE" results.
    """
    if cap <= 0:
        return 0
    used = np.zeros((H, W), dtype=bool)
    n = 0
    for x, y in sorted(coords, key=lambda p: (p[1], p[0])):
        if used[y : y + s, x : x + s].any():
            continue
        used[y : y + s, x : x + s] = True
        n += 1
        if n >= cap:
            break
    return n


def _band_filter(coords, s: int, height: Height, lo: float, hi: float):
    eps = 1e-9
    return [(x, y) for x, y in coords if lo - eps <= height.frac(x, y, s) <= hi + eps]


def _band_candidates(coords, *, s, H, W, height, required, lo, hi, log, label):
    """Widen the band until it can genuinely hold `required` disjoint pieces."""
    if required <= 0:
        return []
    step = 0.06
    while True:
        filtered = _band_filter(coords, s, height, lo, hi)
        full = lo <= 0.0 and hi >= 1.0
        if _greedy_disjoint_count(filtered, s, H, W, required) >= required:
            log_print(log, f"   {label}: {len(filtered)} candidates in band [{lo:.2f}, {hi:.2f}]")
            return filtered
        if full:
            log_print(log, f"   {label}: using full height, {len(filtered)} candidates")
            return filtered
        lo = max(0.0, lo - step)
        hi = min(1.0, hi + step)


# --------------------------------------------------------------------------- #
# model construction
# --------------------------------------------------------------------------- #
def _build_model(avail, all_pos, pools, fill_pos, fill_size, sc, mc, *, include_ones,
                 model=None, monster_count=None, bc=None):
    """
    Shared model skeleton.

    include_ones=False (phase 1) drops the ~2 x |available| 1x1 booleans and
    every constraint they touch. That is exact, not an approximation: a 1x1
    piece fits in any free cell, so the only thing 1x1 counts do to the packing
    is consume cells, which a single capacity bound already expresses.

    monster_count (income mode, phase 1 only): {size: expression} giving the
    number of monsters of each size as solver variables instead of constants.
    `mc` then holds the maximum per size, used only to decide which sizes
    need placement variables.
    """
    H, W = avail.shape
    if model is None:
        model = cp_model.CpModel()

    bc = bc or {s: 0 for s in SIZES}  # "back" pieces share the monster spots
    mand: dict[tuple[str, int], dict[tuple[int, int], object]] = {}
    for group, counts in (("structure", sc), ("monster", mc), ("back", bc)):
        for s in (4, 3, 2):
            if counts[s] == 0:
                continue
            pool = pools.get((group, s)) or pools[("monster", s)]
            v_map = {p: model.NewBoolVar(f"{group[0]}{s}_{p[0]}_{p[1]}") for p in pool}
            mand[(group, s)] = v_map
            if group == "monster" and monster_count is not None:
                model.Add(sum(v_map.values()) == monster_count[s])
            else:
                model.Add(sum(v_map.values()) == counts[s])

    fill_vars = {p: model.NewBoolVar(f"f{fill_size}_{p[0]}_{p[1]}") for p in fill_pos}

    one_vars: dict[str, dict[tuple[int, int], object]] = {g: {} for g in GROUPS}
    if include_ones:
        for group, cnt in (("structure", sc[1]), ("monster", mc[1]), ("back", bc[1])):
            if cnt <= 0:
                continue
            v_map = {p: model.NewBoolVar(f"{group[0]}1_{p[0]}_{p[1]}") for p in all_pos}
            one_vars[group] = v_map
            model.Add(sum(v_map.values()) == cnt)

    # Non-overlap. Flat list-of-lists beats a dict keyed by (y, x) tuples, and
    # AddAtMostOne is a native CP-SAT constraint rather than a generic sum <= 1.
    cover: list[list] = [[] for _ in range(H * W)]
    for (_group, s), v_map in mand.items():
        for (tx, ty), v in v_map.items():
            for dy in range(s):
                base = (ty + dy) * W + tx
                for dx in range(s):
                    cover[base + dx].append(v)
    for (tx, ty), v in fill_vars.items():
        for dy in range(fill_size):
            base = (ty + dy) * W + tx
            for dx in range(fill_size):
                cover[base + dx].append(v)

    for x, y in all_pos:
        terms = cover[y * W + x]
        for g in GROUPS:
            v = one_vars[g].get((x, y))
            if v is not None:
                terms.append(v)
        if len(terms) > 1:
            model.AddAtMostOne(terms)

    return model, mand, one_vars, fill_vars


def _symmetry_terms(model, mand, one_vars, axis):
    """
    One 0/1 term per structure piece that is mirrored by another structure of
    the same size (a piece sitting on the axis mirrors itself). Monsters and
    fill squares are ignored. Maximising the sum = most symmetric structures.
    """
    terms = []
    maps = [v_map for (group, _s), v_map in mand.items() if group == "structure"]
    sizes = [s for (group, s) in mand if group == "structure"]
    if one_vars.get("structure"):
        maps.append(one_vars["structure"])
        sizes.append(1)
    for s, v_map in zip(sizes, maps):
        for (x, y), v in v_map.items():
            m = mirror_top_left(x, y, s, axis)
            if m == (x, y):
                terms.append(v)  # on the axis: symmetric by itself
                continue
            mv = v_map.get(m)
            if mv is None:
                continue  # mirror spot is blocked or out of the pool
            b = model.NewBoolVar(f"sym{s}_{x}_{y}")
            model.AddImplication(b, v)
            model.AddImplication(b, mv)
            terms.append(b)
    return terms


def _parse_catalog(catalog, avail_cells: int) -> list[dict]:
    """
    Normalise the optional-monster catalog. Each entry:
        {"name": str, "size": 1..4, "income": number, "beds": int = 1,
         "min": int = 0, "max": int | None, "back": bool = True}
    `income` is per monster, in whatever unit you like (coins/hour, per day...),
    as long as every entry uses the same one. `beds` is how many beds one of
    that monster takes. max=None means "as many as fit". back=True pulls the
    extra to the in-game back of the island so it doesn't hide the
    collection (arrangement only - never changes income).
    """
    out, seen = [], set()
    for i, e in enumerate(catalog):
        name = str(e.get("name") or f"monster{i + 1}")
        if name in seen:
            raise ValueError(f"monster_catalog: duplicate name {name!r}.")
        seen.add(name)
        s = int(e["size"])
        if s not in SIZES:
            raise ValueError(f"monster_catalog[{name}]: size must be one of {list(SIZES)}, got {s}.")
        lo = int(e.get("min", 0))
        hi = e.get("max")
        hi = avail_cells // (s * s) if hi is None else int(hi)
        if lo < 0 or hi < lo:
            raise ValueError(f"monster_catalog[{name}]: need 0 <= min <= max, got min={lo}, max={hi}.")
        beds = int(e.get("beds", 1))
        if beds < 0:
            raise ValueError(f"monster_catalog[{name}]: beds must be >= 0, got {beds}.")
        out.append({"name": name, "size": s, "income": int(round(float(e.get("income", 0)))),
                    "beds": beds, "min": lo, "max": hi, "back": bool(e.get("back", True))})
    return out


def _solve(model, *, workers, solver_log, time_limit, gap=None):
    solver = cp_model.CpSolver()
    solver.parameters.num_workers = int(workers)
    solver.parameters.log_search_progress = bool(solver_log)
    if time_limit is not None and time_limit > 0:
        solver.parameters.max_time_in_seconds = float(time_limit)
    if gap:
        solver.parameters.relative_gap_limit = float(gap)
    status = solver.Solve(model)
    return solver, status


def solve_full_collection_with_fill(
    avail: np.ndarray,
    structures: dict,
    monsters: dict,
    *,
    log: bool = True,
    solver_log: bool = False,
    workers: int = 8,
    max_time_in_seconds: float | None = None,
    fill_size: int = 2,
    fill_leftovers: bool = False,
    use_bands: bool = True,
    structure_symmetry: bool = False,
    symmetry_axis: dict | None = None,
    monster_catalog: list[dict] | None = None,
    gradient_gap: float = 0.0,
    gradient_time_limit: float | None = None,
    remaining_beds: int | None = None,
    height_mode: str = "game",
    flip_up: bool = False,
    collection_spread: bool = True,
) -> dict:
    """
    Returns a dict. On success: ok=True plus `placements` (list[Placement]),
    `fill_count`, `fill_upper_bound`, `optimal`, timings and the input counts.
    On failure: ok=False plus `reason`.

    structure_symmetry=True adds a step between the two phases: with the fill
    count frozen at its phase-1 maximum, make as many structures as possible
    mirror images of each other across the in-game axis. It can never cost a
    fill square, because the fill count is fixed before symmetry is
    considered. Monsters are left free. `symmetry_axis` overrides the
    auto-detected axis ({"kind": "anti"|"main", "c": int}).

    monster_catalog (income mode): optional extra monsters the solver may add
    on top of the fixed `monsters` counts, each with an income (see
    _parse_catalog). Phase 1 then maximises total income of the extras
    instead of the number of fill squares; the chosen mix is in
    `monster_mix` and the result's `monsters` holds the final per-size
    counts (fixed + extras). Fill squares are not used in this mode.

    remaining_beds (income mode): the extras must use exactly this many beds
    in total (sum of each extra's `beds`). Can also be given as a "beds" key
    in the `monsters` dict. Ignored when there is no catalog.

    height_mode: what "higher on the island" means for the arrangement step
    (big pieces and structures high, small monsters low). "game" (default)
    measures along the in-game vertical axis, so it agrees with structure
    symmetry; "rows" uses grid rows like the original version. flip_up=True
    swaps which end of the island counts as the top.

    collection_spread=True spreads the collection by size (4x4 at the back of
    its area, 1x1 at the front) over only the part of the island left after
    structures and back extras. False spreads it over the whole island.

    gradient_gap / gradient_time_limit: stop the last (cosmetic) phase once
    it is proven within this fraction of the best arrangement, or after this
    many seconds. Neither touches fill, income or symmetry, which are frozen
    by then.
    """
    t_start = time.time()
    H, W = avail.shape
    avail_cells = int(avail.sum())

    if fill_size not in SIZES:
        # Used to sail through and blow up much later with a KeyError inside the
        # Excel colour table.
        raise ValueError(f"fill_size must be one of {list(SIZES)}, got {fill_size!r}.")

    sc = {s: int(structures.get(s, 0)) for s in SIZES}
    mc = {s: int(monsters.get(s, 0)) for s in SIZES}
    for label, counts in (("structures", sc), ("monsters", mc)):
        for s, c in counts.items():
            if c < 0:
                raise ValueError(f"{label}[{s}] must be >= 0, got {c}.")

    # Income mode: mc_pool = most monsters of each size that could be placed,
    # used to build placement variables; mc stays the fixed part until
    # phase 1 has chosen the extras.
    catalog = _parse_catalog(monster_catalog, avail_cells) if monster_catalog else None
    if remaining_beds is None:
        remaining_beds = monsters.get("beds")
    if not catalog:
        remaining_beds = None  # beds only constrain the optional extras
    if remaining_beds is not None:
        remaining_beds = int(remaining_beds)
        if remaining_beds < 0:
            raise ValueError(f"remaining beds must be >= 0, got {remaining_beds}.")
        for e in catalog:  # no monster can use more beds than are left
            if e["beds"] > 0:
                e["max"] = min(e["max"], remaining_beds // e["beds"])
            if e["min"] > e["max"]:
                return {"ok": False, "reason": (
                    f"{e['name']}: min {e['min']} needs {e['min'] * e['beds']} beds, "
                    f"only {remaining_beds} remaining.")}
    mc_fixed = dict(mc)
    mc_pool = dict(mc)
    if catalog:
        for e in catalog:
            mc_pool[e["size"]] += e["max"]
        for s in SIZES:
            mc_pool[s] = min(mc_pool[s], avail_cells // (s * s))
        mc_min = {s: mc[s] + sum(e["min"] for e in catalog if e["size"] == s) for s in SIZES}
    else:
        mc_min = mc

    req = {s: sc[s] + mc[s] for s in SIZES}
    needed = sum((sc[s] + mc_min[s]) * s * s for s in SIZES)
    if needed > avail_cells:
        return {
            "ok": False,
            "reason": f"Required cells ({needed}) exceed available cells ({avail_cells}).",
        }

    log_print(log, f"Grid {W}x{H} | avail={avail_cells} | structures={sc} | monsters={mc}"
                   + (f" + {len(catalog)} optional monster types" if catalog else ""))

    # ---- candidate positions, computed once per size and reused ------------ #
    raw: dict[int, list] = {}
    for s in (2, 3, 4):
        if sc[s] + mc_pool[s] > 0:
            raw[s] = valid_top_left(avail, s)
    if catalog:
        # An optional size that fits nowhere is simply not used.
        for s in (2, 3, 4):
            if mc_pool[s] > mc_fixed[s] and not raw.get(s):
                if mc_min[s] > mc_fixed[s]:
                    return {"ok": False, "reason": f"No {s}x{s} square fits anywhere, but the catalog requires some."}
                mc_pool[s] = mc_fixed[s]
                for e in catalog:
                    if e["size"] == s:
                        e["max"] = 0
    if fill_size not in raw:
        raw[fill_size] = valid_top_left(avail, fill_size)  # was recomputed every run
    all_pos = valid_top_left(avail, 1)
    fill_pos = [] if catalog else raw[fill_size]

    fill_cap = 0 if catalog else max(0, (avail_cells - needed) // (fill_size * fill_size))

    if height_mode not in ("game", "rows"):
        raise ValueError(f"height_mode must be 'game' or 'rows', got {height_mode!r}.")
    game_axis = None
    if structure_symmetry or height_mode == "game":
        game_axis = dict(symmetry_axis) if symmetry_axis else find_mirror_axis(avail)
        if "matched" in game_axis:
            log_print(
                log,
                f"Mirror axis: {game_axis['kind']} diagonal c={game_axis['c']} "
                f"({game_axis['matched']}/{game_axis['total']} open cells have an open mirror)",
            )
    axis = game_axis if structure_symmetry else None  # None = symmetry step off
    height = Height(avail, game_axis if height_mode == "game" else None, flip=flip_up)

    # ---- band pools, with automatic widening on infeasibility -------------- #
    def make_pools(widen: float):
        pools = {}
        for group, counts in (("structure", sc), ("monster", mc_pool)):
            for s in (4, 3, 2):
                if counts[s] == 0:
                    continue
                coords = raw.get(s, [])
                if not coords:
                    return None, f"No {s}x{s} square fits anywhere on this grid ({group})."
                # In income mode the monster count is not known yet, so a
                # band sized for it can't be built - monsters get every spot.
                if use_bands and widen < 1.0 and not (catalog and group == "monster"):
                    lo, hi = BAND_DEFAULTS[group][s]
                    lo, hi = max(0.0, lo - widen), min(1.0, hi + widen)
                    pool = _band_candidates(
                        coords, s=s, H=H, W=W, height=height, required=counts[s],
                        lo=lo, hi=hi, log=log, label=f"{group} {s}x{s}",
                    )
                else:
                    pool = coords
                if axis is not None and group == "structure":
                    # With height_mode="rows" bands cut by grid row, and the
                    # mirror of a spot can sit in a different row - add the
                    # mirror of every spot so a band never removes the partner
                    # of a symmetric pair. (In "game" mode this adds nothing.)
                    valid = set(coords)
                    have = set(pool)
                    pool = list(pool) + [
                        m for m in {mirror_top_left(x, y, s, axis) for x, y in pool}
                        if m in valid and m not in have
                    ]
                if not pool:
                    return None, f"No placement for {group} {s}x{s}."
                pools[(group, s)] = pool
        return pools, None

    # ---- phase 1: maximise fill count ------------------------------------- #
    share1 = 0.4 if axis is not None else 0.6
    budget1 = None if max_time_in_seconds is None else max(1.0, share1 * max_time_in_seconds)
    # Phase 1 always searches the whole island. Bands only speed up the
    # later, cosmetic phases: limiting phase 1 to them could cost fill or
    # income (seen in testing: up to ~2% income lost when "up" flipped).
    phase1 = None
    pools, err = make_pools(1.0)
    if pools is None:
        return {"ok": False, "reason": err}
    if catalog:
        model = cp_model.CpModel()
        n_vars = {e["name"]: model.NewIntVar(e["min"], e["max"], f"n_{i}") for i, e in enumerate(catalog)}
        m_count = {
            s: mc_fixed[s] + sum(n_vars[e["name"]] for e in catalog if e["size"] == s)
            for s in SIZES
        }
        model, mand, _, fill_vars = _build_model(
            avail, all_pos, pools, fill_pos, fill_size, sc, mc_pool, include_ones=False,
            model=model, monster_count=m_count,
        )
        # 1x1s have no placement variables here, so cap total cells used.
        model.Add(sum(s * s * (sc[s] + m_count[s]) for s in SIZES) <= avail_cells)
        if remaining_beds is not None:
            model.Add(sum(e["beds"] * n_vars[e["name"]] for e in catalog) == remaining_beds)
        model.Maximize(sum(e["income"] * n_vars[e["name"]] for e in catalog))
    else:
        n_vars = None
        model, mand, _, fill_vars = _build_model(
            avail, all_pos, pools, fill_pos, fill_size, sc, mc, include_ones=False
        )
        model.Add(sum(fill_vars.values()) <= fill_cap)
        model.Maximize(sum(fill_vars.values()))
    solver, status = _solve(model, workers=workers, solver_log=solver_log, time_limit=budget1)
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        phase1 = (pools, mand, fill_vars, solver, status, n_vars)
    if phase1 is None:
        reason = (
            f"No layout exists for these counts on this grid "
            f"(solver returned {solver.StatusName(status)})."
        )
        if remaining_beds is not None and status == cp_model.INFEASIBLE:
            reason += (
                f" With income mode on, the usual cause is that no mix of extras uses exactly "
                f"{remaining_beds} beds and still fits - check the bed costs, min/max limits, or "
                f"whether the island has room."
            )
        return {"ok": False, "status": solver.StatusName(status), "reason": reason}

    pools, mand1, fill_vars1, solver1, status1, n_vars1 = phase1
    t_phase1 = solver1.WallTime()
    monster_mix = income = income_bound = None
    if catalog:
        # Freeze the chosen mix: from here on it is an ordinary fixed-count
        # layout, so the symmetry and arrangement phases run unchanged.
        monster_mix = {name: int(solver1.Value(v)) for name, v in n_vars1.items()}
        mc = dict(mc_fixed)
        for e in catalog:
            mc[e["size"]] += monster_mix[e["name"]]
        req = {s: sc[s] + mc[s] for s in SIZES}
        income = int(round(solver1.ObjectiveValue()))
        income_bound = int(np.floor(solver1.BestObjectiveBound() + 1e-6))
        best_fill = fill_bound = 0
        log_print(
            log,
            f"Phase 1: income={income:,} (bound {income_bound:,}) | mix={monster_mix} "
            f"| {solver1.StatusName(status1)} | {t_phase1:.2f}s",
        )
    else:
        best_fill = int(round(solver1.ObjectiveValue()))
        fill_bound = int(np.floor(solver1.BestObjectiveBound() + 1e-6))
        log_print(
            log,
            f"Phase 1: fill_{fill_size}x{fill_size}={best_fill} "
            f"(bound {fill_bound}, cap {fill_cap}) | {solver1.StatusName(status1)} | {t_phase1:.2f}s",
        )

    hint = {
        "mand": {k: {p: solver1.Value(v) for p, v in vm.items()} for k, vm in mand1.items()},
        "fill": {p: solver1.Value(v) for p, v in fill_vars1.items()},
    }

    # Later phases: narrow the candidate spots to the bands for speed, but
    # always keep every spot phase 1 used (and, for structures, its mirror),
    # so phase 1's layout - and with it the frozen fill / income - stays
    # reachable. Only the symmetry and arrangement can be limited by bands.
    if use_bands:
        band_pools, _err = make_pools(0.0)
        if band_pools:
            for key, vm in mand1.items():
                used_spots = {p for p, v in vm.items() if hint["mand"][key][p]}
                if axis is not None and key[0] == "structure":
                    full = set(pools[key])
                    used_spots |= {m for m in (mirror_top_left(x, y, key[1], axis) for x, y in used_spots)
                                   if m in full}
                pools[key] = sorted(set(band_pools.get(key, [])) | used_spots)

    # Split the "back" extras out of the monsters. Phase 1 only cared about
    # sizes; from here the model keeps them as their own group so the
    # arrangement can push them to the back. Hint: give them the spots
    # furthest back in phase 1's layout (any split of those spots is valid).
    bc = {s: 0 for s in SIZES}
    if catalog:
        for e in catalog:
            if e["back"]:
                bc[e["size"]] += monster_mix[e["name"]]
                mc[e["size"]] -= monster_mix[e["name"]]
    if collection_spread:
        height.set_front_start(avail, sum(s * s * (sc[s] + bc[s]) for s in SIZES))
        log_print(log, f"Collection area starts at {height.front_start:.0%} of the island height")
        for s in (4, 3, 2):
            key = ("monster", s)
            if bc[s] == 0 or key not in hint["mand"]:
                continue
            chosen = sorted((p for p, val in hint["mand"][key].items() if val),
                            key=lambda p: height.depth2(p[0], p[1], s))
            back_spots = set(chosen[: bc[s]])
            pools[("back", s)] = pools[key]
            hint["mand"][("back", s)] = {p: int(p in back_spots) for p in hint["mand"][key]}
            hint["mand"][key] = {p: int(val and p not in back_spots) for p, val in hint["mand"][key].items()}

    # ---- phase 1b: freeze the fill count, maximise structure symmetry ------ #
    sym_count = sym_bound = None
    symmetry_optimal = False
    t_phase_sym = 0.0
    phase_sym = None  # (solver, mand, one_vars, fill_vars) if it succeeded
    n_struct = sum(sc.values())
    if axis is not None and n_struct > 0:
        used = time.time() - t_start
        budget_s = None if max_time_in_seconds is None else max(1.0, 0.5 * (max_time_in_seconds - used))
        model, mand_s, one_s, fill_s = _build_model(
            avail, all_pos, pools, fill_pos, fill_size, sc, mc, include_ones=True, bc=bc
        )
        if fill_s:
            model.Add(sum(fill_s.values()) == best_fill)
        sym_terms = _symmetry_terms(model, mand_s, one_s, axis)
        if sym_terms:
            model.Maximize(sum(sym_terms))
            for key, v_map in mand_s.items():
                for p, v in v_map.items():
                    model.AddHint(v, hint["mand"].get(key, {}).get(p, 0))
            for p, v in fill_s.items():
                model.AddHint(v, hint["fill"][p])
            solver_s, status_s = _solve(model, workers=workers, solver_log=solver_log, time_limit=budget_s)
            t_phase_sym = solver_s.WallTime()
            if status_s in (cp_model.OPTIMAL, cp_model.FEASIBLE):
                sym_count = int(round(solver_s.ObjectiveValue()))
                sym_bound = int(np.floor(solver_s.BestObjectiveBound() + 1e-6))
                symmetry_optimal = status_s == cp_model.OPTIMAL
                phase_sym = (solver_s, mand_s, one_s, fill_s)
                hint = {
                    "mand": {k: {p: solver_s.Value(v) for p, v in vm.items()} for k, vm in mand_s.items()},
                    "fill": {p: solver_s.Value(v) for p, v in fill_s.items()},
                    "ones": {g: {p: solver_s.Value(v) for p, v in vm.items()} for g, vm in one_s.items()},
                }
            log_print(
                log,
                f"Symmetry: {sym_count}/{n_struct} structures mirrored "
                f"(bound {sym_bound}) | {solver_s.StatusName(status_s)} | {t_phase_sym:.2f}s",
            )
        else:
            sym_count, sym_bound, symmetry_optimal = 0, 0, True
            log_print(log, "Symmetry: no structure spot has an open mirror spot on this island.")

    # ---- phase 2: freeze the fill count, maximise the gradient ------------- #
    used = time.time() - t_start
    budget2 = None if max_time_in_seconds is None else max(1.0, max_time_in_seconds - used)
    if gradient_time_limit:
        budget2 = gradient_time_limit if budget2 is None else min(budget2, gradient_time_limit)

    model, mand, one_vars, fill_vars = _build_model(
        avail, all_pos, pools, fill_pos, fill_size, sc, mc, include_ones=True, bc=bc
    )
    if fill_vars:
        model.Add(sum(fill_vars.values()) == best_fill)
    if sym_count:
        model.Add(sum(_symmetry_terms(model, mand, one_vars, axis)) >= sym_count)

    terms = []
    for (group, s), v_map in mand.items():
        for (x, y), v in v_map.items():
            terms.append(height.score(group, s, x, y) * v)
    for group in GROUPS:
        for (x, y), v in one_vars[group].items():
            terms.append(height.score(group, 1, x, y) * v)
    for (x, y), v in fill_vars.items():
        terms.append(height.score("extra", fill_size, x, y) * v)
    model.Maximize(sum(terms))

    # Warm start from phase 1; the 1x1 pieces are left for the solver to place.
    for key, v_map in mand.items():
        for p, v in v_map.items():
            model.AddHint(v, hint["mand"].get(key, {}).get(p, 0))
    for p, v in fill_vars.items():
        model.AddHint(v, hint["fill"][p])
    for group, v_map in hint.get("ones", {}).items():
        for p, val in v_map.items():
            if p in one_vars[group]:
                model.AddHint(one_vars[group][p], val)

    solver2, status2 = _solve(model, workers=workers, solver_log=solver_log, time_limit=budget2,
                              gap=gradient_gap)

    gradient_gap_reached = None
    if status2 in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        placements = _extract(solver2, mand, one_vars, fill_vars, fill_size)
        # With a gap limit CP-SAT reports OPTIMAL once inside the gap, so
        # measure the real distance to the bound instead of trusting the label.
        obj, bnd = solver2.ObjectiveValue(), solver2.BestObjectiveBound()
        gradient_gap_reached = max(0.0, (bnd - obj) / max(1.0, abs(obj)))
        gradient_optimal = status2 == cp_model.OPTIMAL and gradient_gap_reached < 1e-9
        t_phase2 = solver2.WallTime()
        status_name = solver2.StatusName(status2)
    elif phase_sym is not None:
        # The symmetry step already produced a complete, legal layout.
        log_print(log, f"Phase 2 returned {solver2.StatusName(status2)} - keeping the symmetry-step layout.")
        solver_s, mand_s, one_s, fill_s = phase_sym
        placements = _extract(solver_s, mand_s, one_s, fill_s, fill_size)
        gradient_optimal = False
        t_phase2 = solver2.WallTime()
        status_name = solver_s.StatusName(status_s)
    else:
        # Safety net: phase 1's layout is always feasible here, so fall back to
        # it and drop the 1x1 pieces into free cells by preference order.
        log_print(log, f"Phase 2 returned {solver2.StatusName(status2)} - falling back to phase 1 layout.")
        placements = _extract(solver1, mand1, {"structure": {}, "monster": {}}, fill_vars1, fill_size)
        placements += _greedy_ones(avail, placements, sc[1], mc[1] + bc[1], height)
        gradient_optimal = False
        t_phase2 = solver2.WallTime()
        status_name = solver1.StatusName(status1)

    if catalog:
        placements = _name_monsters(placements, catalog, monster_mix, height)

    occupied = np.zeros_like(avail, dtype=bool)
    for p in placements:
        occupied[p.y : p.y + p.s, p.x : p.x + p.s] = True

    if fill_leftovers and not catalog:  # income mode: leftover cells just stay empty
        for x, y in all_pos:
            if not occupied[y, x]:
                placements.append(Placement(x, y, 1, "extra_1x1", "extra"))
                occupied[y, x] = True

    fill_optimal = status1 == cp_model.OPTIMAL
    log_print(
        log,
        f"Done | {'income=' + format(income, ',') if catalog else 'fill=' + str(best_fill)} "
        f"| gradient {'optimal' if gradient_optimal else 'best effort'} "
        f"| total {time.time() - t_start:.2f}s",
    )

    return {
        "ok": True,
        "status": status_name,
        "optimal": fill_optimal and gradient_optimal,
        "fill_optimal": fill_optimal,  # in income mode: income is proven maximal
        "gradient_optimal": gradient_optimal,
        "gradient_gap": gradient_gap_reached,
        "income": income,
        "income_upper_bound": income_bound,
        "monster_mix": monster_mix,
        "remaining_beds": remaining_beds,
        "beds_used": (sum(e["beds"] * monster_mix[e["name"]] for e in catalog) if catalog else None),
        "W": W,
        "H": H,
        "available_cells": avail_cells,
        "structures": sc,
        "monsters": {s: mc[s] + bc[s] for s in SIZES},
        "required_totals": req,
        "fill_size": fill_size,
        "fill_count": best_fill,
        "fill_upper_bound": fill_bound,
        "fill_capacity": fill_cap,
        "free_cells": int(avail_cells - occupied.sum()),
        "phase1_seconds": t_phase1,
        "phase2_seconds": t_phase2,
        "symmetry_seconds": t_phase_sym,
        "symmetry_axis": axis,
        "height_mode": height_mode,
        "game_axis": game_axis,
        "collection_start": height.front_start,
        "top_cell": height.top_cell(avail) if height_mode == "game" else None,
        "up_vector": height.up_vector() if height_mode == "game" else None,
        "symmetric_structures": sym_count,
        "symmetric_structures_bound": sym_bound,
        "total_structures": n_struct,
        "symmetry_optimal": symmetry_optimal,
        "solve_wall_time_seconds": t_phase1 + t_phase_sym + t_phase2,
        "total_seconds": time.time() - t_start,
        "placements": placements,
    }


def _extract(solver, mand, one_vars, fill_vars, fill_size) -> list[Placement]:
    out: list[Placement] = []
    for (group, s), v_map in mand.items():
        for (x, y), v in v_map.items():
            if solver.Value(v):
                out.append(Placement(x, y, s, f"{group}_{s}x{s}", group))
    for group in GROUPS:
        for (x, y), v in one_vars.get(group, {}).items():
            if solver.Value(v):
                out.append(Placement(x, y, 1, f"{group}_1x1", group))
    for (x, y), v in fill_vars.items():
        if solver.Value(v):
            out.append(Placement(x, y, fill_size, f"extra_{fill_size}x{fill_size}", "extra"))
    return out


def _name_monsters(placements, catalog, monster_mix, height) -> list[Placement]:
    """
    Label monster pieces with catalog names (pieces are interchangeable within
    a size). "back" pieces get the back=True extras; other extras go on the
    furthest-back remaining spots; the fixed collection keeps the generic
    label. Every piece comes out as group "monster".
    """
    out = [p for p in placements if p.group not in ("monster", "back")]
    for s in SIZES:
        mine = sorted((p for p in placements if p.group in ("monster", "back") and p.s == s),
                      key=lambda p: (p.group != "back", height.depth2(p.x, p.y, s), p.y, p.x))
        names = []
        for e in sorted((e for e in catalog if e["size"] == s), key=lambda e: (not e["back"], -e["income"])):
            names += [e["name"]] * monster_mix[e["name"]]
        for i, p in enumerate(mine):
            kind = f"monster_{names[i]}" if i < len(names) else f"monster_{s}x{s}"
            out.append(Placement(p.x, p.y, p.s, kind, "monster"))
    return out


def _greedy_ones(avail, placements, n_struct, n_monster, height) -> list[Placement]:
    """Fallback 1x1 placement: best remaining cell by preference, structures first."""
    occ = np.zeros_like(avail, dtype=bool)
    for p in placements:
        occ[p.y : p.y + p.s, p.x : p.x + p.s] = True
    free = [(x, y) for y in range(avail.shape[0]) for x in range(avail.shape[1]) if avail[y, x] and not occ[y, x]]
    out = []
    for group, n in (("structure", n_struct), ("monster", n_monster)):
        free.sort(key=lambda p: -height.score(group, 1, p[0], p[1]))
        for x, y in free[:n]:
            out.append(Placement(x, y, 1, f"{group}_1x1", group))
        free = free[n:]
    return out


# --------------------------------------------------------------------------- #
# validation
# --------------------------------------------------------------------------- #
def validate_solution(avail: np.ndarray, sol: dict, structures: dict, monsters: dict) -> list[str]:
    """
    Check the solution really is a legal packing. The old notebook only
    asserted the piece counts; nothing verified overlaps or blocked cells.
    Returns a list of problems (empty means the layout is sound).
    """
    problems: list[str] = []
    H, W = avail.shape
    occ = np.zeros((H, W), dtype=int)
    tally = {"structure": Counter(), "monster": Counter(), "extra": Counter()}

    for p in sol["placements"]:
        if p.x < 0 or p.y < 0 or p.x + p.s > W or p.y + p.s > H:
            problems.append(f"piece outside the grid: {p}")
            continue
        if not avail[p.y : p.y + p.s, p.x : p.x + p.s].all():
            problems.append(f"piece on a blocked cell: {p}")
        occ[p.y : p.y + p.s, p.x : p.x + p.s] += 1
        tally[p.group][p.s] += 1

    overlaps = int((occ > 1).sum())
    if overlaps:
        problems.append(f"{overlaps} cell(s) covered by more than one piece")

    for label, want in (("structure", structures), ("monster", monsters)):
        expected = {s: int(want.get(s, 0)) for s in SIZES}
        got = {s: int(tally[label][s]) for s in SIZES}
        if got != expected:
            problems.append(f"{label} counts {got} != requested {expected}")

    return problems


# --------------------------------------------------------------------------- #
# output
# --------------------------------------------------------------------------- #
# Colours for the named extras in income mode, in catalog order. Chosen to
# stay clear of the structure blue, monster red and blocked grey.
EXTRA_PALETTE = ("#9467BD", "#F58518", "#17BECF", "#EECA3B", "#FF9DA6", "#8C564B", "#B5BD61", "#7F7F7F")


def _extra_name(p) -> str | None:
    """Catalog name of an income-mode extra monster, else None."""
    if p.group == "monster" and p.kind != f"monster_{p.s}x{p.s}":
        return p.kind.removeprefix("monster_")
    return None


def extra_colors(sol: dict) -> dict[str, str]:
    """{extra name: hex colour}, one distinct colour per catalog entry."""
    names = list((sol.get("monster_mix") or {}).keys())
    return {n: EXTRA_PALETTE[i % len(EXTRA_PALETTE)] for i, n in enumerate(names)}


def draw_solution(avail: np.ndarray, sol: dict, ax=None, show: bool = True):
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    H, W = avail.shape
    fs = sol["fill_size"]
    if ax is None:
        _fig, ax = plt.subplots(figsize=(max(12, W / 2.2), max(10, H / 2.2)))

    bg = np.where(avail[:, :, None], [1.0, 1.0, 1.0], [43 / 255, 43 / 255, 43 / 255])
    ax.imshow(bg, interpolation="none", origin="upper")

    colors = extra_colors(sol)
    for p in sol["placements"]:
        name = _extra_name(p)
        ax.add_patch(
            mpatches.Rectangle(
                (p.x - 0.5, p.y - 0.5), p.s, p.s,
                facecolor=colors.get(name, DISPLAY_COLORS[p.group]), edgecolor="#111111",
                linewidth=1.2, alpha=0.82,
            )
        )
        if p.s >= 2:
            ax.text(
                p.x + (p.s - 1) / 2, p.y + (p.s - 1) / 2, str(p.s),
                ha="center", va="center", fontsize=8 + p.s,
                color="white", fontweight="bold",
            )

    if sol.get("monster_mix") is not None:
        mix = ", ".join(f"{n} x{c}" for n, c in sol["monster_mix"].items() if c)
        title = (
            f"income: {sol['income']:,}  (upper bound {sol['income_upper_bound']:,})   extras: {mix or 'none'}"
        )
    else:
        title = (
            f"blue = structure   red = monster   green = extra {fs}x{fs}\n"
            f"extra {fs}x{fs}: {sol['fill_count']}  (upper bound {sol['fill_upper_bound']})"
        )
    axis = sol.get("symmetry_axis")
    if axis:
        c = axis["c"]
        xs = np.array([-0.5, W - 0.5])
        ys = (c - xs) if axis["kind"] == "anti" else (xs - c)
        ax.plot(xs, ys, linestyle="--", color="#F2B701", linewidth=2.2, zorder=5)
        if sol.get("symmetric_structures") is not None:
            title += (f"\nmirrored structures: {sol['symmetric_structures']}/{sol['total_structures']}"
                      f"  (yellow dashed line = in-game mirror axis)")
    top = sol.get("top_cell")
    if top:
        # Arrow along the in-game vertical, ending at the island's top cell.
        dx, dy = sol["up_vector"]
        tail = (top[0] - 7 * dx, top[1] - 7 * dy)
        ax.annotate("", xy=top, xytext=tail, zorder=6,
                    arrowprops=dict(arrowstyle="-|>", color="#F2B701", lw=3))
        ax.text(tail[0], tail[1], "in-game up", color="#F2B701", fontsize=11, fontweight="bold",
                ha="center", va="center", zorder=6,
                bbox=dict(facecolor="#2B2B2B", edgecolor="none", alpha=0.75, pad=2))
    ax.set_title(title, pad=16)
    ax.set_xlim(-0.5, W - 0.5)
    ax.set_ylim(H - 0.5, -0.5)
    ax.set_xticks(np.arange(-0.5, W, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, H, 1), minor=True)
    ax.grid(which="minor", color="#DDDDDD", linewidth=0.35)
    ax.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)
    handles = [mpatches.Patch(color=DISPLAY_COLORS["structure"], label="Structure"),
               mpatches.Patch(color=DISPLAY_COLORS["monster"], label="Monster")]
    if sol.get("monster_mix") is not None:
        handles += [mpatches.Patch(color=colors[n], label=f"{n} (extra x{c})")
                    for n, c in sol["monster_mix"].items() if c]
    else:
        handles.append(mpatches.Patch(color=DISPLAY_COLORS["extra"], label="Extra"))
    handles.append(mpatches.Patch(color=DISPLAY_COLORS["blocked"], label="Blocked"))
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.04),
              ncol=min(len(handles), 6), frameon=False)
    if show:
        plt.tight_layout()
        plt.show()
    return ax


_EXCEL_COLORS = {
    ("blocked", -1): "FF2B2B2B", ("empty", 0): "FFFFFFFF",
    ("structure", 1): "FFDCE8F5", ("structure", 2): "FFA9C4E2",
    ("structure", 3): "FF6F9FD1", ("structure", 4): "FF3E79B2",
    ("monster", 1): "FFF8D6D6", ("monster", 2): "FFF2A7A7",
    ("monster", 3): "FFE46F6F", ("monster", 4): "FFD64545",
    ("extra", 1): "FFD9F0D2", ("extra", 2): "FFA9D99B",
    ("extra", 3): "FF78C266", ("extra", 4): "FF4AA64A",
}


def export_solution_to_excel(avail: np.ndarray, sol: dict, out_path: str = "solution.xlsx") -> str:
    try:
        import openpyxl
        from openpyxl.styles import Font, PatternFill
        from openpyxl.utils import get_column_letter
    except ImportError as exc:
        raise ImportError("openpyxl is required for the Excel export: pip install openpyxl") from exc

    H, W = avail.shape
    size_layer = np.where(avail, 0, -1).astype(int)
    group_layer = np.where(avail, "empty", "blocked").astype(object)
    for p in sol["placements"]:
        name = _extra_name(p)
        size_layer[p.y : p.y + p.s, p.x : p.x + p.s] = p.s
        group_layer[p.y : p.y + p.s, p.x : p.x + p.s] = f"name:{name}" if name else p.group

    # One PatternFill per distinct colour instead of one per cell.
    fills = {key: PatternFill(start_color=c, end_color=c, fill_type="solid")
             for key, c in _EXCEL_COLORS.items()}
    for name, hexc in extra_colors(sol).items():  # extras: one colour each, any size
        argb = "FF" + hexc.lstrip("#").upper()
        for sz in SIZES:
            fills[(f"name:{name}", sz)] = PatternFill(start_color=argb, end_color=argb, fill_type="solid")

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "solution"
    for y in range(H):
        ws.row_dimensions[y + 1].height = 18
        for x in range(W):
            ws.cell(row=y + 1, column=x + 1).fill = fills[(group_layer[y, x], int(size_layer[y, x]))]
    # Name tags for income-mode extras (generic pieces stay unlabeled).
    for p in sol["placements"]:
        name = _extra_name(p)
        if name:
            cell = ws.cell(row=p.y + 1, column=p.x + 1)
            cell.value = name
            cell.font = Font(size=7)
    for x in range(W):
        ws.column_dimensions[get_column_letter(x + 1)].width = 4
    wb.save(out_path)
    return out_path
