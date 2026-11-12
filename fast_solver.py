"""
MSM island layout optimizer.

Packs a fixed collection of structures and monsters (1x1 .. 4x4) onto an island
grid read from a CSV mask, then fills the leftover space with as many NxN
"fill" squares as possible, while pulling each piece toward its preferred
vertical band (big pieces high, small pieces low; structures biased to the top).

Solved lexicographically with CP-SAT:
    Phase 1  maximise the number of fill squares.
    Phase 2  freeze that count, then maximise the vertical-gradient score.

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
    "solve_full_collection_with_fill",
    "validate_solution",
    "draw_solution",
    "export_solution_to_excel",
]

SIZES = (1, 2, 3, 4)
GROUPS = ("structure", "monster")

DISPLAY_COLORS = {
    "blocked": "#2B2B2B",
    "empty": "#FFFFFF",
    "structure": "#4C78A8",
    "monster": "#E45756",
    "extra": "#54A24B",
}

# Where each piece "wants" to sit, as a fraction of grid height.
TARGET_RATIOS = {
    "structure": {4: 0.05, 3: 0.12, 2: 0.20, 1: 0.28},
    "monster": {4: 0.10, 3: 0.32, 2: 0.66, 1: 0.92},
    "extra": {4: 0.10, 3: 0.32, 2: 0.66, 1: 0.92},
}
SIZE_WEIGHT = {4: 6, 3: 4, 2: 2, 1: 1}

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


def gradient_score(group: str, s: int, y: int, H: int) -> int:
    """
    Preference score for putting an s x s piece of `group` at row y.

    Integer arithmetic in half-cell units. The old version multiplied by 1000
    for no reason: centres land on half-integers, so doubling is already exact,
    and the x1000 inflated every objective coefficient ~500x.
    """
    span = max(1, H - 1)
    centre2 = 2 * y + (s - 1)
    target2 = int(round(2 * TARGET_RATIOS[group][s] * span))
    return SIZE_WEIGHT[s] * (2 * span - abs(centre2 - target2))


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


def _band_filter(coords, s: int, H: int, lo: float, hi: float):
    if H <= 1:
        return list(coords)
    span = H - 1
    lo_y, hi_y = lo * span, hi * span
    return [(x, y) for x, y in coords if lo_y <= y + (s - 1) / 2 <= hi_y]


def _band_candidates(coords, *, s, H, W, required, lo, hi, log, label):
    """Widen the band until it can genuinely hold `required` disjoint pieces."""
    if required <= 0:
        return []
    step = 0.06
    while True:
        filtered = _band_filter(coords, s, H, lo, hi)
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
def _build_model(avail, all_pos, pools, fill_pos, fill_size, sc, mc, *, include_ones):
    """
    Shared model skeleton.

    include_ones=False (phase 1) drops the ~2 x |available| 1x1 booleans and
    every constraint they touch. That is exact, not an approximation: a 1x1
    piece fits in any free cell, so the only thing 1x1 counts do to the packing
    is consume cells, which a single capacity bound already expresses.
    """
    H, W = avail.shape
    model = cp_model.CpModel()

    mand: dict[tuple[str, int], dict[tuple[int, int], object]] = {}
    for group, counts in (("structure", sc), ("monster", mc)):
        for s in (4, 3, 2):
            if counts[s] == 0:
                continue
            pool = pools[(group, s)]
            v_map = {p: model.NewBoolVar(f"{group[0]}{s}_{p[0]}_{p[1]}") for p in pool}
            mand[(group, s)] = v_map
            model.Add(sum(v_map.values()) == counts[s])

    fill_vars = {p: model.NewBoolVar(f"f{fill_size}_{p[0]}_{p[1]}") for p in fill_pos}

    one_vars: dict[str, dict[tuple[int, int], object]] = {"structure": {}, "monster": {}}
    if include_ones:
        for group, cnt in (("structure", sc[1]), ("monster", mc[1])):
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
        sv = one_vars["structure"].get((x, y))
        mv = one_vars["monster"].get((x, y))
        if sv is not None:
            terms.append(sv)
        if mv is not None:
            terms.append(mv)
        if len(terms) > 1:
            model.AddAtMostOne(terms)

    return model, mand, one_vars, fill_vars


def _solve(model, *, workers, solver_log, time_limit):
    solver = cp_model.CpSolver()
    solver.parameters.num_workers = int(workers)
    solver.parameters.log_search_progress = bool(solver_log)
    if time_limit is not None and time_limit > 0:
        solver.parameters.max_time_in_seconds = float(time_limit)
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
) -> dict:
    """
    Returns a dict. On success: ok=True plus `placements` (list[Placement]),
    `fill_count`, `fill_upper_bound`, `optimal`, timings and the input counts.
    On failure: ok=False plus `reason`.
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
    req = {s: sc[s] + mc[s] for s in SIZES}
    for label, counts in (("structures", sc), ("monsters", mc)):
        for s, c in counts.items():
            if c < 0:
                raise ValueError(f"{label}[{s}] must be >= 0, got {c}.")

    big_cells = sum(req[s] * s * s for s in (2, 3, 4))
    needed = big_cells + req[1]
    if needed > avail_cells:
        return {
            "ok": False,
            "reason": f"Required cells ({needed}) exceed available cells ({avail_cells}).",
        }

    log_print(log, f"Grid {W}x{H} | avail={avail_cells} | structures={sc} | monsters={mc}")

    # ---- candidate positions, computed once per size and reused ------------ #
    raw: dict[int, list] = {}
    for s in (2, 3, 4):
        if req[s] > 0:
            raw[s] = valid_top_left(avail, s)
    if fill_size not in raw:
        raw[fill_size] = valid_top_left(avail, fill_size)  # was recomputed every run
    all_pos = valid_top_left(avail, 1)
    fill_pos = raw[fill_size]

    fill_cap = max(0, (avail_cells - needed) // (fill_size * fill_size))

    # ---- band pools, with automatic widening on infeasibility -------------- #
    def make_pools(widen: float):
        pools = {}
        for group, counts in (("structure", sc), ("monster", mc)):
            for s in (4, 3, 2):
                if counts[s] == 0:
                    continue
                coords = raw.get(s, [])
                if not coords:
                    return None, f"No {s}x{s} square fits anywhere on this grid ({group})."
                if use_bands and widen < 1.0:
                    lo, hi = BAND_DEFAULTS[group][s]
                    lo, hi = max(0.0, lo - widen), min(1.0, hi + widen)
                    pool = _band_candidates(
                        coords, s=s, H=H, W=W, required=counts[s],
                        lo=lo, hi=hi, log=log, label=f"{group} {s}x{s}",
                    )
                else:
                    pool = coords
                if not pool:
                    return None, f"No placement for {group} {s}x{s}."
                pools[(group, s)] = pool
        return pools, None

    # ---- phase 1: maximise fill count ------------------------------------- #
    budget1 = None if max_time_in_seconds is None else max(1.0, 0.6 * max_time_in_seconds)
    phase1 = None
    for attempt, widen in enumerate((0.0, 0.15, 1.0)):
        pools, err = make_pools(widen)
        if pools is None:
            return {"ok": False, "reason": err}
        model, mand, _, fill_vars = _build_model(
            avail, all_pos, pools, fill_pos, fill_size, sc, mc, include_ones=False
        )
        model.Add(sum(fill_vars.values()) <= fill_cap)
        model.Maximize(sum(fill_vars.values()))
        solver, status = _solve(model, workers=workers, solver_log=solver_log, time_limit=budget1)
        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            phase1 = (pools, mand, fill_vars, solver, status)
            break
        if attempt == 0:
            log_print(log, f"Phase 1 {solver.StatusName(status)} with default bands - widening.")
        elif attempt == 1:
            log_print(log, "Phase 1 still infeasible - dropping bands entirely.")
    if phase1 is None:
        return {
            "ok": False,
            "status": solver.StatusName(status),
            "reason": (
                f"No layout exists for these counts on this grid "
                f"(solver returned {solver.StatusName(status)} even with bands disabled)."
            ),
        }

    pools, mand1, fill_vars1, solver1, status1 = phase1
    best_fill = int(round(solver1.ObjectiveValue()))
    fill_bound = int(np.floor(solver1.BestObjectiveBound() + 1e-6))
    t_phase1 = solver1.WallTime()
    log_print(
        log,
        f"Phase 1: fill_{fill_size}x{fill_size}={best_fill} "
        f"(bound {fill_bound}, cap {fill_cap}) | {solver1.StatusName(status1)} | {t_phase1:.2f}s",
    )

    hint = {
        "mand": {k: {p: solver1.Value(v) for p, v in vm.items()} for k, vm in mand1.items()},
        "fill": {p: solver1.Value(v) for p, v in fill_vars1.items()},
    }

    # ---- phase 2: freeze the fill count, maximise the gradient ------------- #
    used = time.time() - t_start
    budget2 = None if max_time_in_seconds is None else max(1.0, max_time_in_seconds - used)

    model, mand, one_vars, fill_vars = _build_model(
        avail, all_pos, pools, fill_pos, fill_size, sc, mc, include_ones=True
    )
    model.Add(sum(fill_vars.values()) == best_fill)

    terms = []
    for (group, s), v_map in mand.items():
        for (_x, y), v in v_map.items():
            terms.append(gradient_score(group, s, y, H) * v)
    for group in GROUPS:
        for (_x, y), v in one_vars[group].items():
            terms.append(gradient_score(group, 1, y, H) * v)
    for (_x, y), v in fill_vars.items():
        terms.append(gradient_score("extra", fill_size, y, H) * v)
    model.Maximize(sum(terms))

    # Warm start from phase 1; the 1x1 pieces are left for the solver to place.
    for key, v_map in mand.items():
        for p, v in v_map.items():
            model.AddHint(v, hint["mand"][key][p])
    for p, v in fill_vars.items():
        model.AddHint(v, hint["fill"][p])

    solver2, status2 = _solve(model, workers=workers, solver_log=solver_log, time_limit=budget2)

    if status2 in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        placements = _extract(solver2, mand, one_vars, fill_vars, fill_size)
        gradient_optimal = status2 == cp_model.OPTIMAL
        t_phase2 = solver2.WallTime()
        status_name = solver2.StatusName(status2)
    else:
        # Safety net: phase 1's layout is always feasible here, so fall back to
        # it and drop the 1x1 pieces into free cells by preference order.
        log_print(log, f"Phase 2 returned {solver2.StatusName(status2)} - falling back to phase 1 layout.")
        placements = _extract(solver1, mand1, {"structure": {}, "monster": {}}, fill_vars1, fill_size)
        placements += _greedy_ones(avail, placements, sc[1], mc[1], H)
        gradient_optimal = False
        t_phase2 = solver2.WallTime()
        status_name = solver1.StatusName(status1)

    occupied = np.zeros_like(avail, dtype=bool)
    for p in placements:
        occupied[p.y : p.y + p.s, p.x : p.x + p.s] = True

    if fill_leftovers:
        for x, y in all_pos:
            if not occupied[y, x]:
                placements.append(Placement(x, y, 1, "extra_1x1", "extra"))
                occupied[y, x] = True

    fill_optimal = status1 == cp_model.OPTIMAL
    log_print(
        log,
        f"Done | fill={best_fill} | gradient {'optimal' if gradient_optimal else 'best effort'} "
        f"| total {time.time() - t_start:.2f}s",
    )

    return {
        "ok": True,
        "status": status_name,
        "optimal": fill_optimal and gradient_optimal,
        "fill_optimal": fill_optimal,
        "gradient_optimal": gradient_optimal,
        "W": W,
        "H": H,
        "available_cells": avail_cells,
        "structures": sc,
        "monsters": mc,
        "required_totals": req,
        "fill_size": fill_size,
        "fill_count": best_fill,
        "fill_upper_bound": fill_bound,
        "fill_capacity": fill_cap,
        "free_cells": int(avail_cells - occupied.sum()),
        "phase1_seconds": t_phase1,
        "phase2_seconds": t_phase2,
        "solve_wall_time_seconds": t_phase1 + t_phase2,
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


def _greedy_ones(avail, placements, n_struct, n_monster, H) -> list[Placement]:
    """Fallback 1x1 placement: best remaining cell by preference, structures first."""
    occ = np.zeros_like(avail, dtype=bool)
    for p in placements:
        occ[p.y : p.y + p.s, p.x : p.x + p.s] = True
    free = [(x, y) for y in range(avail.shape[0]) for x in range(avail.shape[1]) if avail[y, x] and not occ[y, x]]
    out = []
    for group, n in (("structure", n_struct), ("monster", n_monster)):
        free.sort(key=lambda p: -gradient_score(group, 1, p[1], H))
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
def draw_solution(avail: np.ndarray, sol: dict, ax=None, show: bool = True):
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    H, W = avail.shape
    fs = sol["fill_size"]
    if ax is None:
        _fig, ax = plt.subplots(figsize=(max(12, W / 2.2), max(10, H / 2.2)))

    bg = np.where(avail[:, :, None], [1.0, 1.0, 1.0], [43 / 255, 43 / 255, 43 / 255])
    ax.imshow(bg, interpolation="none", origin="upper")

    for p in sol["placements"]:
        ax.add_patch(
            mpatches.Rectangle(
                (p.x - 0.5, p.y - 0.5), p.s, p.s,
                facecolor=DISPLAY_COLORS[p.group], edgecolor="#111111",
                linewidth=1.2, alpha=0.82,
            )
        )
        if p.s >= 2:
            ax.text(
                p.x + (p.s - 1) / 2, p.y + (p.s - 1) / 2, str(p.s),
                ha="center", va="center", fontsize=8 + p.s,
                color="white", fontweight="bold",
            )

    ax.set_title(
        f"blue = structure   red = monster   green = extra {fs}x{fs}\n"
        f"extra {fs}x{fs}: {sol['fill_count']}  (upper bound {sol['fill_upper_bound']})",
        pad=16,
    )
    ax.set_xlim(-0.5, W - 0.5)
    ax.set_ylim(H - 0.5, -0.5)
    ax.set_xticks(np.arange(-0.5, W, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, H, 1), minor=True)
    ax.grid(which="minor", color="#DDDDDD", linewidth=0.35)
    ax.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)
    ax.legend(
        handles=[mpatches.Patch(color=DISPLAY_COLORS[g], label=g.capitalize())
                 for g in ("structure", "monster", "extra", "blocked")],
        loc="upper center", bbox_to_anchor=(0.5, -0.04), ncol=4, frameon=False,
    )
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
        from openpyxl.styles import PatternFill
        from openpyxl.utils import get_column_letter
    except ImportError as exc:
        raise ImportError("openpyxl is required for the Excel export: pip install openpyxl") from exc

    H, W = avail.shape
    size_layer = np.where(avail, 0, -1).astype(int)
    group_layer = np.where(avail, "empty", "blocked").astype(object)
    for p in sol["placements"]:
        size_layer[p.y : p.y + p.s, p.x : p.x + p.s] = p.s
        group_layer[p.y : p.y + p.s, p.x : p.x + p.s] = p.group

    # One PatternFill per distinct colour instead of one per cell.
    fills = {key: PatternFill(start_color=c, end_color=c, fill_type="solid")
             for key, c in _EXCEL_COLORS.items()}

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "solution"
    for y in range(H):
        ws.row_dimensions[y + 1].height = 18
        for x in range(W):
            ws.cell(row=y + 1, column=x + 1).fill = fills[(group_layer[y, x], int(size_layer[y, x]))]
    for x in range(W):
        ws.column_dimensions[get_column_letter(x + 1)].width = 4
    wb.save(out_path)
    return out_path
