import time
from dataclasses import dataclass
from datetime import datetime

import numpy as np
from ortools.sat.python import cp_model


def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


def log_print(enabled: bool, msg: str) -> None:
    if enabled:
        print(f"[{_ts()}] {msg}", flush=True)


@dataclass(frozen=True)
class Placement:
    x: int
    y: int
    s: int
    kind: str
    group: str


def all_valid_top_left(avail: np.ndarray, s: int, log: bool = False, label: str = ""):
    t0 = time.time()
    H, W = avail.shape
    total = max(1, (H - s + 1) * (W - s + 1))

    if H < s or W < s:
        coords = []
    else:
        blocked = (~avail).astype(np.int16)
        sat = np.pad(blocked, ((1, 0), (1, 0)), mode="constant").cumsum(axis=0).cumsum(axis=1)
        window_blocked = sat[s:, s:] - sat[:-s, s:] - sat[s:, :-s] + sat[:-s, :-s]
        ys, xs = np.where(window_blocked == 0)
        coords = list(zip(xs.tolist(), ys.tolist()))

    if log:
        log_print(True, f"   candidates for {label}{s}x{s}: {len(coords)}/{total} windows | {time.time()-t0:.2f}s")
    return coords


def row_gradient_score(s: int, y: int, H: int) -> int:
    if H <= 1:
        return 1

    # Target vertical bands by size:
    # 4x4 -> very top, 3x3 -> upper band, 2x2 -> lower-middle, 1x1 -> bottom.
    target_ratio = {
        4: 0.10,
        3: 0.32,
        2: 0.66,
        1: 0.92,
    }[s]
    size_weight = {
        4: 6,
        3: 4,
        2: 2,
        1: 1,
    }[s]

    center = y + (s - 1) / 2
    target = target_ratio * (H - 1)
    max_distance = max(1.0, H - 1)
    return int(round(size_weight * 1000 * (max_distance - abs(center - target))))


def center_y_for_square(s: int, y: int) -> float:
    return y + (s - 1) / 2


def filter_positions_by_band(
    coords: list[tuple[int, int]],
    *,
    s: int,
    H: int,
    min_ratio: float,
    max_ratio: float,
) -> list[tuple[int, int]]:
    if not coords:
        return []

    min_y = min_ratio * (H - 1)
    max_y = max_ratio * (H - 1)
    return [
        (x, y)
        for (x, y) in coords
        if min_y <= center_y_for_square(s, y) <= max_y
    ]


def build_band_candidates(
    coords: list[tuple[int, int]],
    *,
    s: int,
    H: int,
    required_count: int,
    log: bool,
) -> list[tuple[int, int]]:
    if required_count <= 0:
        return []

    band_targets = {
        4: (0.00, 0.75),
        3: (0.00, 1.00),
        2: (0.00, 1.00),
        1: (0.00, 1.00),
    }
    min_ratio, max_ratio = band_targets[s]
    relax_step = 0.06

    while True:
        filtered = filter_positions_by_band(
            coords,
            s=s,
            H=H,
            min_ratio=min_ratio,
            max_ratio=max_ratio,
        )
        if len(filtered) >= required_count:
            if log:
                log_print(
                    True,
                    f"   band {s}x{s}: {len(filtered)} candidates in [{min_ratio:.2f}, {max_ratio:.2f}]",
                )
            return filtered

        if min_ratio <= 0.0 and max_ratio >= 1.0:
            if log:
                log_print(True, f"   band {s}x{s}: fell back to full height with {len(filtered)} candidates")
            return filtered

        min_ratio = max(0.0, min_ratio - relax_step)
        max_ratio = min(1.0, max_ratio + relax_step)


def solve_full_collection_with_fill(
    avail: np.ndarray,
    structures: dict[int, int],
    monsters: dict[int, int],
    *,
    log: bool = True,
    solver_log: bool = False,
    workers: int = 8,
    max_time_in_seconds: float | None = None,
):
    grand0 = time.time()
    H, W = avail.shape
    avail_cells = int(avail.sum())
    available_positions = [(x, y) for y in range(H) for x in range(W) if avail[y, x]]

    structure_counts = {s: int(structures.get(s, 0)) for s in [1, 2, 3, 4]}
    monster_counts = {s: int(monsters.get(s, 0)) for s in [1, 2, 3, 4]}
    required_totals = {s: structure_counts[s] + monster_counts[s] for s in [1, 2, 3, 4]}

    for label, counts in [("structures", structure_counts), ("monsters", monster_counts)]:
        for s, c in counts.items():
            if c < 0:
                raise ValueError(f"{label}[{s}] must be non-negative.")

    if sum(required_totals[s] * (s * s) for s in [1, 2, 3, 4]) > avail_cells:
        return {"ok": False, "reason": "Impossible: required cells exceed available cells."}

    log_print(log, "==============================")
    log_print(log, "? START: MSM Optimal Layout")
    log_print(log, f"Grid: {W} x {H} | available cells: {avail_cells}")
    log_print(log, f"Structures: {structure_counts}")
    log_print(log, f"Monsters:   {monster_counts}")
    log_print(log, f"Totals:     {required_totals}")
    log_print(log, "==============================")

    t0 = time.time()
    log_print(log, "? [1/6] Building CP-SAT model")
    model = cp_model.CpModel()
    log_print(log, f"? Model created | {time.time()-t0:.2f}s")

    t0 = time.time()
    log_print(log, "? [2/6] Building shared candidate placements")
    raw_size_candidates = {
        s: all_valid_top_left(avail, s, log=log, label="size ")
        for s in [2, 3, 4]
        if required_totals[s] > 0
    }
    size_candidates = {
        s: build_band_candidates(
            raw_size_candidates[s],
            s=s,
            H=H,
            required_count=required_totals[s],
            log=log,
        )
        for s in raw_size_candidates
    }
    log_print(log, f"? Shared candidates built | {time.time()-t0:.2f}s")

    t0 = time.time()
    log_print(log, "? [3/6] Creating mandatory vars by group/size")
    mandatory_vars: dict[tuple[str, int], dict[tuple[int, int], cp_model.IntVar]] = {}
    for group, counts in [("structure", structure_counts), ("monster", monster_counts)]:
        for s in [4, 3, 2]:
            req = counts[s]
            if req == 0:
                continue
            cands = size_candidates.get(s, [])
            if not cands:
                return {"ok": False, "reason": f"No placement exists for required {group} {s}x{s} in this grid."}
            vars_for_group = {
                (x, y): model.NewBoolVar(f"{group}_{s}x{s}_at_{x}_{y}")
                for (x, y) in cands
            }
            mandatory_vars[(group, s)] = vars_for_group
            model.Add(sum(vars_for_group.values()) == req)
            log_print(log, f"   {group} {s}x{s}: {len(vars_for_group)} candidates | required={req}")
    log_print(log, f"? Mandatory vars built | {time.time()-t0:.2f}s")

    t0 = time.time()
    log_print(log, "? [4/6] Creating 1x1 vars and extra 2x2 vars")
    one_by_one_vars = {"structure": {}, "monster": {}}
    banded_positions_1x1 = build_band_candidates(
        available_positions,
        s=1,
        H=H,
        required_count=required_totals[1],
        log=log,
    )
    for group, req in [("structure", structure_counts[1]), ("monster", monster_counts[1])]:
        if req > avail_cells:
            return {"ok": False, "reason": f"Impossible: required {group} 1x1={req} exceeds available cells={avail_cells}."}
        vars_for_group = {
            (x, y): model.NewBoolVar(f"{group}_1x1_{x}_{y}")
            for (x, y) in banded_positions_1x1
        }
        one_by_one_vars[group] = vars_for_group
        model.Add(sum(vars_for_group.values()) == req)
        log_print(log, f"   {group} 1x1 vars: {len(vars_for_group)} | required={req}")

    fill2_cands = all_valid_top_left(avail, 2, log=log, label="extra ")
    fill2_vars = {(x, y): model.NewBoolVar(f"extra_2x2_at_{x}_{y}") for (x, y) in fill2_cands}
    log_print(log, f"? Extra vars built | extra_2x2={len(fill2_vars)} | {time.time()-t0:.2f}s")

    t0 = time.time()
    log_print(log, "? [5/6] Adding non-overlap constraints per available cell")
    cover_terms = [[[] for _ in range(W)] for __ in range(H)]

    for (group, s), vars_for_group in mandatory_vars.items():
        for (tx, ty), v in vars_for_group.items():
            for yy in range(ty, ty + s):
                for xx in range(tx, tx + s):
                    cover_terms[yy][xx].append(v)

    for (tx, ty), v in fill2_vars.items():
        cover_terms[ty][tx].append(v)
        cover_terms[ty][tx + 1].append(v)
        cover_terms[ty + 1][tx].append(v)
        cover_terms[ty + 1][tx + 1].append(v)

    for (x, y) in available_positions:
        per_cell = sum(cover_terms[y][x])
        per_cell += one_by_one_vars["structure"].get((x, y), 0)
        per_cell += one_by_one_vars["monster"].get((x, y), 0)
        model.Add(per_cell <= 1)
    log_print(log, f"? Non-overlap constraints added | {time.time()-t0:.2f}s")

    t0 = time.time()
    log_print(log, "? [6/6] Objective: maximize extra 2x2, then enforce top-to-bottom size gradient")
    primary_obj = sum(fill2_vars.values())
    secondary_terms = []
    secondary_scale = 1000

    for (group, s), vars_for_group in mandatory_vars.items():
        for (x, y), v in vars_for_group.items():
            secondary_terms.append((secondary_scale * row_gradient_score(s, y, H)) * v)

    for group in ["structure", "monster"]:
        for (x, y), v in one_by_one_vars[group].items():
            secondary_terms.append((secondary_scale * row_gradient_score(1, y, H)) * v)

    for (x, y), v in fill2_vars.items():
        secondary_terms.append((secondary_scale * row_gradient_score(2, y, H)) * v)

    max_secondary = (
        sum(len(vars_for_group) for vars_for_group in mandatory_vars.values())
        + structure_counts[1]
        + monster_counts[1]
        + len(fill2_cands)
    ) * secondary_scale * max(1, 6 * max(1, H - 1))
    primary_weight = max_secondary + 1
    model.Maximize(primary_weight * primary_obj + sum(secondary_terms))
    log_print(log, f"? Objective set | {time.time()-t0:.2f}s")

    log_print(log, "? Starting solver")
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = int(workers)
    solver.parameters.log_search_progress = bool(solver_log)
    if max_time_in_seconds is not None:
        solver.parameters.max_time_in_seconds = float(max_time_in_seconds)

    status = solver.Solve(model)
    status_name = solver.StatusName(status)
    log_print(log, f"? Solver finished: {status_name}")

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return {"ok": False, "status": status_name, "reason": f"Status: {status_name}."}

    placements = []
    for (group, s), vars_for_group in mandatory_vars.items():
        for (x, y), v in vars_for_group.items():
            if solver.Value(v) == 1:
                placements.append(Placement(x, y, s, f"{group}_{s}x{s}", group))

    required_one_counts = {"structure": 0, "monster": 0}
    for group in ["structure", "monster"]:
        for (x, y), v in one_by_one_vars[group].items():
            if solver.Value(v) == 1:
                placements.append(Placement(x, y, 1, f"{group}_1x1", group))
                required_one_counts[group] += 1

    extra2 = 0
    for (x, y), v in fill2_vars.items():
        if solver.Value(v) == 1:
            placements.append(Placement(x, y, 2, "extra_2x2", "extra"))
            extra2 += 1

    occupied = np.zeros_like(avail, dtype=bool)
    for p in placements:
        occupied[p.y:p.y + p.s, p.x:p.x + p.s] = True

    extra1 = 0
    for (x, y) in available_positions:
        if not occupied[y, x]:
            placements.append(Placement(x, y, 1, "extra_1x1", "extra"))
            occupied[y, x] = True
            extra1 += 1

    log_print(
        log,
        f"?? DONE | status={status_name} | structure_1x1={required_one_counts['structure']} | "
        f"monster_1x1={required_one_counts['monster']} | extra2={extra2} | extra1={extra1} | "
        f"total {time.time()-grand0:.2f}s",
    )

    objective_value = solver.ObjectiveValue()
    best_bound = solver.BestObjectiveBound()

    return {
        "ok": True,
        "status": status_name,
        "optimal": status == cp_model.OPTIMAL,
        "W": W,
        "H": H,
        "available_cells": avail_cells,
        "structures": structure_counts,
        "monsters": monster_counts,
        "required_totals": required_totals,
        "fill_2x2": extra2,
        "fill_1x1": extra1,
        "objective_value": objective_value,
        "best_objective_bound": best_bound,
        "relative_gap": None if objective_value == 0 else (best_bound - objective_value) / abs(objective_value),
        "solve_wall_time_seconds": solver.WallTime(),
        "placements": placements,
    }
