"""
MSM island layout optimizer.

    import msm
    layout = msm.solve(msm.Island.from_dict("Plant", ISLANDS["Plant"]))
    print(msm.report(layout)); msm.draw(layout); msm.export_excel(layout)

    results = msm.run_all(ISLANDS)   # every island, one Excel file each

Modules: grid (CSV mask, mirror axis, in-game height), config (island
settings), solver (the three steps), output (plot, Excel, summaries).
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from .config import Island
from .output import draw, export_excel, report, summary
from .solver import Layout, Placement, auto_workers, check, physical_cores, solve

__all__ = ["Island", "Layout", "Placement", "solve", "check", "draw", "export_excel", "report", "summary", "run_all"]


def run_all(islands: dict, *, out_dir: str = ".", parallel: int | None = None, **solve_options) -> dict:
    """
    Solve every island in `islands` (name -> settings), write each one's Excel
    file into out_dir, and print a summary table. Returns name -> Layout, or
    name -> error message for islands that failed (the rest still run).

    parallel: how many islands are solved at the same time. None = as many as
    this computer's cores fit (cores // workers, e.g. 16 cores / 8 workers = 2).
    The solver works in C++ threads, so islands really do run side by side.
    """
    if solve_options.get("workers") is None:
        solve_options["workers"] = auto_workers()
    if parallel is None:
        parallel = max(1, physical_cores() // int(solve_options["workers"]))

    def one(name, settings):
        try:
            island = Island.from_dict(name, settings)
            layout = solve(island, **solve_options)
            path = os.path.join(out_dir, island.excel_path)  # an absolute "excel" path wins
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            export_excel(layout, path)
            return layout
        except Exception as exc:  # keep going; report it in the table
            return f"{type(exc).__name__}: {exc}"

    if parallel > 1:
        print(f"Solving {min(parallel, len(islands))} islands at a time,"
              f" {solve_options['workers']} workers each.", flush=True)
    results = {}
    with ThreadPoolExecutor(max_workers=max(1, parallel)) as pool:
        futures = {pool.submit(one, name, settings): name for name, settings in islands.items()}
        for future in as_completed(futures):
            name = futures[future]
            results[name] = future.result()
            print(summary({name: results[name]}).splitlines()[-1], flush=True)
    results = {name: results[name] for name in islands}  # back in the notebook's order
    print("\n" + summary(results))
    return results
