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

from .config import Island
from .output import draw, export_excel, report, summary
from .solver import Layout, Placement, check, solve

__all__ = ["Island", "Layout", "Placement", "solve", "check", "draw", "export_excel", "report", "summary", "run_all"]


def run_all(islands: dict, *, out_dir: str = ".", **solve_options) -> dict:
    """
    Solve every island in `islands` (name -> settings), write each one's Excel
    file into out_dir, and print a summary table. Returns name -> Layout, or
    name -> error message for islands that failed (the rest still run).
    """
    results = {}
    for name, settings in islands.items():
        try:
            island = Island.from_dict(name, settings)
            layout = solve(island, **solve_options)
            export_excel(layout, os.path.join(out_dir, os.path.basename(island.excel_path)))
            results[name] = layout
        except Exception as exc:  # keep going; report it in the table
            results[name] = f"{type(exc).__name__}: {exc}"
        print(summary({name: results[name]}).splitlines()[-1], flush=True)
    print("\n" + summary(results))
    return results
