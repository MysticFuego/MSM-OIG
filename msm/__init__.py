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
import warnings
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

from . import solver as _solver
from .config import Island
from .output import draw, export_excel, report, summary
from .solver import Layout, Placement, auto_workers, check, physical_cores, solve

__all__ = ["Island", "Layout", "Placement", "solve", "check", "draw", "export_excel", "report", "summary", "run_all"]


def run_all(islands: dict, *, out_dir: str = ".", parallel: int | None = None, **solve_options) -> dict:
    """
    Solve every island in `islands` (name -> settings), write each one's Excel
    file into out_dir, and print a summary table. Returns name -> Layout, or
    name -> error message for islands that failed (the rest still run). An
    island whose Excel file can't be saved is still returned (with a warning).

    parallel: how many islands are solved at the same time. None = as many as
    this computer's cores fit (cores // workers, e.g. 16 cores / 8 workers = 2).
    The solver works in C++ threads, so islands really do run side by side.
    Interrupting (Ctrl+C / Jupyter's stop button) stops every island at once.
    """
    workers = solve_options.get("workers")
    if workers is not None:
        workers = solve_options["workers"] = _solver._whole(workers, "workers", "at least 1, or None for automatic", 1)
    if parallel is None:
        parallel = max(1, physical_cores() // (workers or auto_workers()))
    parallel = _solver._whole(parallel, "parallel", "at least 1, or None for automatic", 1)
    parallel = min(parallel, max(1, len(islands)))
    if parallel > 1 and workers is None:
        # side by side, each island keeps to its share (no extra packing workers)
        solve_options["workers"] = auto_workers()

    def one(name, settings):
        try:
            island = Island.from_dict(name, settings)
            layout = solve(island, **solve_options)
        except _solver.Cancelled:
            return "stopped"
        except Exception as exc:  # keep going; report it in the table
            return f"{type(exc).__name__}: {exc}"
        try:
            path = os.path.join(out_dir, island.excel_path)  # an absolute "excel" path wins
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            export_excel(layout, path)
        except Exception as exc:  # the layout is still good: hand it back
            warnings.warn(f"{name}: solved, but the Excel file could not be saved"
                          f" ({type(exc).__name__}: {exc}).", stacklevel=2)
        return layout

    if parallel > 1:
        print(f"Solving {parallel} islands at a time, {solve_options['workers']} workers each.", flush=True)
    results = {}
    _solver._reset_cancel()
    pool = ThreadPoolExecutor(max_workers=parallel)
    try:
        futures = {pool.submit(one, name, settings): name for name, settings in islands.items()}
        pending = set(futures)
        while pending:  # short waits, so an interrupt is noticed at once
            done, pending = wait(pending, timeout=0.5, return_when=FIRST_COMPLETED)
            for future in done:
                name = futures[future]
                results[name] = future.result()
                print(summary({name: results[name]}).splitlines()[-1], flush=True)
    except BaseException:
        _solver.cancel()  # stop the islands being solved; drop the ones waiting
        pool.shutdown(wait=True, cancel_futures=True)
        print("Stopped.", flush=True)
        raise
    finally:
        pool.shutdown(wait=True)
        _solver._reset_cancel()
    results = {name: results[name] for name in islands}  # back in the notebook's order
    print()
    print(summary(results))
    return results
