"""
Checks for the msm package on a small made-up island (fast).
Run from the project folder:   python -m unittest -v
"""

import contextlib
import io
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import msm  # noqa: E402
from msm.grid import find_axis, load_grid  # noqa: E402

FAST = dict(time_limit=None, arrange_time_limit=5, workers=4)


def make_island(folder: str) -> str:
    """12x12 island, mirror-symmetric across the anti-diagonal x + y = 11."""
    path = os.path.join(folder, "test_island.csv")
    with open(path, "w") as f:
        for y in range(12):
            f.write(",".join("." if abs(x - y) <= 5 and 3 <= x + y <= 19 else "#" for x in range(12)) + "\n")
    return path


class MsmTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.csv = make_island(cls.tmp.name)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def island(self, **settings):
        base = {"csv": self.csv, "structures": {3: 2, 1: 2}, "monsters": {2: 4, 1: 3}}
        base.update(settings)
        return msm.Island.from_dict("Test", base)

    def solve(self, **settings):
        layout = msm.solve(self.island(**{k: v for k, v in settings.items() if k != "symmetry"}),
                           symmetry=settings.get("symmetry", True), **FAST)
        self.assertEqual(msm.check(layout), [])
        return layout

    def test_axis_found(self):
        axis = find_axis(load_grid(self.csv))
        self.assertEqual((axis.kind, axis.c), ("anti", 11))
        self.assertEqual(axis.matched, axis.total)

    def test_fill_mode_valid_and_proven(self):
        layout = self.solve()
        self.assertTrue(layout.packing_optimal)
        self.assertGreater(layout.fill, 0)

    def test_symmetry_never_costs_fill(self):
        self.assertEqual(self.solve(symmetry=True).fill, self.solve(symmetry=False).fill)

    def test_income_mode_uses_exact_beds_and_best_mix(self):
        catalog = [{"name": "Big", "size": 2, "income": 10, "beds": 1},
                   {"name": "Small", "size": 1, "income": 3, "beds": 1}]
        layout = self.solve(catalog=catalog, beds=5)
        self.assertEqual(layout.beds_used, 5)
        self.assertEqual(layout.mix, {"Big": 5, "Small": 0})  # 2x2s earn more and 5 of them fit
        self.assertEqual(layout.income, 50)
        self.assertTrue(layout.packing_optimal)

    def test_symmetry_never_costs_income(self):
        catalog = [{"name": "Big", "size": 2, "income": 10}, {"name": "Small", "size": 1, "income": 3}]
        with_sym = self.solve(catalog=catalog, beds=6, symmetry=True)
        without = self.solve(catalog=catalog, beds=6, symmetry=False)
        self.assertEqual(with_sym.income, without.income)

    def test_impossible_beds_fail_clearly(self):
        catalog = [{"name": "Big", "size": 2, "income": 10, "beds": 2}]
        with self.assertRaisesRegex(RuntimeError, "beds"):
            self.solve(catalog=catalog, beds=5)  # 2 beds each can't add up to 5

    def test_locked_pieces_stay_put(self):
        catalog = [{"name": "Big", "size": 2, "income": 10}]
        layout = self.solve(catalog=catalog, beds=3, locked=[
            {"what": "structure", "size": 3, "at": "E5"},
            {"what": "Big", "at": "H8"},
            {"what": "collection", "size": 1, "at": "D6"},
        ])
        placed = {(p.kind, msm.output.cell_name(p.x, p.y)) for p in layout.placements if p.locked}
        self.assertEqual(placed, {("structure 3x3", "E5"), ("Big", "H8"), ("collection 1x1", "D6")})

    def test_named_structures_mirror_shape_then_type(self):
        structures = [("Bakery", 2, 3), ("Hotel", 2, 1), ("Statue", 1, 2)]
        layout = self.solve(structures=structures, monsters={2: 3, 1: 2})
        at = {(p.x, p.y, p.s): p.kind for p in layout.placements if p.group == "structure"}
        mirrored = same = 0
        for (x, y, s), kind in at.items():
            partner = at.get((*layout.axis.mirror_square(x, y, s), s))
            mirrored += partner is not None
            same += partner == kind
        self.assertEqual(mirrored, layout.mirrored)
        self.assertEqual(mirrored, 6)  # every shape mirrored, as with plain sizes
        self.assertEqual(same, layout.same_type)
        unnamed = self.solve(structures={2: 4, 1: 2}, monsters={2: 3, 1: 2})
        self.assertEqual(layout.mirrored, unnamed.mirrored)  # names never cost symmetry

    def test_structures_sit_behind_the_collection(self):
        layout = self.solve()
        depth = {g: [layout.height.frac(p.x, p.y, p.s) for p in layout.placements if p.group == g]
                 for g in ("structure", "collection")}
        self.assertLess(sum(depth["structure"]) / len(depth["structure"]),
                        sum(depth["collection"]) / len(depth["collection"]))

    def test_bad_settings_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "unknown setting"):
            msm.Island.from_dict("Test", {"csv": self.csv, "monstres": {}})
        with self.assertRaisesRegex(ValueError, "open tiles"):
            self.solve(locked=[{"what": "structure", "size": 3, "at": "A1"}])  # A1 is blocked

    # ---- regressions for fixed bugs ------------------------------------- #
    def test_symmetry_uses_spots_outside_the_back_band(self):
        # With an empty back band, symmetry used to depend on where step 1
        # happened to put the structures, and a worse result could still be
        # reported as "proven best". Every spot is now considered.
        from unittest import mock
        with mock.patch.object(msm.solver, "STRUCTURE_BAND", {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0}):
            for structures in ({4: 1, 2: 2}, {3: 3, 2: 2, 1: 1}):
                layout = self.solve(structures=structures, monsters={1: 2})
                self.assertEqual(layout.mirrored, layout.structures_total)
                self.assertTrue(layout.symmetry_optimal)

    def test_reported_mirrored_matches_final_layout(self):
        layout = self.solve(structures={3: 3, 2: 2, 1: 3})
        self.assertEqual(layout.mirrored, msm.solver.count_mirrored(layout.placements, layout.axis))
        self.assertLessEqual(layout.same_type, layout.mirrored)

    def test_fractional_incomes_are_exact(self):
        catalog = [{"name": "A", "size": 1, "income": 3.4, "beds": 2},
                   {"name": "B", "size": 1, "income": 1.6, "beds": 1}]
        layout = self.solve(catalog=catalog, beds=2)  # rounded, 2xB (2+2) beat A (3)
        self.assertEqual(layout.mix, {"A": 1, "B": 0})
        self.assertAlmostEqual(layout.income, 3.4)

    def test_timeout_is_not_reported_as_impossible(self):
        from unittest import mock
        from ortools.sat.python import cp_model
        real = msm.solver._run

        def no_time(m, *a, **k):
            solver, _, secs = real(m, *a, **k)
            return solver, cp_model.UNKNOWN, secs
        with mock.patch.object(msm.solver, "_run", no_time):
            with self.assertRaisesRegex(RuntimeError, "ran out of time"):
                self.solve()

    def test_zero_time_limit_rejected(self):
        with self.assertRaisesRegex(ValueError, "time_limit"):
            msm.solve(self.island(), time_limit=0)

    def test_workers_picked_for_this_computer(self):
        n = msm.solver.auto_workers()
        self.assertTrue(1 <= n <= msm.solver.AUTO_MAX_WORKERS)
        self.assertLessEqual(n, os.cpu_count() or 1)
        layout = msm.solve(self.island(), time_limit=None, arrange_time_limit=5)  # workers=None
        self.assertEqual(msm.check(layout), [])
        with self.assertRaisesRegex(ValueError, "workers"):
            msm.solve(self.island(), workers=0)

    def test_blank_rows_in_csv_are_kept(self):
        path = os.path.join(self.tmp.name, "blank.csv")
        with open(path, "w") as f:
            f.write(".,.,.\n,,\n.,,.\n\n")
        self.assertEqual(load_grid(path).shape, (3, 3))  # middle row blocked, trailing blank line dropped
        with open(path, "w") as f:
            f.write("#,#\n#,#\n")
        with self.assertRaisesRegex(ValueError, "no open tiles"):
            load_grid(path)

    def test_size_lock_with_named_structures(self):
        named = [("Castle", 3, 1), ("Shop", 3, 1), ("Statue", 1, 2)]
        layout = self.solve(structures=[("Castle", 3, 2), ("Statue", 1)],
                            locked=[{"what": "structure", "size": 3, "at": "E5"}])
        self.assertIn(("Castle", 4, 4), {(p.kind, p.x, p.y) for p in layout.placements if p.locked})
        with self.assertRaisesRegex(ValueError, "lock one by name"):
            self.solve(structures=named, locked=[{"what": "structure", "size": 3, "at": "E5"}])

    def test_unclear_settings_give_clear_errors(self):
        with self.assertRaisesRegex(ValueError, "needs 'what' and 'at'"):
            self.solve(locked=[{"what": "collection", "size": 1}])
        with self.assertRaisesRegex(ValueError, "needs a 'size'"):
            self.solve(locked=[{"what": "structure", "at": "E5"}])
        with self.assertRaisesRegex(ValueError, "cell name"):
            self.solve(locked=[{"what": "structure", "size": 3, "at": (4, 4)}])
        with self.assertRaisesRegex(ValueError, "unknown key"):
            self.solve(catalog=[{"name": "Big", "size": 2, "incom": 10}], beds=2)
        with self.assertWarnsRegex(UserWarning, "ignored"):
            self.solve(beds=3)

    def test_main_diagonal_island(self):
        path = os.path.join(self.tmp.name, "main.csv")
        with open(path, "w") as f:  # symmetric across x = y only
            for y in range(12):
                f.write(",".join("." if abs(x - y) <= 4 and 2 <= x + y <= 15 else "#" for x in range(12)) + "\n")
        axis = find_axis(load_grid(path))
        self.assertEqual((axis.kind, axis.c), ("main", 0))
        layout = self.solve(csv=path, structures={2: 2, 1: 2}, monsters={2: 2, 1: 2})
        self.assertEqual(layout.mirrored, 4)

    def test_check_catches_broken_layouts(self):
        layout = self.solve()
        p = layout.placements[0]
        layout.placements.append(p)  # same piece twice: overlap and a wrong count
        problems = " ".join(msm.check(layout))
        self.assertIn("covered twice", problems)
        self.assertIn("expected", problems)

    # ---- trios: 3 identical monsters side by side ------------------------- #
    def test_trios_labelled_with_swaps(self):
        # collection A sits apart; B and C touch, and so does extra D: the trio
        # becomes B, C, D's spot, and D moves to A's spot (positions unchanged)
        from msm.grid import Height
        from msm.solver import Placement, _finish_trios
        grid = load_grid(self.csv)
        height = Height(grid, find_axis(grid))
        a, b, c = (Placement(x, y, 2, "collection 2x2", "collection") for x, y in ((9, 0), (4, 4), (6, 4)))
        d = Placement(8, 4, 2, "Big", "extra")
        placed, result = _finish_trios([a, b, c, d], {2: 1}, height, workers=2, back={"Big"})
        self.assertEqual(result, {2: (1, 1)})
        self.assertEqual({(p.x, p.y) for p in placed}, {(9, 0), (4, 4), (6, 4), (8, 4)})
        trio = {(p.x, p.y) for p in placed if p.trio == 1}
        self.assertEqual(trio, {(4, 4), (6, 4), (8, 4)})
        self.assertTrue(all(p.kind == "collection 2x2" for p in placed if p.trio))
        self.assertEqual([p.kind for p in placed if (p.x, p.y) == (9, 0)], ["Big"])

    def test_trio_gap_allows_close_but_prefers_touching(self):
        from msm.grid import Height
        from msm.solver import Placement, _finish_trios
        grid = load_grid(self.csv)
        height = Height(grid, find_axis(grid))
        spaced = [Placement(x, 5, 1, "collection 1x1", "collection") for x in (2, 4, 6)]  # 1 tile apart
        self.assertEqual(_finish_trios(spaced, {1: 1}, height, 2, gap=0)[1], {1: (0, 1)})
        self.assertEqual(_finish_trios(spaced, {1: 1}, height, 2, gap=1)[1], {1: (1, 1)})
        # a touching trio (with a swap) beats a spaced one (without)
        coll = [Placement(x, 0, 2, "collection 2x2", "collection") for x in (0, 3, 6)]
        extra = [Placement(x, 2, 2, "Big", "extra") for x in (0, 2)]  # (0,0), (0,2), (2,2) touch
        placed, _ = _finish_trios(coll + extra, {2: 1}, height, 2, back={"Big"}, gap=1)
        trio = sorted((p.x, p.y) for p in placed if p.trio)
        self.assertEqual(trio, [(0, 0), (0, 2), (2, 2)])

    def test_trios_in_a_full_solve(self):
        catalog = [{"name": "Big", "size": 2, "income": 10}]
        layout = self.solve(catalog=catalog, beds=2, structures={1: 2}, monsters={3: 3, 2: 6, 1: 3},
                            trios={3: 1, 2: 2, 1: 1})
        plain = self.solve(catalog=catalog, beds=2, structures={1: 2}, monsters={3: 3, 2: 6, 1: 3})
        self.assertEqual((layout.income, layout.mix), (plain.income, plain.mix))  # never costs income
        self.assertEqual({s: w for s, (f, w) in layout.trios.items()}, {3: 1, 2: 2, 1: 1})
        for s, (formed, _) in layout.trios.items():
            self.assertEqual(formed, len({p.trio for p in layout.placements if p.trio and p.s == s}))
        # check() (run by self.solve) verifies trios; a moved member must be caught
        if any(p.trio for p in layout.placements):
            bad = list(layout.placements)
            i = next(i for i, p in enumerate(bad) if p.trio)
            bad[i] = msm.Placement(bad[i].x + 20, bad[i].y, bad[i].s, bad[i].kind, bad[i].group, False, bad[i].trio)
            broken = msm.Layout(**{**layout.__dict__, "placements": bad})
            self.assertTrue(any(f"trio {bad[i].trio}" in problem for problem in msm.check(broken)))

    def test_trios_plus_individuals_is_the_whole_collection(self):
        def collection(**settings):
            return sorted((k.size, k.count) for k in msm.Island.from_dict("T", settings).kinds(100)
                          if k.group == "collection")
        whole = collection(monsters={4: 3, 2: 14 * 3 + 10, 1: 3 * 3 + 14}, trios={2: 14, 1: 3})
        split = collection(trios={2: 14, 1: 3}, individuals={4: 3, 2: 10, 1: 14})
        self.assertEqual(whole, split)
        self.assertEqual(split, [(1, 23), (2, 52), (4, 3)])
        self.assertEqual(collection(trios={3: 2}), [(3, 6)])  # trios alone
        with self.assertRaisesRegex(ValueError, "not both"):
            collection(monsters={2: 4}, individuals={2: 1})
        layout = self.solve(monsters={}, trios={2: 1}, individuals={2: 1, 1: 3})
        self.assertEqual(sum(p.s == 2 and p.group == "collection" for p in layout.placements), 4)

    def test_too_many_trios_rejected(self):
        with self.assertRaisesRegex(ValueError, "trios of 2x2 need 6"):
            self.solve(monsters={2: 4}, trios={2: 2})

    def test_excel_open_elsewhere_saves_a_copy(self):
        from unittest import mock
        from openpyxl import Workbook
        layout = self.solve()
        path = os.path.join(self.tmp.name, "locked.xlsx")
        real = Workbook.save

        def save(wb, filename):
            if filename == path:  # as if Excel had it open
                raise PermissionError(13, "Permission denied", filename)
            return real(wb, filename)
        with mock.patch.object(Workbook, "save", save):
            with self.assertWarnsRegex(UserWarning, "open in another program"):
                saved = msm.export_excel(layout, path)
        self.assertEqual(os.path.basename(saved), "locked (1).xlsx")
        self.assertTrue(os.path.exists(saved))

    def test_run_all_in_parallel_keeps_order_and_failures(self):
        islands = {"A": {"csv": self.csv, "structures": {3: 2}, "monsters": {2: 2}},
                   "Broken": {"csv": self.csv, "monstres": {}},
                   "B": {"csv": self.csv, "structures": {2: 2, 1: 2}, "monsters": {1: 3}}}
        with contextlib.redirect_stdout(io.StringIO()):
            results = msm.run_all(islands, out_dir=self.tmp.name, parallel=2, workers=2,
                                  time_limit=None, arrange_time_limit=5)
        self.assertEqual(list(results), ["A", "Broken", "B"])
        self.assertIsInstance(results["A"], msm.Layout)
        self.assertIsInstance(results["B"], msm.Layout)
        self.assertIn("unknown setting", results["Broken"])
        self.assertTrue(os.path.exists(os.path.join(self.tmp.name, "A.xlsx")))

    def test_run_all_keeps_layout_when_excel_fails(self):
        from unittest import mock
        islands = {"A": {"csv": self.csv, "structures": {3: 2}, "monsters": {2: 2}}}
        failing = mock.patch.object(msm, "export_excel", side_effect=OSError("disk full"))
        with failing, contextlib.redirect_stdout(io.StringIO()), self.assertWarnsRegex(UserWarning, "disk full"):
            results = msm.run_all(islands, out_dir=self.tmp.name, parallel=1, workers=2,
                                  time_limit=None, arrange_time_limit=5)
        self.assertIsInstance(results["A"], msm.Layout)

    def test_run_all_rejects_bad_counts(self):
        with self.assertRaisesRegex(ValueError, "workers"):
            msm.run_all({}, workers=0)
        with self.assertRaisesRegex(ValueError, "parallel"):
            msm.run_all({}, parallel=0, workers=2)

    def test_run_all_can_be_interrupted(self):
        import time
        from unittest import mock
        islands = {n: {"csv": self.csv, "structures": {3: 2}, "monsters": {2: 2}} for n in "ABCD"}
        real = msm.wait

        def interrupt(*a, **k):  # as if the user pressed stop while the first islands run
            real(*a, **k)
            raise KeyboardInterrupt
        t = time.time()
        with mock.patch.object(msm, "wait", interrupt), contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(KeyboardInterrupt):
                msm.run_all(islands, out_dir=self.tmp.name, parallel=1, workers=2, time_limit=None,
                            arrange_time_limit=60)
        self.assertLess(time.time() - t, 20)
        self.assertFalse(msm.solver._cancel.is_set())  # the next run starts clean
        self.assertEqual(msm.check(self.solve()), [])

    def test_cancel_stops_a_running_solve(self):
        import threading
        from unittest import mock
        real = msm.solver._run

        def run(m, *a, **k):  # cancel from another thread as soon as the arrange step starts
            if k.get("patience"):
                threading.Thread(target=msm.solver.cancel).start()
                msm.solver._cancel.wait(5)
            return real(m, *a, **k)
        try:
            with mock.patch.object(msm.solver, "_run", run):
                with self.assertRaises(msm.solver.Cancelled):
                    msm.solve(self.island(), time_limit=None, arrange_time_limit=60, workers=2)
        finally:
            msm.solver._reset_cancel()

    def test_explicit_workers_used_for_packing(self):
        from unittest import mock
        real, seen = msm.solver._pack, []

        def pack(*a):
            seen.append(a[-1])
            return real(*a)
        with mock.patch.object(msm.solver, "_pack", pack):
            msm.solve(self.island(), **FAST)  # workers=4
        self.assertEqual(seen, [4])

    def test_more_settings_give_clear_errors(self):
        with self.assertRaisesRegex(ValueError, "whole numbers"):
            self.solve(monsters={2: 6}, trios={2: None})
        with self.assertRaisesRegex(ValueError, "whole numbers"):
            self.solve(monsters={2: 2.5})
        with self.assertRaisesRegex(ValueError, "fill squares can't be locked"):
            self.solve(locked=[{"what": "fill 2x2", "at": "E5"}])
        with self.assertRaisesRegex(ValueError, "unknown key"):
            self.solve(structures=[{"name": "Castle", "size": 3, "cout": 2}])
        with self.assertRaisesRegex(ValueError, "trio_gap"):
            msm.solve(self.island(), trio_gap=1.5)
        self.assertEqual(self.island(monsters={"2": "4"}).kinds(100)[-2].count, 4)  # text numbers still work

    def test_check_reports_a_short_trio(self):
        layout = self.solve()
        i = next(i for i, p in enumerate(layout.placements) if p.group == "collection")
        layout.placements[i] = msm.Placement(*[getattr(layout.placements[i], f) for f in
                                               ("x", "y", "s", "kind", "group", "locked")], trio=7)
        self.assertTrue(any("trio 7" in problem for problem in msm.check(layout)))  # no crash

    def test_locked_collection_monster_can_be_in_a_trio(self):
        from msm.grid import Height
        from msm.solver import Placement, _finish_trios
        grid = load_grid(self.csv)
        height = Height(grid, find_axis(grid))
        trio = [Placement(x, 5, 1, "collection 1x1", "collection", locked=(x == 3)) for x in (2, 3, 4)]
        lone = Placement(9, 0, 1, "collection 1x1", "collection", locked=True)
        extra = Placement(9, 2, 1, "Small", "extra")
        placed, result = _finish_trios(trio + [lone, extra], {1: 1}, height, 2, back={"Small"})
        self.assertEqual(result, {1: (1, 1)})
        self.assertEqual(sorted((p.x, p.y) for p in placed if p.trio), [(2, 5), (3, 5), (4, 5)])
        self.assertTrue(all(p.kind == "collection 1x1" for p in placed if p.locked))  # locked kept their kind
        # locked monster at (2,5) next to two extras: that trio needs 2 swaps, but only one
        # unlocked monster can trade places (the other locked one can't), so no trio
        pieces = [Placement(2, 5, 1, "collection 1x1", "collection", locked=True),
                  Placement(9, 9, 1, "collection 1x1", "collection", locked=True),
                  Placement(6, 9, 1, "collection 1x1", "collection"),
                  Placement(3, 5, 1, "Small", "extra"), Placement(4, 5, 1, "Small", "extra")]
        placed, result = _finish_trios(pieces, {1: 1}, height, 2, back={"Small"})
        self.assertEqual(result, {1: (0, 1)})
        self.assertEqual(sorted(p.kind for p in placed), sorted(p.kind for p in pieces))  # nothing lost

    def test_fully_mirrored_arrange_ties_pairs(self):
        from unittest import mock
        real, gaps = msm.solver._run, []

        def run(m, *a, **k):
            if k.get("patience"):
                gaps.append(k.get("gap"))
            return real(m, *a, **k)
        with mock.patch.object(msm.solver, "_run", run):
            layout = self.solve()
        self.assertEqual(layout.mirrored, layout.structures_total)
        self.assertTrue(layout.symmetry_optimal)
        self.assertLessEqual(layout.mirrored_bound, layout.structures_total)
        self.assertEqual(gaps, [msm.solver.ARRANGE_GAP_MIRRORED])

    def test_excel_export(self):
        layout = self.solve()
        path = msm.export_excel(layout, os.path.join(self.tmp.name, "out.xlsx"))
        from openpyxl import load_workbook
        wb = load_workbook(path)
        self.assertEqual(wb.sheetnames, ["Layout", "Summary"])
        self.assertIn("UP", [c.value for row in wb["Layout"].iter_rows() for c in row if c.value])


if __name__ == "__main__":
    unittest.main()
