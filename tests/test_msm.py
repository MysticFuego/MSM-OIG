"""
Checks for the msm package on a small made-up island (fast).
Run from the project folder:   python -m unittest -v
"""

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

    def test_excel_export(self):
        layout = self.solve()
        path = msm.export_excel(layout, os.path.join(self.tmp.name, "out.xlsx"))
        from openpyxl import load_workbook
        wb = load_workbook(path)
        self.assertEqual(wb.sheetnames, ["Layout", "Summary"])
        self.assertIn("UP", [c.value for row in wb["Layout"].iter_rows() for c in row if c.value])


if __name__ == "__main__":
    unittest.main()
