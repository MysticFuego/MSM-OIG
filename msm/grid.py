"""
The island grid: reading the CSV mask, where squares fit, and the in-game
geometry (mirror axis and which way is "up").

MSM draws the grid as a diamond, so the island's on-screen vertical axis is a
*diagonal* of the CSV grid:
    "anti": axis x + y = c  ->  cell (x, y) mirrors to (c - y, c - x)
    "main": axis x - y = c  ->  cell (x, y) mirrors to (y + c, x - c)
Which diagonal (and c) depends on how the CSV was drawn, so it is detected from
the mask: the axis that maps the most open cells onto open cells wins.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass

import numpy as np

Pos = tuple[int, int]  # (x, y) = (column, row), 0-based

_OPEN = {".", "1", "true", "t", "yes", "y", "o", "avail", "available", "open"}
_BLOCKED = {"#", "0", "false", "f", "no", "n", "", "-", "x", "na", "n/a", "blocked"}


def load_grid(path: str) -> np.ndarray:
    """Read a CSV mask ('.' open, '#' blocked). Returns a bool array, True = open."""
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        rows = [[v.strip() for v in r] for r in csv.reader(f)]
    # Drop only trailing blank lines. A blank row in the middle is a fully
    # blocked row (Excel saves those as ",,,"); removing it would shift every
    # row below it, and with them the mirror axis and the locked cells.
    while rows and not any(rows[-1]):
        rows.pop()
    if not rows:
        raise ValueError(f"{path}: CSV is empty.")
    grid = np.zeros((len(rows), max(len(r) for r in rows)), dtype=bool)
    for y, row in enumerate(rows):
        for x, value in enumerate(row):
            v = value.lower()
            if v in _OPEN:
                grid[y, x] = True
            elif v not in _BLOCKED:
                raise ValueError(f"{path}: unrecognised cell {value!r} at row {y + 1}, column {x + 1}. "
                                 f"Use '.' for open and '#' for blocked.")
    if not grid.any():
        raise ValueError(f"{path}: no open tiles ('.') in the CSV.")
    return grid


def fits(grid: np.ndarray, s: int) -> list[Pos]:
    """Every top-left (x, y) where an s x s square lies entirely on open cells."""
    H, W = grid.shape
    if s > min(H, W):
        return []
    if s == 1:
        ys, xs = np.nonzero(grid)
    else:
        blocked = np.pad((~grid).astype(np.int32), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
        bad = blocked[s:, s:] - blocked[:-s, s:] - blocked[s:, :-s] + blocked[:-s, :-s]
        ys, xs = np.nonzero(bad == 0)
    return list(zip(xs.tolist(), ys.tolist()))


def square(x: int, y: int, s: int):
    """All cells of the s x s square with top-left (x, y)."""
    return ((x + dx, y + dy) for dy in range(s) for dx in range(s))


# --------------------------------------------------------------------------- #
# Excel-style cell names ("A1" = top-left), used for locked pieces and output
# --------------------------------------------------------------------------- #
def cell_name(x: int, y: int) -> str:
    col, n = "", x + 1
    while n:
        n, r = divmod(n - 1, 26)
        col = chr(65 + r) + col
    return f"{col}{y + 1}"


def parse_cell(name: str) -> Pos:
    m = re.fullmatch(r"\s*([A-Za-z]+)\s*(\d+)\s*", name)
    if not m:
        raise ValueError(f"{name!r} is not a cell like 'N3'.")
    x = 0
    for ch in m.group(1).upper():
        x = x * 26 + ord(ch) - 64
    return x - 1, int(m.group(2)) - 1


# --------------------------------------------------------------------------- #
# mirror axis
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Axis:
    kind: str     # "anti" or "main"
    c: int
    matched: int  # open cells whose mirror is also open
    total: int    # open cells

    def mirror_cell(self, x: int, y: int) -> Pos:
        if self.kind == "anti":
            return self.c - y, self.c - x
        return y + self.c, x - self.c

    def mirror_square(self, x: int, y: int, s: int) -> Pos:
        """Top-left corner of the mirror image of the s x s square at (x, y)."""
        ax, ay = self.mirror_cell(x, y)
        bx, by = self.mirror_cell(x + s - 1, y + s - 1)
        return min(ax, bx), min(ay, by)

    def on_axis(self, x: int, y: int) -> bool:
        return self.mirror_cell(x, y) == (x, y)


def find_axis(grid: np.ndarray) -> Axis:
    H, W = grid.shape
    ys, xs = np.nonzero(grid)
    cells = set(zip(xs.tolist(), ys.tolist()))
    best = None
    for kind, cs in (("anti", range(0, H + W)), ("main", range(-H, W + 1))):
        for c in cs:
            ax = Axis(kind, c, 0, len(cells))
            matched = sum(ax.mirror_cell(x, y) in cells for x, y in cells)
            if best is None or matched > best.matched:
                best = Axis(kind, c, matched, len(cells))
    return best


# --------------------------------------------------------------------------- #
# in-game height
# --------------------------------------------------------------------------- #
class Height:
    """
    Distance from the in-game back (top) of the island, measured along the
    mirror axis, so a piece and its mirror image are always equally high.
    By default "up" is the end of the axis nearer grid row 0; flip=True uses
    the other end. Integer half-cell units: piece centres can land on half
    cells, and doubling keeps everything exact.
    """

    def __init__(self, grid: np.ndarray, axis: Axis, flip: bool = False):
        self.axis, self.flip = axis, flip
        self._sign = -1 if flip else 1
        ys, xs = np.nonzero(grid)
        vals = [self._raw(x, y, 1) for x, y in zip(xs.tolist(), ys.tolist())]
        self.lo = min(vals)
        self.span2 = max(1, max(vals) - self.lo)

    def _raw(self, x: int, y: int, s: int) -> int:
        if self.axis.kind == "anti":   # height changes with x - y
            return self._sign * 2 * (y - x)
        return self._sign * (2 * (x + y) + 2 * (s - 1))  # changes with x + y

    def depth2(self, x: int, y: int, s: int) -> int:
        """0 at the in-game back, span2 at the front."""
        return self._raw(x, y, s) - self.lo

    def frac(self, x: int, y: int, s: int) -> float:
        return self.depth2(x, y, s) / self.span2

    def depth_after(self, grid: np.ndarray, n_cells: int) -> float:
        """Height (0..1) where the first n_cells open cells, counted from the back, end."""
        if n_cells <= 0:
            return 0.0
        ys, xs = np.nonzero(grid)
        d = sorted(self.depth2(x, y, 1) for x, y in zip(xs.tolist(), ys.tolist()))
        return d[min(len(d) - 1, n_cells)] / self.span2

    def top_cell(self, grid: np.ndarray) -> Pos:
        """The open cell at the very back, closest to the axis (for the 'up' arrow)."""
        ys, xs = np.nonzero(grid)
        cells = list(zip(xs.tolist(), ys.tolist()))
        best = min(self.depth2(x, y, 1) for x, y in cells)
        top = [p for p in cells if self.depth2(p[0], p[1], 1) == best]
        c = self.axis.c
        off = (lambda p: abs(p[0] + p[1] - c)) if self.axis.kind == "anti" else (lambda p: abs(p[0] - p[1] - c))
        return min(top, key=off)

    def up_vector(self) -> tuple[float, float]:
        """Unit direction (dx, dy) on the grid pointing to the in-game back."""
        v = (0.7071, -0.7071) if self.axis.kind == "anti" else (-0.7071, -0.7071)
        return (-v[0], -v[1]) if self.flip else v
